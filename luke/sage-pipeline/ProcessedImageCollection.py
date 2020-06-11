import pickle
import os.path
from typing import Optional
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from LocalImageClasses import *
from BackgroundSubtractor import BackgroundSubtractor
import json


class ProcessedImageCollection(LocalImageCollection):

    """
    A class that extends the LocalImageCollection with computational features.
    """

    N_BINS = 15
    ALL_IMAGES = -1

    def __init__(self, root_folder, name):
        super().__init__(root_folder, name)
        self.brightnesses = None
        # Classification
        self.bgsub_day = BackgroundSubtractor('day')  # If there is only one peak in the brightness data, use this only
        self.bgsub_night = BackgroundSubtractor('night')  # Not used in the case of one brightness peak
        self._checkpoint_bg_frame_day: Optional[np.ndarray] = None
        self._checkpoint_bg_frame_night: Optional[np.ndarray] = None
        # Time of day data
        self.day_brightness = None
        self.night_brightness = None
        self.middle_brightness = None

    def get_brightnesses(self, max_n=None, output=True) -> List[int]:
        imgs = self.get_all_images()[:max_n]
        if self.brightnesses is None:
            if output:
                print('%s Computing brightness values for %d images' % (self.CLASS_TAG, len(self.local_images)))
            self.brightnesses = [img.get_brightness() for img in imgs]
        if output:
            print('%s Finished computing brightness values' % self.CLASS_TAG)
        return self.brightnesses

    def _compute_day_and_night_peaks(self, sample_n=None, output=True, save_after_compute=False):
        """
        Finds the average brightness values of images taken in the day time and images taken in the night time,
        respectively. Used to probabilistically tag images as being day images or night images.
        """

        brightnesses = self.get_brightnesses(max_n=sample_n, output=output)
        if save_after_compute:
            self.save()

        # Find the peaks of the brightness histogram, which are usually the average day and night brightnesses
        (n_in_bins, bins, patches) = plt.hist(brightnesses, bins=self.N_BINS)
        plt.clf()
        avg_count = sum(n_in_bins) / len(n_in_bins)
        n_in_bins = np.insert(n_in_bins, [0, len(n_in_bins)], [avg_count, avg_count])
        n_in_bins_top_4 = min(sorted(n_in_bins)[-4:])
        peaks, _ = find_peaks(n_in_bins, distance=3, height=n_in_bins_top_4)

        # Compute the values of the middle of the peaks
        brightnesses_of_peaks = [(bins[peak] + bins[peak+1])/2 for peak in peaks]

        # A warning
        if len(peaks) > 2:
            print('[!!] %s Brightness binning algorithm found more than 2 brightness peaks in (%s)' %
                  (self.CLASS_TAG, self.root_folder))

        if len(peaks) == 1:
            peaks = brightnesses_of_peaks
        else:
            # Pick the peaks that are as far away from each other as possible
            peaks = [min(brightnesses_of_peaks), max(brightnesses_of_peaks)]

        if output:
            print('%s Day and night peaks: %s' % (self.CLASS_TAG, str(peaks)))
        return peaks

    def tag_images_with_tod(self, output=False):
        """
        Uses the day and night peak algorithm to assign the tag 'time_of_day' the value of either day or night for
        all images in this collection.
        """

        # This will take some time because the computation requires the loading and brightness assessment of each image
        peaks = self._compute_day_and_night_peaks(output=output)

        if len(peaks) == 2:
            print('%s Number of day and night peaks: %d' % (self.CLASS_TAG, len(peaks)))
            self.night_brightness, self.day_brightness = peaks
            assert self.day_brightness >= self.night_brightness
            print('%s Day brightness: %d | Night brightness: %d' % (self.CLASS_TAG,
                                                                    self.day_brightness, self.night_brightness))
            self.middle_brightness = (self.day_brightness + self.night_brightness) // 2
        else:
            print('%s There is only one peak in this brightness data' % self.CLASS_TAG)
            self.middle_brightness = self.day_brightness = self.night_brightness = peaks[0]

        self.graph_day_and_night_split()

        for img in self.iter_with(output=output, unloading=True):
            if len(peaks) > 1:
                tod = 'day' if img.get_brightness() > self.middle_brightness else 'night'
            else:
                tod = 'day'  # Tag all the images in a one-peak collection with 'day'
            img.tag('time_of_day', tod)
        print('%s Finished tagging images' % self.CLASS_TAG)

    def _background_subtract(self, img) -> np.ndarray:
        blurred_img = cv2.blur(img.get(), (5, 5))

        tod = img.get_tag('time_of_day')
        if tod == 'day':
            out = self.bgsub_day.get_foreground(blurred_img)
        elif tod == 'night':
            out = self.bgsub_night.get_foreground(blurred_img)
        else:
            raise RuntimeError('%s Image tagged something other than \'day\' or \'night\'' % self.CLASS_TAG)

        return out

    def train_background_subtractors(self, output=True, n_for_each=200) -> None:
        """
        Train this collection's background subtractor(s) with a sample of images. If there is one brightness peak in
        this image collection, only the day subtractor is trained, else both day and night are trained with the same
        amount of images, specified by `n_for_each`.
        """

        # Train the BG subtractor on N images from both day and night
        day_imgs = [img for img in self.local_images if img.get_tag('time_of_day') == 'day']
        night_imgs = [img for img in self.local_images if img.get_tag('time_of_day') == 'night']

        self.bgsub_day.train_on(day_imgs[:n_for_each], output=output)
        self.bgsub_night.train_on(night_imgs[:n_for_each], output=output)

    def tag_images_under_fg_threshold(self, threshold: int):
        """
        Tags images in this collection with tag 'background' and value 'true'. The threshold value should be determined
        by scanning through the processed images to find the lowest threshold possible that still includes images with
        small foreground features, like a single pedestrian. This way, we can confidently throw away boring images while
        retaining most of our interesting, feature-filled images.
        """

        print('[Tagging] Tagging images under the foreground threshold of: %d' % threshold)
        for i, img in enumerate(self.local_images):
            if i % 100 == 0:
                print('[Tagging] Processing image %d/%d' % (i, len(self.local_images)))
            if img.get_tag('fg_count') < threshold:
                img.tag('background', True)

    def tag_images_with_fg_count(self):
        """
        This method tags every image with the foreground count, a metric the script uses to determine the most
        interesting images. Images with a higher foreground count generally are more interesting. Images with 0 or very
        low foreground count are almost always boring background images.
        """

        for i, img in enumerate(self.local_images):
            # Progress updates
            if i % 100 == 0:
                print('[Tagging] Tagged image %d/%d with foreground count data' % (i, len(self.local_images)))

            # Preprocess the foreground mask
            tod_tag = img.get_tag('time_of_day')
            assert tod_tag is not None  # If this is encountered, you need to run TOD tagging method first
            if tod_tag == 'day':
                foreground_mask = self.bgsub_day.get_foreground(img.get())
            else:
                foreground_mask = self.bgsub_night.get_foreground(img.get())
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((5, 5)))
            img.unload()

            # Put together debug text
            fg_only_mask = cv2.threshold(foreground_mask, 128, 255, cv2.THRESH_BINARY)[1]  # Threshold-out the shadows
            fg_count = cv2.countNonZero(fg_only_mask)
            img.tag('fg_count', fg_count)

    def output_metadata_file(self, root_path: str):
        """
        Scans through the collection, finds images tagged as background, and compiles their paths into a structured
        JSON file.
        """

        bg_img_hierarchy = {'background_images': {}}
        bg_img_n = 0
        for img in self.local_images:
            if img.get_tag('background'):
                full_path = img.path
                folders_from_path = os.path.split(os.path.relpath(full_path, root_path))
                assert len(folders_from_path) > 0
                image_name = folders_from_path[-1]
                if len(folders_from_path[:-1]) == 0:
                    hierarchy_folders = '.'
                else:
                    hierarchy_folders = '/'.join(folders_from_path[:-1])

                bg_img_n += 1
                if bg_img_hierarchy['background_images'].get(hierarchy_folders) is None:
                    bg_img_hierarchy['background_images'][hierarchy_folders] = [image_name]
                else:
                    # Fill out part of the list
                    bg_img_hierarchy['background_images'][hierarchy_folders].append(image_name)

        metadata_path = os.path.join(root_path, 'metadata.json')
        print('[JSON Writing] Dumping %d background image paths into a JSON metadata file (%s)' %
              (bg_img_n, metadata_path))
        json_text = json.dumps(bg_img_hierarchy)
        with open(metadata_path, 'w+') as fp:
            fp.write(json_text)

        return json_text

    def debug_background_subtractor(self, output=True, save_fg_images_to: Optional[str] = None, save_n=None):
        """
        Run the OpenCV MOG2 background subtractor over `save_n` images, outputing checkpoint prints if selected,
        and save the foreground images to a specified folder.
        """

        # Assertions
        if save_n is not None:
            assert save_fg_images_to is not None

        if output:
            print('%s Running background subtractor' % self.CLASS_TAG)
        self.tag_images_with_tod(output=output)

        cwd = os.path.abspath('.')
        if save_fg_images_to is not None:
            os.makedirs(os.path.abspath(save_fg_images_to), exist_ok=True)
            os.chdir(os.path.abspath(save_fg_images_to))

        if save_n == self.ALL_IMAGES:
            save_n = None

        # Save images to disk
        print('Tagging fg_count...')
        for i, img in enumerate(self.iter_with(output=output, max_n=save_n)):
            foreground_mask = self._background_subtract(img)

            if save_fg_images_to is not None:
                # Preprocess the foreground mask
                foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((5, 5)))
                # Put together debug text
                fg_only_mask = cv2.threshold(foreground_mask, 128, 255, cv2.THRESH_BINARY)[1]  # Threshold-out the shadows
                # fg_count = cv2.countNonZero(fg_only_mask)
                text = 'Foreground Count: %d' % img.get_tag('fg_count')
                # img.tag('fg_count', fg_count)

                combined_output = stitch_together(fg_only_mask, img.get(), text=text)
                cv2.imwrite(str(i)+'.jpg', combined_output)

        if save_fg_images_to is not None:
            os.chdir(cwd)

    def test_background_subtractor(self):
        """
        Computes the number of day and night-tagged pictures and shows the neural net's internal structure by displaying
        two background images, for the day subtractor and the night subtractor
        """

        day_pics = sum([1 for img in self.local_images if img.get_tag('time_of_day') == 'day'])
        night_pics = len(self.local_images) - day_pics
        print('[TEST] Collection contains %d day pictures and %d night pictures' % (day_pics, night_pics))
        show_img(self.bgsub_day.get_background())
        if night_pics > 0:
            input('press enter to continue')
            show_img(self.bgsub_night.get_background())

    def graph_day_and_night_split(self, max_n=None):
        """
        Graphs a histogram that allows us to visualize how many images in this collection fall under which category,
        either day or night.
        """

        brightnesses = []
        for img in self.iter_with(output=True, max_n=max_n, unloading=True):
            brightnesses.append(img.get_brightness())

        if self.day_brightness != self.night_brightness:
            n, bins, patches = plt.hist(brightnesses, bins=20)

            for i in range(len(patches)):
                if bins[i] < self.middle_brightness < bins[i+1]:
                    patches[i].set_facecolor('b')
                elif bins[i] > self.middle_brightness:
                    patches[i].set_facecolor('y')
                else:
                    patches[i].set_facecolor('black')
            plt.axvline(x=self.day_brightness, color='y', linestyle='dashed')
            plt.axvline(x=self.night_brightness, color='black', linestyle='dashed')
            plt.title('Brightness Histogram for %s (two peaks, two background subtractors)' % self.name)
        else:
            plt.hist(brightnesses, bins=20)
            plt.title('Brightness Histogram for %s (one peak, one background subtractor)' % self.name)

        plt.xlabel('Absolute brightness (sum of V in HSV for each image)')
        plt.ylabel('Frequency (# of images in this collection)')
        plt.ioff()
        plt.show()

    def graph_foreground_counts(self):
        fg_counts = [img.get_tag('fg_count') for img in self.local_images]

        # Offer statistics on the counts
        fg_counts_sorted = sorted(fg_counts)
        ten_percentile = len(fg_counts_sorted)//10
        print('----- Bottom Brightnesses -----\n10%%: %d\n25%%: %d\n50%%: %d\n' %
              (fg_counts_sorted[ten_percentile], fg_counts_sorted[int(ten_percentile*2.5)],
               fg_counts_sorted[int(ten_percentile*5)]))

        # Graph helpful histogram (should look like a power-law distribution)
        plt.hist(fg_counts, bins=25)
        plt.title('Histogram of Foreground Counts for %s (Sum of pixels in each image identified as foreground)' %
                  self.name)
        plt.xlabel('Foreground Counts (in # of pixels)')
        plt.ylabel('Frequency (in # of images)')
        plt.show()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        if state.get('bgsub_day') is None:
            print('[!!] [%s] No day background subtractor saved' % state.get('name'))
            state['bgsub_day'] = BackgroundSubtractor('day')
        if state.get('bgsub_night') is None:
            print('[!!] [%s] No night background subtractor saved' % state.get('name'))
            state['bgsub_night'] = BackgroundSubtractor('night')
        self.__dict__ = state

    def save(self):
        print('%s Saving collection (%s)' % (self.CLASS_TAG, self.root_folder))
        pickle.dump(self, open(self.name + '.state', 'wb+'))

    @staticmethod
    def restore(path: str):
        try:
            obj = pickle.load(open(path, 'rb'))
        except IOError:
            return None
        return obj

    @staticmethod
    def load_many_from_folder(folder_of_collections: str):
        items = [item for item in os.listdir(folder_of_collections)
                 if item.count('001e06') > 0 and os.path.isdir(os.path.join(folder_of_collections, item))]
        return [ProcessedImageCollection(os.path.join(folder_of_collections, item), item) for item in items]

    @staticmethod
    def restore_many_from_folder(folder_of_states: str):
        items = [item for item in os.listdir(folder_of_states) if item.count('.state') > 0]
        return [ProcessedImageCollection.restore(os.path.join(folder_of_states, item)) for item in items]
