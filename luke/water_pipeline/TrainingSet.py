import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '2'
# os.environ['OMP_NUM_THREADS'] = '2'
# os.environ['MKL_NUM_THREADS'] = '2'
# os.environ['NUMEXPR_NUM_THREADS'] = '4'
from TrainingSequence import TrainingSequence
from Classifiers import *
from matplotlib import pyplot as plt
from typing import List, Tuple
import numpy as np
# import cupy as np
import cv2
import mydia


class TrainingSet:

    """
    A class to keep track of TrainingVideos which will also incorporate training and testing functions which can be
    applied to the whole training set.
    """

    TEXTURE_BLOCKSIZE = 10

    @staticmethod
    def init_from_folder(vid_folder: str, mask_folder: str, vwd_mode=True) -> 'TrainingSet':
        """
        Load many videos from a video folder and a mask folder. These folders should be in the same root folder. This
        method can load videos and masks in those folders even if the directory structure is hierarchical (folders in
        folders), but the only conditions with this is that the hierarchy must be mirrored between the videos folder and
        the masks folder. VWD mode is for compatibility with the VideoWaterDatabase which does not store masks for the
        videos which are non-water, so with this mode selected, all videos which are in the directory 'non-water' are
        automatically considered non-water TrainingSequences.
        """

        # Find video files by walking through the video folder
        VIDEO_FORMATS = ['.avi', '.mp4']
        files = [os.path.abspath(os.path.join(dp, f))
                 for dp, dn, filenames in os.walk(vid_folder)
                 for f in filenames
                 if os.path.splitext(f)[1] in VIDEO_FORMATS]

        out = []
        for file in files:
            # Since we are assuming the masks folder has the same folder layout and base names as the videos, we can just
            # extract the relative path within the videos folder and append that to the masks folder to get a corresponding
            # path for the mask that goes with this video.
            vid_rel_ext = os.path.relpath(os.path.splitext(file)[0], vid_folder)
            rel_path_ext = os.path.join(mask_folder, vid_rel_ext)

            if vwd_mode and rel_path_ext.count('non_water') == 0:
                mask_path = os.path.join(mask_folder, rel_path_ext) + '.png'
                all_non_water = False
            else:
                mask_path = None
                all_non_water = True

            name = os.path.splitext(os.path.basename(file))[0]
            out.append(TrainingSequence(file, name=name, water_mask_path=mask_path, all_non_water=all_non_water))

        return TrainingSet(out)

    def __init__(self, vids: List[TrainingSequence]):
        # Stored data
        self.vids = vids

        # Classifiers
        self.texture_classifier = TextureClassifierLBP()
        self.color_ggm = ColorClassifierGMM()
        self.temporal_classifier = TemporalClassifierFFT()
        self.combine_classifier = CombinationClassifier()
        self.tt_classifier = TextureTemporalClassifier()
        self.classifiers = [self.texture_classifier, self.color_ggm, self.temporal_classifier, self.tt_classifier]

    # FPS-based frame selection for mydia video reader
    class ModeFunctionFactory:
        def __init__(self, fps):
            self.fps = fps

        def __call__(self, *args, **kwargs):
            indices = np.arange(args[0], dtype=np.uint8)
            return (indices * self.fps)[:args[1]]

    def load(self, in_color=False, dims=None, num_frames=None, sampling_rate=1, only_category=None) -> np.ndarray:
        """
        Loads some/all of the frames contained in this TrainingSet.

        :param in_color: Whether to load the frames in color or in grayscale
        :param dims: Frames will be resized to this dimension (x, y) if specified
        :param num_frames: The number of frames to load
        :param sampling_rate: The sampling rate by which frames are taken from the video. Ex., 1 means keep the normal
        framerate, and 2 means split the framerate in half. What this option is effectively doing is only storing
        a frame from a given video once every "sampling_rate" number of frames has passed.
        :param only_category: Load only this category of TrainingSequence, currently: Water, No-water, Both, Unlabeled

        :return: An ndarray which will be in the shape of (# videos, num_frames, y-size, x-size, [HSV channels if specified])
        """

        print('Loading %d training sequences' % len(self.vids))
        all_x = mydia.Videos(target_size=dims,
                             to_gray=(not in_color),
                             num_frames=num_frames,
                             mode=self.ModeFunctionFactory(sampling_rate)).read(self.get_paths(category=only_category),
                                                                                workers=N_PROCESSES)

        # Collapse the last dimension provided by mydia if it is just for the grayscale channel
        if all_x.shape[-1] == 1:
            all_x = all_x[..., -1]

        return all_x

    def get_vid_by_category(self, category=None) -> List[TrainingSequence]:
        """
        Get TrainingSequence's of this specific category.

        :param category: A labeling category specified by TrainingSequence, currently: Water, No-water, Both, or Unlabeled
        :return: A list of TrainingSequences
        """
        return [vid for vid in self.vids if vid.category == category]

    def get_paths(self, category=None) -> List[str]:
        """
        Get the paths contained in this TrainingSet object

        :return: String paths in this TrainingSet object
        """
        if category is None:
            return [vid.input_sequence for vid in self.vids]
        else:
            return [vid.input_sequence for vid in self.vids if category == vid.category]

    def get_masks(self) -> np.ndarray:
        """
        Gets water masks from this TrainingSet.

        (NOTE: Should return an ndarray with a shape of length 3: vid #, y-size, x-size. If there is one None in the
         masks array then numpy will store the masks as objects, which could be problematic.)
        """
        masks = [vid.get_water_mask() for vid in self.vids]
        return np.array(masks, dtype=bool)

    def add(self, vid: TrainingSequence) -> None:
        """
        Add a TrainingSequence to this set.

        :param vid: New TrainingSequence
        """
        self.vids.append(vid)

    def visualize_classifiers(self, train_sample: np.ndarray, prob_mode=True, output='temp.png', show=True):
        """
        # TODO
        :param train_sample:
        :param prob_mode:
        :param output:
        :param show:
        :return:
        """

        plt.clf()
        n_vids, frame_n, y, x = train_sample.shape[:4]
        canvas_img = np.zeros((n_vids * y, (len(self.classifiers) + 2) * x, 3), dtype=np.uint8)

        # Fill in classifier predictions
        all_segmentations = []
        for j, classifier in enumerate(self.classifiers):
            predictions = classifier.segment(train_sample, prob_mode=prob_mode)
            for p, pred in enumerate(predictions):
                canvas_img[p*y:p*y+y, (j+1)*x:(j+2)*x] = cv2.cvtColor((pred*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            all_segmentations.append(predictions)

        # Fill in example images and combination images
        all_segmentations = np.moveaxis(np.array(all_segmentations), 0, -1)  # (# vids, y-size, x-size, # classifiers)
        for n_vid in range(n_vids):
            # Show gray/HSV original image
            if len(train_sample.shape) == 4:
                canvas_img[n_vid*y:n_vid*y+y, 0:x] = cv2.cvtColor(train_sample[n_vid, 0], cv2.COLOR_GRAY2RGB)
            else:
                canvas_img[n_vid * y:n_vid * y + y, 0:x] = cv2.cvtColor(train_sample[n_vid, 0], cv2.COLOR_HSV2RGB)

            this_vid_predictions = all_segmentations[n_vid][np.newaxis, ...]
            img = self.combine_classifier.segment(this_vid_predictions)
            if img is not None:
                canvas_img[n_vid*y:n_vid*y+y, -x:] = cv2.cvtColor((img[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        cv2.imwrite(output, canvas_img)
        # plt.imshow(canvas_img, interpolation='nearest')
        # plt.title('Example images classified by various classifiers')
        # plt.tight_layout()
        # plt.savefig(output, dpi=500)

    def visualize(self, classifier: Classifier, input_x: np.ndarray, show_img=True, output_img='output.png',
                  prob_mode=True) -> None:
        """
        Visualize the segmentation results of a given classifier by generating a large original-and-segmentation-result
        comparison image. This image is good for debugging a classifier and it displays the original image on the left
        column and the segmentation result on the right column.

        :param classifier: The input classifier of type 'Classifier'
        :param input_x: The input x-set to segment, in the shape of (#vids, #frames, y-size, x-size, [HSV])
        :param prob_mode: Whether to segment with probability mode or binary classification mode
        :param output_img: The filename to output the generated image
        :param show_img: Whether to show the image graphically
        """

        # Grab some example images from this collection
        plt.clf()
        canvas_img = np.zeros(())



        # for i in range(input_x):
        #     vids[i].load_frames(max_n=video_n_frames)
        #     if vids[i].img_gray_ar is None:
        #         print('Unable to load images')
        #         continue
        #     frame_seq = vids[i].img_gray_ar[:video_n_frames]
        #     if len(frame_seq) == 0:
        #         print('Unable to load images: frame sequence is of length 0')
        #         continue
        #
        #     if canvas_img is None:
        #         canvas_img = np.zeros((frame_seq.shape[1]*n_imgs, frame_seq.shape[2]*2))
        #
        #     y, x = frame_seq.shape[1:]
        #     canvas_img[i*y:i*y+y, :x] = frame_seq[0]
        #     img = classifier.segment(frame_seq[np.newaxis, ...], prob_mode=True)[0]*255
        #     canvas_img[i*y:i*y+y, x:2*x] = img

        plt.imshow(canvas_img)
        plt.suptitle('Example images classified by various classifiers')
        plt.tight_layout()

        if output_img is not None:
            print('Writing to %s' % output_img)
            plt.savefig(output_img, dpi=250)

        if show_img:
            plt.show()

    def train_all(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train every classifier attached to this TrainingSet.
        """

        for classifier in self.classifiers:
            print('[TRAIN] Training classifier named (%s)' % classifier.name)
            classifier.train(x, y)

    def test_all(self, x: np.ndarray, y: np.ndarray) -> list:
        """
        Test every classifier attached to this TrainingSet.
        """
        return [classifier.test(x, y) for classifier in self.classifiers]

    def segment_all(self, x: np.ndarray, prob_mode=True):
        """
        Segment video sequences with multiple trained classifiers

        :param x: Input shape of (# vids, # frames, y-size, x-size, HSV channels) as int8
        :param prob_mode: Whether to generate probability maps
        :return: Results in shape of (# vids, y-size, x-size, # classifiers) as float32
        """

        if len(x.shape) != 5:
            raise RuntimeError('Segmentation input must have 5 dimensions')

        n_vids, n_frames, img_y, img_x, hsv = x.shape
        all_segmentations = np.zeros((n_vids, img_y, img_x, len(self.classifiers)), dtype=np.float32)
        for j, classifier in enumerate(self.classifiers):
            all_segmentations[..., j] = classifier.segment(x, prob_mode=prob_mode)

        return all_segmentations
