import os.path
from typing import List
from ImageManipulationUtils import *
import imutils
from imutils.object_detection import non_max_suppression
from pathlib import Path


# Image processing constants
RESIZED_IMAGE_WIDTH = 400
# Initialize pedestrian detector
hog_ped_detector = cv2.HOGDescriptor()
hog_ped_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class LocalImage:

    """
    A class for keeping track of local images.
    """

    def __init__(self, path: str):
        self.path = path
        self.data = None

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None  # Do not save image data, the path is all we need

    def load(self):
        self.data = cv2.imread(self.path)

    def show(self):
        if self.data is None:
            self.load()
        show_img(self.data)

    def unload(self):
        del self.data
        self.data = None

    def get(self):
        if self.data is None:
            self.load()
        return self.data


class ProcessedImage(LocalImage):

    def __init__(self, path):
        super().__init__(path)
        self.tags = {}
        self.modified = False
        self.brightness = None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def tag(self, tag_name: str, tag_value):
        self.tags[tag_name] = tag_value

    def get_tag(self, tag_name: str):
        return self.tags.get(tag_name)

    def get_brightness(self, unload_after=True) -> int:
        # Use cached brightness so that we do not have to load an image
        if self.brightness is not None:
            return self.brightness
        img_data = self.get()
        hsv_img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
        value_total = hsv_img_data[:, :, 2].sum()
        if unload_after:
            self.unload()
        self.brightness = value_total
        return self.brightness

    def set_img_data(self, img_data: np.ndarray):
        self.data = img_data
        self.modified = True

    def set_gamma(self, gamma_level: float):
        self.data = self.adjust_gamma(self.get(), gamma_level)

    @staticmethod
    def detect_pedestrians(img: np.ndarray, return_visual=True):
        """Could work for a ground-view perspective, but not helpful for higher elevation nodes."""

        small_img = imutils.resize(img, width=min(RESIZED_IMAGE_WIDTH, img.shape[1]))
        (rects, weights) = hog_ped_detector.detectMultiScale(small_img, winStride=(4, 4),
                                                             padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        picks = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        if return_visual:
            for (x1, x2, y1, y2) in picks:
                cv2.rectangle(small_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                Image.fromarray(small_img).show()

        return picks

    @staticmethod
    def get_brightness_img(img_data) -> int:
        hsv_img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
        value_total = hsv_img_data[:, :, 2].sum()
        return value_total

    @staticmethod
    def adjust_gamma(image, gamma_level: float):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma_level
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def get_gamma_adjustment_for_lumosity(self, lumosity: int):
        b = self.get().shape[0] * self.get().shape[1] * 255
        a = self.get_brightness() - b
        return a / (lumosity - b)


class LocalImageCollection:

    """
    A class to simplify the importing of hundreds of images from a folder with a complex subfolder hierarchy
    """

    CLASS_TAG = "[LocalImageCollection]"

    def __init__(self, root_folder: str, name: str):
        self.name = name
        self.root_folder = root_folder
        self.local_images: List[ProcessedImage] = []
        self._scan()
        self.n = len(self.local_images)

        self._iter_out = False
        self.unloading = False
        self.max_n = None

    def __iter__(self):
        if self._iter_out:
            print('%s Starting iteration of (%s)' % (self.CLASS_TAG, self.root_folder))
        self._img_n = -1
        return self

    def __next__(self) -> ProcessedImage:
        self._img_n += 1
        if self._img_n > 0:
            self.local_images[self._img_n - 1].unload()
        if self.max_n is not None:
            if self._img_n > self.max_n:
                raise StopIteration
        if self._iter_out and (self._img_n % 100 == 0):
            print("%s Loaded image %d" % (self.CLASS_TAG, self._img_n))
        if self._img_n < self.n:
            return self.local_images[self._img_n]
        else:
            raise StopIteration

    def _scan(self):
        """Look through the root folder and compile a flat list of images."""
        if not os.path.exists(self.root_folder):
            raise RuntimeError('%s Folder (%s) does not exist' % (self.CLASS_TAG, self.root_folder))
        for path in sorted(Path(self.root_folder).rglob('*.jpg')):
            self.local_images.append(ProcessedImage(os.path.join(self.root_folder, str(path.absolute()))))

    def get_all_images(self):
        return self.local_images

    def get_img(self, n: int):
        return self.local_images[n]

    def iter_with(self, output=True, max_n=None, unloading=True):
        self._iter_out = output
        self.max_n = max_n
        self.unloading = unloading
        return self
