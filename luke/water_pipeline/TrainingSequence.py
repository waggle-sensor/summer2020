import os
from typing import Tuple, Optional
from joblib import cpu_count
import numpy as np
# import cupy as np
import cv2

N_PROCESSES = cpu_count() - 8

class TrainingSequence:

    """
    A class to keep track of the different forms of training videos in my library and to load them in a RAM-conscious
    way. Features a method that can attach a binary mask to each frame in the video, classifying the underlying pixels
    as water.
    """

    # Categories that TrainingSequences can fall under:
    CAT_NO_WATER = 0
    CAT_WATER = 1
    CAT_BOTH = -1
    CAT_UNLABELED = 2

    def __init__(self, input_sequence: str, category=None, name=None, fps=None, water_mask_path=None,
                 all_water=None, all_non_water=None):
        """
        Initialize a TrainingSequence from a folder or a video. If you are loading a video with an attached mask, make
        sure to specify `water_mask_path` with the path to the water mask image.

        If loading from a folder, every frame in the folder must be numbered according to the format: %d.png
        """

        self.name = name
        self.input_sequence = input_sequence
        self.is_folder = os.path.isdir(input_sequence)

        if not self.is_folder:
            self.vid_resource = cv2.VideoCapture(input_sequence)
            if self.vid_resource is None:
                raise RuntimeError('Unable to load video resource at %s' % input_sequence)
            # self.vid_resource.release()
        else:
            self.vid_resource = None

        self.fps = fps
        self.n_frames = None
        self.dims = self.get_dims()

        if self.is_folder and fps is None:
            print('[WARNING] Specify FPS for this folder (%s)' % self.input_sequence)
        elif not self.is_folder:
            self.fps = self.vid_resource.get(cv2.CAP_PROP_FRAME_COUNT)
            self.n_frames = self.vid_resource.get(cv2.CAP_PROP_FRAME_COUNT)

        # Load water mask as boolean ndarray or None
        self.water_mask = np.array(cv2.imread(water_mask_path, cv2.IMREAD_GRAYSCALE) > 50) \
            if water_mask_path is not None else None

        # Category of this training video so that I can sort my dataset
        if all_water:
            self.category = self.CAT_WATER
        elif all_non_water:
            self.category = self.CAT_NO_WATER
        elif self.water_mask is not None:
            self.category = self.CAT_BOTH
        elif category == self.CAT_UNLABELED:
            self.category = category
        else:
            raise RuntimeError('Please specify the category of this TrainingSequence!')

        # Initialized when images are loaded
        self.img_gray_ar = None   # Shape: (frame #, y-size, x-size)
        self.img_color_ar = None  # Shape: (frame #, y-size, x-size, HSV color channels)

    # Private helper functions

    def _resize_water_mask(self):
        """
        Resize internal water mask if necessary to overlap completely with the loaded grayscale array (img_gray_ar)
        """

        if self.img_gray_ar.shape[0] != self.water_mask.shape[0] and self.img_gray_ar.shape[1] != self.water_mask.shape[1]:
            scaleY = self.img_gray_ar.shape[0] / self.water_mask.shape[0]
            scaleX = self.img_gray_ar.shape[1] / self.water_mask.shape[1]

            # If there is an aspect ratio problem with resizing the input mask
            if int(scaleX) != int(scaleY):
                print('[WARNING] Mask dimensions will be off by a pixel or two: %f %f' % (scaleY, scaleX))

            print('Resizing water mask...')
            self.water_mask = cv2.resize(self.water_mask.astype(np.uint8), self.img_gray_ar.shape[:0:-1],
                                         interpolation=cv2.INTER_AREA) > 0

    def _load_frames_video(self, dims=None, in_color_too=False, max_n=None):
        """
        Load frames specifically from a video source.
        """

        # Load video capture if possible
        if self.vid_resource is None or not self.vid_resource.isOpened():
            self.vid_resource = cv2.VideoCapture(self.input_sequence)
            if self.vid_resource is None or not self.vid_resource.isOpened():
                print('Unable to load video %s' % self.input_sequence)
                raise RuntimeError('Video resource is None')

        x = self.vid_resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        y = self.vid_resource.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if x == 0 and y == 0:
            print('ERROR IN LOADING VIDEO')
            raise RuntimeError('Unable to load video (0 width and height)')

        # Resizing
        if dims is None:
            img_y = int(y)
            img_x = int(x)
        else:
            img_y, img_x = dims
        if x != img_x or y != img_y:
            print('[LOAD] Frames will be resized')

        # Allocate numpy array of sufficient size for grayscale array (and color array if specified too)
        self.img_gray_ar = np.zeros((max_n, img_y, img_x), dtype=np.uint8)
        if in_color_too:
            self.img_color_ar = np.zeros((max_n, img_y, img_x, 3), dtype=np.uint8)

        # Load images from video file into numpy array, converting to grayscale and saving color images if specified
        frame_n = 0
        while self.vid_resource.isOpened():
            ret, frame = self.vid_resource.read()
            if ret:
                frame = cv2.resize(frame, (img_x, img_y))
                if in_color_too:
                    self.img_color_ar[frame_n] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.img_gray_ar[frame_n] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                break
            frame_n += 1
            if max_n is not None:
                if frame_n >= max_n:
                    break

        # Resize water mask if necessary (if it does not fit the downscaled image)
        if self.water_mask is not None:
            self._resize_water_mask()

        # Return instance
        return self

    def _load_frames_folder(self, in_color_too=False, max_n=None, scaling=1.0):
        """
        Loads a numpy data array of shape:
            (n_of_imgs_in_folder, images_y, images_x, images_color_channels)
        Works on a folder with images in the format of <number>.jpg
        """

        frames = sorted([file for file in os.listdir(self.input_sequence) if file.find('.png') != -1],
                        key=lambda item: int(item[:-4]))[:max_n]
        if len(frames) == 0:
            raise RuntimeError('There are no frames to load in: %s' % self.input_sequence)

        first_frame = cv2.imread(os.path.join(self.input_sequence, frames[0]))
        height, width, channels = first_frame.shape

        self.img_gray_ar = np.zeros((len(frames), int(height * scaling), int(width * scaling)))
        if in_color_too:
            self.img_color_ar = np.zeros((len(frames), int(height * scaling), int(width * scaling), 3))

        for file in frames:
            img = cv2.imread(os.path.join(self.input_sequence, file), cv2.IMREAD_GRAYSCALE)
            dim = (int(img.shape[1] * scaling), int(img.shape[0] * scaling))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            self.img_gray_ar[int(file[:-4]) - 1] = img
            if in_color_too:
                img = cv2.imread(os.path.join(self.input_sequence, file))
                dim = (int(img.shape[1] * scaling), int(img.shape[0] * scaling))
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                self.img_color_ar[int(file[:-4]) - 1] = img

        # Resize water mask if necessary (if it does not fit the downscaled image)
        if self.water_mask is not None:
            self._resize_water_mask()

        return self

    # Low-level frame functions

    def read_frame(self, n: int, release_immediately=True) -> Optional[np.ndarray]:
        """
        Return frame at position n. Now only implemented for video files.

        NOTE: This is very inefficient and should not be used over a large dataset. Use TrainingSet.load method for mass
        loads of frames.
        """

        # Load an OpenCV capture if not already loaded
        if self.vid_resource is None or not self.vid_resource.isOpened():
            self.vid_resource = cv2.VideoCapture(self.input_sequence)

        # Set correct marker
        if n != 0:
            success = self.vid_resource.set(cv2.CAP_PROP_POS_FRAMES, n)
            if not success:
                print('[Warning] Unable to set video to frame #%d in video %s' % (n, self.input_sequence))

        ret, frame = self.vid_resource.read()

        if release_immediately:
            self.vid_resource.release()
            self.vid_resource = None

        # Return loaded frame in HSV or None
        if not ret:
            return None
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def load_frames(self, dims=None, in_color_too=False, max_n=None):
        """
        Load frames from the location specified by this object's initialization.

        :param dims: Dimensions of frames to resize in format (y, x)
        :param in_color_too: Whether to load frames in color too
        :param max_n: Max number of frames to load
        :return: Self instance
        """

        try:
            if self.is_folder:
                return self._load_frames_folder(in_color_too=in_color_too, max_n=max_n)
            else:
                return self._load_frames_video(in_color_too=in_color_too, dims=dims, max_n=max_n)
        except RuntimeError:
            return None

    def is_frames_loaded(self):
        if self.img_color_ar is not None or self.img_gray_ar is not None:
            return True

    def unload_frames(self):
        # NOTE: Python garbage collection should take care of the unreferenced frames buffer
        self.img_gray_ar = None
        self.img_color_ar = None

    # Grabbing functions

    def get_dims(self) -> Optional[Tuple[int, int]]:
        return (int(self.vid_resource.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.vid_resource.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def get_water_mask(self) -> Optional[np.ndarray]:
        """
        Get the water mask attached to this TrainingSequence. This is automatically generated if the sequence is labeled
        as all water or all non-water, so a mask is returned either with all True's or all False's.

        :return: ndarray of size (y-size, x-size) with dtype bool or None if the category is "Both" and there is no stored
        mask
        """

        if self.category == self.CAT_BOTH:
            return self.water_mask
        elif self.water_mask is None:
            if self.vid_resource is None or not self.vid_resource.isOpened():
                self.vid_resource = cv2.VideoCapture(self.input_sequence)
            y, x = (int(self.vid_resource.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.vid_resource.get(cv2.CAP_PROP_FRAME_WIDTH)))

            # If this is a water-only image

            if self.category == self.CAT_NO_WATER:
                return np.zeros((y, x), dtype=bool)
            elif self.category == self.CAT_WATER:
                return np.ones((y, x), dtype=bool)
            else:
                return None
        else:
            return None
