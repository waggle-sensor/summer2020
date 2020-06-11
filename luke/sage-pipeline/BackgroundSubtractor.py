import cv2
import numpy as np
from ProcessedImageCollection import ProcessedImage
from typing import List


class BackgroundSubtractor:

    def __init__(self, tag: str):
        self.tag = tag
        self.model: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    def __setstate__(self, state):
        self.tag = state['tag']

        # Reinit model using stored background image
        self.model: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        bg_img = state.get('stored_bg_img')
        if bg_img is not None:
            self.model.apply(bg_img, learningRate=1)
        else:
            print('[!!] Restoring BackgroundSubtractor without a saved background image!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['stored_bg_img'] = self.model.getBackgroundImage()
        state['model'] = None
        return state

    def train_on(self, imgs: List[ProcessedImage], output=True):
        for i, img in enumerate(imgs):
            if i % 100 == 0 and output:
                print('[BackgroundSubtractor Training (%s)] Training on image: %d/%d' % (self.tag, i, len(imgs)))
            self.model.apply(img.get())
            img.unload()

    def get_foreground(self, img: np.ndarray) -> np.ndarray:
        blurred_img = cv2.blur(img, (5, 5))
        out = self.model.apply(blurred_img)
        return out

    def get_background(self):
        return self.model.getBackgroundImage()
