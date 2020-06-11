from PIL import Image
import cv2
import numpy as np


def show_img(img):
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


def stitch_together(*imgs, text: str = None) -> np.ndarray:
    height = max([img.shape[0] for img in imgs])
    width = sum([img.shape[1] for img in imgs])
    output_canvas = np.zeros((height, width, 3)).astype(np.uint8)
    prev_marker = 0
    for img in imgs:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output_canvas[:, prev_marker:prev_marker+img.shape[1]] = img
        prev_marker = prev_marker+img.shape[1]
    if text is not None:
        cv2.putText(output_canvas, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    return output_canvas
