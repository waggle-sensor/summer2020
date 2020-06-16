import cv2
import numpy as np
from PIL import Image
import os.path
from typing import Tuple
from sklearn.cluster import KMeans
import skimage
from matplotlib import pyplot as plt


# Constants
BALLFIELD_MASK = cv2.cvtColor(cv2.imread('/home/ljacobs/Argonne/water-data/ball_park_1/bin_mask_ballpark1.png'),
                              cv2.COLOR_BGR2GRAY)
FOLDER_1_021A = '/home/ljacobs/Argonne/water-data/ball_park_0/bp1_021A'
FOLDER_0_016A = '/home/ljacobs/Argonne/water-data/ball_park_0/bp0_016A'
EXAMPLES_FOLDER = '/home/ljacobs/Argonne/water-data/example_cases'
ZOOMED_EXAMPLES_FOLDER = '/home/ljacobs/Argonne/water-data/example_cases/zoomed'
BALLFIELD_1_CROP = (319, 382, 794, 189)  # X, Y, Width, Height
BALLFIELD_1_TIGHT_CROP = (537, 441, 534, 104)
BALLFIELD_0_CROP = (384, 354, 870, 206)  # X, Y, Width, Height
BALLFIELD_0_TIGHT_CROP = (574, 379, 581, 113)
# ALL_CROP = (319, 300, 900, 300)

# People detection initialization
hog_classify = cv2.HOGDescriptor()
hog_classify.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# def apply_mask(img: np.ndarray):
#     return cv2.bitwise_and(img, img, mask=BALLFIELD_MASK)

def apply_crop(img: np.ndarray, crop: Tuple[int, int, int, int]):
    x, y, w, h = crop
    return img[y:y+h, x:x+w]

def show_img(img: np.ndarray):
    Image.fromarray(img).show()

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
        cv2.putText(output_canvas, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return output_canvas

def get_images_from(folder: str, first_n: int = None, exclude_containing: str = None) -> Tuple[str, np.ndarray]:
    img_formats = ['.png', '.jpg']
    img_paths = [os.path.join(folder, img_path) for img_path in os.listdir(folder)
                 if os.path.splitext(img_path)[-1] in img_formats and img_path.count(exclude_containing) == 0]

    def sort_by_frame(img_path):
        return int(os.path.basename(img_path)[:-4])

    for i, img_path in enumerate(sorted(img_paths, key=sort_by_frame)):
        if first_n is None:
            yield img_path, cv2.imread(img_path)
        else:
            if i < first_n:
                yield img_path, cv2.imread(img_path)
            else:
                return
    return

def apply_and_show_detection(img: np.ndarray, bg_sub: cv2.BackgroundSubtractorMOG2):
    # Apply mask to the image to extract the ballfield
    bg_sub_mask = bg_sub.apply(img)
    ret, fg_mask = cv2.threshold(bg_sub_mask, 200, 255, cv2.THRESH_BINARY)
    ret, shadow_mask = cv2.threshold(bg_sub_mask, 100, 200, cv2.THRESH_BINARY)

    # Edge detection to find dynamic-textured puddles
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    # People detection to ignore those edges
    # rects, weights = hog_classify.detectMultiScale(img, winStride=(1, 1), scale=1.01)
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     print('Detected person at %s' % str((x, y)))
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    sigma = 0.33
    med = np.median(img)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(img, lower, upper)
    edges = cv2.dilate(edges, (2, 2))

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Background subtractor gives very accurate shadow detection
    shadows = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    shadows = cv2.morphologyEx(shadows, cv2.MORPH_CLOSE, np.ones((5, 5)))

    # Foreground objects are not water
    # TODO Ignore the area around foreground objects (people), since they are definitely not water

    # Different edge detection techniques
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # sobelx = cv2.Sobel(hsv_img, cv2.CV_8U, 1, 0, ksize=5)
    # sobely = cv2.Sobel(hsv_img, cv2.CV_8U, 0, 1, ksize=5)
    # gradient_img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # Give output
    # value_map = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]
    debug_img = stitch_together(img, edges, shadows, fg_mask, text='Contours: %d' % len(contours))
    # show_img(debug_img)
    # input('CONTINUE')

    return debug_img, contours


if __name__ == '__main__':
    # people_path = '/home/ljacobs/Argonne/water-data/ball_park_1/example_cases/people1.png'
    # people_img = cv2.imread(people_path)
    # apply_and_show_detection(people_img)

    # ----- Gradient Script -----
    os.makedirs('./debug_imgs_4', exist_ok=True)
    os.chdir('./debug_imgs_4')
    gradient_sum_list = []
    filename_list = []
    for i, (path, img) in enumerate(get_images_from(FOLDER_0_016A, exclude_containing='_gradient')):
        img = apply_crop(img, BALLFIELD_0_TIGHT_CROP)
        img: np.ndarray = cv2.GaussianBlur(img, (5, 5), 0)
        gradient = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_16S, ksize=5)
        gradient_file = 'gradient_' + os.path.basename(path)
        gradient_abs: np.ndarray = np.absolute(gradient).astype(np.uint8)

        if i % 1000 == 0:
            print('Iteration: %d' % i)

        # Ballfield isolating
        # flat_img = img.reshape((img.shape[0] * img.shape[1], 3))
        # clt = KMeans()
        # clt.fit(flat_img)
        # show_img(clt.labels_.reshape((img.shape[0], img.shape[1])) * 50)

        gradient_sum = gradient_abs.sum()
        gradient_sum_list.append(gradient_sum)
        # print('%s | Sum: %d' % (gradient_file, gradient_sum))
        debug_img = stitch_together(img, gradient_abs, text=str(gradient_sum))
        cv2.imwrite(gradient_file, debug_img)

    # Histogram to show the most frequent gradient sum values
    plt.hist(gradient_sum_list, bins=20)
    plt.show()

    # Line graph to show gradient sum values over time
    plt.plot(gradient_sum_list)
    plt.show()

    # ----- Debug Image Script -----
    bg_subtract = cv2.createBackgroundSubtractorMOG2(history=3, detectShadows=True)
    os.makedirs('./debug_imgs_4', exist_ok=True)
    os.chdir('./debug_imgs_4')

    for path, img in get_images_from(FOLDER_0_016A):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_crop(img, BALLFIELD_0_CROP)
        debug_img, contours = apply_and_show_detection(img, bg_subtract)
        print('%s | Contours: %d' % (path, len(contours)))
        cv2.imwrite(os.path.basename(path), debug_img)
