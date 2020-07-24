import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
image = cv2.resize(image, (750, 570))

pts = []
text1 = "Click Top Left Point"
text2 = "Click Top Right Point"
text3 = "Click Bottom Right Point"
text4 = "Click Bottom Left Point"
text5 = "Press any key for warped image"
font = cv2.FONT_HERSHEY_SIMPLEX
black = (0, 0, 0)

image = cv2.putText(image, text1, (0, 30), font, 0.5, black, 1)


def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) == 0:
            cv2.putText(image, text2, (0, 50), font, 0.5,  black, 1)
        if len(pts) == 1:
            cv2.putText(image, text3, (0, 70), font, 0.5, black, 1)
        if len(pts) == 2:
            cv2.putText(image, text4, (0, 90), font, 0.5, black, 1)
        if len(pts) == 3:
            cv2.putText(image, text5, (0, 110), font, 0.5, black, 2)

        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("image", image)
        pts.append((x, y))


def four_point_transform(img, points):
    tl = points[0]
    tr = points[1]
    br = points[2]
    bl = points[3]

    rect = np.array([tl, tr, br, bl], dtype="float32")

    # compute the width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # creating new image using calculated width and height
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


cv2.imshow("image", image)
cv2.setMouseCallback("image", get_points)
cv2.waitKey(0)

bev = four_point_transform(image, pts)
cv2.imshow("Bird's Eye View", bev)
cv2.waitKey(0)

cv2.destroyAllWindows()
