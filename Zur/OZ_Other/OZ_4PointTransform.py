import cv2
import numpy as np
from scipy.spatial import distance as dist


def order_points(pts):
    # sort 4 points based on their x-coordinate
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab left 2 points and right 2 points
    tlbl = xSorted[:2, :]
    trbr = xSorted[2:, :]

    # sort left points by y-coordinate to separate tl and bl
    tlbl = tlbl[np.argsort(tlbl[:, 1]), :]
    (tl, bl) = tlbl

    # calculate distance between tl and 2 right points
    # larger distance will be br, smaller distance will be tr
    D = dist.cdist(tl[np.newaxis], trbr, "euclidean")[0]
    (br, tr) = trbr[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(img, pts):
    # order points using above function
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
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
