# import packages
import cv2
import numpy as np


"""
Function that ensures that the first four inputted mouse points are in the correct order:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left
Input: List of (x, y-coordinates
Output: Numpy array of (x, y)-coordinates 
"""
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # split point into the two left points and the two right points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # sort left two points to get top-left and bottom-left point
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # calculate distance between top-left and two right points.
    # greater distance is bottom-right point, other is bottom-left point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
