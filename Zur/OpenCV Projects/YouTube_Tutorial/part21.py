import cv2
import numpy as np

img = cv2.imread('lena.jpg')
layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    # cv2.imshow(str(i), layer)

layer = gp[5]
cv2.imshow('upper level Gaussian Pyramid', layer)
lp = [layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp[i])
    laplacian = cv2.subtract(gp[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

'''
low_res = cv2.pyrDown(img)
low_res2 = cv2.pyrDown(low_res)
high_res = cv2.pyrUp(low_res2)

cv2.imshow('Original Image', img)
cv2.imshow('PyrDown 1', low_res)
cv2.imshow('pyrDown 2', low_res2)
cv2.imshow('pyrUp 1', high_res)
'''
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
