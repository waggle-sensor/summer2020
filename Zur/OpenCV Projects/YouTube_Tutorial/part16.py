import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('gradient.png')
_, th1 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
_, th2 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
_, th3 = cv.threshold(img, 100, 255, cv.THRESH_TRUNC)
_, th4 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO)
_, th5 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO_INV)


titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, th1, th2, th3, th4, th5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

# cv.imshow('Image', img)
# cv.imshow('th1', th1)
# cv.imshow('th2', th2)
# cv.imshow('th3', th3)
# cv.imshow('th4', th4)
# cv.imshow('th5', th5)

plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()



'''
img = cv2.imread('lena.jpg', -1)
cv2.imshow('image', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
