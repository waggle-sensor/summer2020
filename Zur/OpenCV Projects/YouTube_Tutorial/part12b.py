import numpy as np
import cv2


def nothing(x):
    print(x)


cv2.namedWindow('image')

cv2.createTrackbar('cp', 'image', 10, 400, nothing)

switch = 'color/gray'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while 1:
    img = cv2.imread('lena.jpg')
    k = cv2.waitKey(1) & 0xFF
    pos = cv2.getTrackbarPos('cp', 'image')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(pos), (50, 150), font, 6, (0, 0, 255), 10)

    if k == 27:
        break

    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imshow('image', img)

cv2.destroyAllWindows()
