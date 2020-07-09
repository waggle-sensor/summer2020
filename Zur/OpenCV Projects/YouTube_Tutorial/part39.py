import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fgbg.apply(frame)

    cv.imshow('Frame', frame)
    cv.imshow('FG', fgmask)

    k = cv.waitKey(30)
    if k == 'q' or k == 27:
        break

cap.release()
cv.destroyAllWindows()
