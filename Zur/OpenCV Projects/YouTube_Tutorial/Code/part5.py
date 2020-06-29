import numpy as np
import cv2

# img = cv2.imread('lena.jpg', 1)
img = np.zeros([512, 512, 3], np.uint8)

img = cv2.line(img, (0, 0), (300, 300), (0, 0, 255), 10)
img = cv2.arrowedLine(img, (0, 300), (300, 300), (100, 0, 0), 10)
img = cv2.rectangle(img, (20, 20), (100, 100), (0, 255, 0), -1)
img = cv2.circle(img, (400, 400), 30, (255, 255, 0), 5)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'OpenCV', (40, 400), font, 2, (255, 255, 255), 3)


cv2.imshow('image',  img)

cv2.waitKey(0)
cv2.destroyAllWindows()

