import cv2
import numpy as np

img = cv2.imread('messi5.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = img[90:132, 218:257]

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template.shape[:: -1]

res = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
print(res)
threshold = 0.8
loc = np.where(res >= threshold)
print(loc)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 2)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
