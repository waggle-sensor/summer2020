import cv2

img = cv2.imread("lego_pics/lego1.png")
# newW = img.shape[0] * 0.25
# newH = img.shape[1] * 0.25
# newW = int(newW)
# newH = int(newH)
# img = cv2.resize(img, (newW, newH))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()