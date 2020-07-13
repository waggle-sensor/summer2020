import cv2

print ("OpenCV Version:")
print(cv2.__version__)

img = cv2.imread("fb_prof_pic.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Profile", img)
cv2.imshow("Profile-Gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

