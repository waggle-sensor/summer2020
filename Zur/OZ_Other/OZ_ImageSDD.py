import cv2
import numpy as np
import argparse
import os
from OZ_HelperFunctions import *

mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        elif 4 <= len(mouse_pts) < 6:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        if 1 <= len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]),
                     (0, 0, 0), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        mouse_pts.append((x, y))
        cv2.imshow("image", image)


def image_SDD(img, points, yolo_net, layerNames, confid, threshold):
    (H, W) = img.shape[:2]
    boundingboxes = []

    src = np.float32(np.array(points[:4]))
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    perspective_transform = cv2.getPerspectiveTransform(src, dst)

    pts = np.float32(np.array([points[4:6]]))
    warped_pt = cv2.perspectiveTransform(pts, perspective_transform)[0]

    safe_dist = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)

    pnts = np.array(points[:4], np.int32)
    cv2.polylines(img, [pnts], True, (255, 0, 0), thickness=2)

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layerOutputs = yolo_net.forward(layerNames)

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filtering detections to just people
            if classID == 0 and confidence > confid:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # using center, width, and height to calculate top-left and bottom-right points
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))
                endX = int(centerX + (width / 2))
                endY = int(centerY + (height / 2))

                # update our list of bounding box coordinates
                box = (startX, startY, endX, endY)
                boundingboxes.append(box)

    bboxes = np.array(boundingboxes).astype(int)
    boundingboxes = non_max_suppression(bboxes, threshold)

    bottom_points = transform_box_points(boundingboxes, perspective_transform)
    distance_pairs, box_pairs = violation_detection(boundingboxes, bottom_points, safe_dist)
    img = SDD_output(img, boundingboxes, box_pairs)

    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


########################################################################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str,
                help="path to image")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maximum suppression")
args = vars(ap.parse_args())

# deriving the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# loading YOLO object detector
print("INFO: LOADING YOLO FROM DISK...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# calling mouse callback
# cv2.namedWindow("image")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=750)
copy = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)

while True:
    cv2.imshow("image", image)
    cv2.waitKey(1)
    if len(mouse_pts) == 7:
        cv2.destroyWindow("out1")
        break

image_SDD(copy, mouse_pts, net, ln, args["confidence"], args["threshold"])
