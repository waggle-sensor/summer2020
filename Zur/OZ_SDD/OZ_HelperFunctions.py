# import packages
import cv2
import numpy as np
from scipy.spatial import distance as dist
import datetime
import time
import dlib
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from OZ_NMS import non_max_suppression


# yolo detection function
# input: current frame, yolo detector, confidence threshold, NMS threshold
# output: list of dlib trackers used in tracking phase in main function
def yolo_detection(frame, yolo_net, layerNames, confid, threshold):
    # updating status and initializing list of trackers
    trackers = []
    (H, W) = frame.shape[:2]

    # convert color from BGR to RGB, used for dlib trackers
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # passing frame through YOLO detector
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layerOutputs = yolo_net.forward(layerNames)

    # initializing lists of detected bounding boxes
    boxes = []

    # loop over each of the layer outputs
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
                # *Note: YOLO  returns the center (x, y)-coordinates of bbox, then width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # using center, width, and height to calculate top-left and bottom-right points
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))
                endX = int(centerX + (width / 2))
                endY = int(centerY + (height / 2))

                # update our list of bounding box coordinates
                box = (startX, startY, endX, endY)
                boxes.append(box)

    # apply non-maximum suppression algorithm
    bboxes = np.array(boxes).astype(int)
    boxes = non_max_suppression(bboxes, threshold)

    if len(boxes) > 0:
        for box in boxes:
            # construct a dlib rectangle object from the bounding box coordinates
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(box[0], box[1], box[2], box[3])

            # start the dlib correlation tracker and add to list of trackers
            tracker.start_track(rgb, rect)
            trackers.append(tracker)

    return trackers


# function to transform and calculate bottom points of each bounding boxer
# input: list of boxes, transformation matrix
# output: list of (x, y) coordinates
def transform_box_points(boxes, matrix):
    bottom_points = []
    for box in boxes:
        x1, y1, x2, y2 = box

        # calculate coordinate of center of bottom of box
        pnts = np.array([[[int(x1 + (x2 / 2)), int(y2)]]], dtype="float32")

        # transform and append transformed coordinate to list
        bd_pnt = cv2.perspectiveTransform(pnts, matrix)[0][0]
        pnt = (int(bd_pnt[0]), int(bd_pnt[1]))
        bottom_points.append(pnt)

    return bottom_points


# function that that calculates distances between each pair of bottom_points and compares to minimum safe distance
# input: list of boxes, list of bottom points, minimum safe distance
# output: list of pairs of points tagged with safe boolean, list of pairs of boxes tagged with safe boolean
def violation_detection(boxes, bottom_points, safe_dist):
    distances = []
    checked_boxes = []

    # loop through each pair of bottom points
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i != j:
                # calculate distance
                distance = np.sqrt((bottom_points[i][0] - bottom_points[j][0]) ** 2 +
                                   (bottom_points[i][1] - bottom_points[j][1]) ** 2)
                # compare to min safe distance, tag with appropriate safe boolean
                if distance > safe_dist:
                    safe = True
                    distances.append([bottom_points[i], bottom_points[j], safe])
                    checked_boxes.append([boxes[i], boxes[j], safe])
                else:
                    safe = False
                    distances.append([bottom_points[i], bottom_points[j], safe])
                    checked_boxes.append([boxes[i], boxes[j], safe])
    return distances, checked_boxes


# function that checks each pair of distances and sorts individual distances based on safe boolean
# input: list of distances
# output: number of unsafe and safe points
def get_violation_count(distances):
    red = []
    green = []

    for i in range(len(distances)):
        if not distances[i][2]:
            if (distances[i][0] not in red) and (distances[i][0] not in green):
                red.append(distances[i][0])
            if (distances[i][1] not in red) and (distances[i][1] not in green):
                red.append(distances[i][1])

    for i in range(len(distances)):
        if distances[i][2]:
            if (distances[i][0] not in red) and (distances[i][0] not in green):
                green.append(distances[i][0])
            if (distances[i][1] not in red) and (distances[i][1] not in green):
                green.append(distances[i][1])

    return len(red), len(green)


def SDD_output(frame, box_list, checked_boxes):
    black = (0, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)

    for bbox in box_list:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), black, 2)

    for i in range(len(checked_boxes)):
        person1 = checked_boxes[i][0]
        person2 = checked_boxes[i][1]
        safe = checked_boxes[i][2]

        if not safe:
            x1, y1, x2, y2 = person1
            cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)

            x1, y1, x2, y2 = person2
            cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)

        else:
            x1, y1, x2, y2 = person1
            cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)

            x1, y1, x2, y2 = person2
            cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)

    return frame
