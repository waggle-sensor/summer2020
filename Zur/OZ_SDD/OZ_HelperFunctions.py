# import packages
import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from scipy.spatial.distance import pdist, squareform
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
        # calculate coordinate of bottom center point of each box
        bottom_center = np.array([[[int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]]], dtype="float32")

        # transform point coordinates using transformation matrix, append to list
        warped_pt = cv2.perspectiveTransform(bottom_center, matrix)[0][0]
        pnt = [int(warped_pt[0]), int(warped_pt[1])]
        bottom_points.append(pnt)

    return bottom_points


# function that that calculates distances between each pair of bottom_points and compares to minimum safe distance
# input: list of boxes, list of bottom points, minimum safe distance
# output: list of pairs of points tagged with safe boolean, list of pairs of boxes tagged with safe boolean
def violation_detection(boxes, bottom_points, safe_dist):
    distance_pairs = []
    box_pairs = []

    # loop through each pair of bottom points
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i < j:
                # calculate distance
                distance = np.sqrt((bottom_points[j][0] - bottom_points[i][0]) ** 2 +
                                   (bottom_points[j][1] - bottom_points[i][1]) ** 2)

                p1 = bottom_points[i]
                p2 = bottom_points[j]
                h = abs(p2[1] - p1[1])
                w = abs(p2[0] - p1[0])
                dis_w = float((w / safe_dist) * 180)
                dis_h = float((h / safe_dist) * 180)
                distance = int(np.sqrt((dis_h ** 2) + (dis_w ** 2)))

                # compare to min safe distance, tag with appropriate safe boolean
                if distance < 180:
                    safe = False
                    distance_pairs.append([bottom_points[i], bottom_points[j], safe])
                    box_pairs.append([boxes[i], boxes[j], safe])
                else:
                    safe = True
                    distance_pairs.append([bottom_points[i], bottom_points[j], safe])
                    box_pairs.append([boxes[i], boxes[j], safe])
    return distance_pairs, box_pairs


# function that checks each pair of distances and sorts individual distances based on safe boolean
# input: list of distance pairs tagged with safe boolean
# output: number of unsafe and safe points
def get_violation_count(distance_pairs):
    red = []
    green = []

    # if not safe, add points to red list
    for i in range(len(distance_pairs)):
        if not distance_pairs[i][2]:
            if (distance_pairs[i][0] not in red) and (distance_pairs[i][0] not in green):
                red.append(distance_pairs[i][0])
            if (distance_pairs[i][1] not in red) and (distance_pairs[i][1] not in green):
                red.append(distance_pairs[i][1])

    # if safe, add points to green list
    for i in range(len(distance_pairs)):
        if distance_pairs[i][2]:
            if (distance_pairs[i][0] not in red) and (distance_pairs[i][0] not in green):
                green.append(distance_pairs[i][0])
            if (distance_pairs[i][1] not in red) and (distance_pairs[i][1] not in green):
                green.append(distance_pairs[i][1])

    return len(red), len(green)


# function that draws red and green rectangles around each person depending on safe boolean
# input: current frame, list of bounding boxes, list of box pairs tagged with safe boolean
def SDD_output(frame, box_list, box_pairs):
    for i in range(len(box_list)):
        x, y, w, h = box_list[i][:]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i in range(len(box_list)):

        per1 = box_list[i][0]
        per2 = box_list[i][1]
        safe = box_list[i][2]

        if not safe:
            x, y, w, h = per1[:]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            x1, y1, w1, h1 = per2[:]
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

            # frame = cv2.line(frame, (int(x + w / 2), int(y + h / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)), red, 2)

    return frame


def BEV_output(frame, distance_pairs, bottom_points, scale_w, scale_h,):
    (H, W) = frame.shape[:2]

    blank_image = np.zeros((int(H * scale_h), int(W * scale_w), 3), np.uint8)
    blank_image[:] = (255, 255, 255)

    red = []
    green = []

    for i in range(len(distance_pairs)):
        if not distance_pairs[i][2]:
            if distance_pairs[i][0] not in red and distance_pairs[i][0] not in green:
                red.append(distance_pairs[i][0])
            if distance_pairs[i][1] not in red and distance_pairs[i][1] not in green:
                red.append(distance_pairs[i][1])

            blank_image = cv2.line(blank_image,
                                   (int(distance_pairs[i][0][0] * scale_w), int(distance_pairs[i][0][1] * scale_h)),
                                   (int(distance_pairs[i][1][0] * scale_w), int(distance_pairs[i][1][1] * scale_h)),
                                   (0, 0, 255), 2)

    for i in range(len(distance_pairs)):
        if distance_pairs[i][2]:
            if distance_pairs[i][0] not in red and distance_pairs[i][0] not in green:
                green.append(distance_pairs[i][0])
            if distance_pairs[i][1] not in red and distance_pairs[i][1] not in green:
                green.append(distance_pairs[i][1])

    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, (0, 255, 0), 10)
    for i in red:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, (0, 0, 255), 10)

    return blank_image





def transform_plot_points(frame, boundingboxes, matrix, scale_w, scale_h):
    (H, W) = frame.shape[:2]

    bird_image = np.zeros((int(H * scale_h), int(W * scale_w), 3), np.uint8)
    bird_image[:] = (0, 0, 0)

    warped_pts = []
    for box in boundingboxes:
        (x1, y1, x2, y2) = box

        x = int(x1 + ((x2 - x1) / 2))
        y = int(y2)

        pts = np.array([[[x, y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, matrix)[0][0]
        warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]

        warped_pts.append(warped_pt_scaled)
        bird_image = cv2.circle(bird_image, (warped_pt_scaled[0], warped_pt_scaled[1]), 10, (255, 255, 255), -1,)

    return warped_pts, bird_image


def draw_pedestrian_boxes(frame, boundingboxes):
    for box in boundingboxes:
        (x1, y1, x2, y2) = box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


def plot_lines_between_nodes(warped_points, bird_image, safe_dist):
    p = np.array(warped_points)
    dist_condensed = pdist(p)
    dist = squareform(dist_condensed)

    dd = np.where(dist < safe_dist)
    six_feet_violations = len(np.where(dist_condensed < safe_dist)[0])
    total_pairs = len(dist_condensed)
    danger_p = []
    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            danger_p.append([point1, point2])
            cv2.line(bird_image, (p[point1][0], p[point1][1]),  (p[point2][0], p[point2][1]), (0, 0, 255), 4)

    # Display Birdeye view
    cv2.imshow("Bird Eye View", bird_image)
    cv2.waitKey(1)

    return six_feet_violations, total_pairs
