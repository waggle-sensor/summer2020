# Usage
# python OZ_SDD_2.py --input videos/test_video2.mp4
# python OZ_SDD_2.py --input videos/pedestrians.mp4

# import packages
import cv2
import numpy as np
import argparse
import datetime
import time
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from OZ_HelperFunctions import *

# **MOUSE CALLBACK FUNCTION** #########################################################################################

# initialize list of mouse input coordinates
mouse_pts = []


# mouse callback function takes 7 clicks
# 1-4: rectangle region of interest (red circles)
#   Points must be clicked in the following order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
# 5-6: 2 points that are 6 feet apart on the ground plane
# 7: break, destroy window, start video
def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        else:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        if 1 <= len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (0, 0, 0), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        mouse_pts.append((x, y))


# **SOCIAL DISTANCE DETECTOR FUNCTION** ###############################################################################
def social_distance_detector(vid_input, vid_output, yolo_net, layerNames, confid, threshold):
    vs = cv2.VideoCapture(vid_input)
    global image

    # initializing:
    # writer = None  # video output writer
    (H, W) = (None, None)  # frame dimensions
    frameCount = 0  # keep track of number of frames so far
    points = []  # list of mouse input coordinates

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    scale_w, scale_h = float(400 / width), float(600 / height)

    # output video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    street_movie = cv2.VideoWriter("output/SDD_Street_Output.avi", fourcc, fps, (width, height))
    bird_movie = cv2.VideoWriter("output/SDD_Bird_Output.avi", fourcc, fps,
                                 (int(width * scale_w), int(height * scale_h)))

    # looping over each frame of video input
    while True:
        (grabbed, frame) = vs.read()

        # break if no frame left to grab
        if not grabbed:
            print("INFO: END OF VIDEO FILE")
            break

        # resizing frames for consistency, regardless of size of input video
        # frame = imutils.resize(frame, width=750)

        # grabbing frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # pulling first frame of the video and waiting for mouse click inputs
        if frameCount == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("image")
                    break
            points = mouse_pts

        # **IMAGE TRANSFORMATION** ####################################################################################

        # creating transformation matrix from first four mouse input points
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        matrix = cv2.getPerspectiveTransform(src, dst)

        # warping mouse points 5 - 6 using transformation matrix
        pts = np.float32(np.array([points[4:6]]))
        warped_pt = cv2.perspectiveTransform(pts, matrix)[0]

        # calculating number of pixels that make up approx. 6 feet
        safe_dist = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)

        # drawing the rectangle on the video for the remaining frames
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (0, 0, 0), thickness=2)

        # **PERSON DETECTION** ########################################################################################

        # YOLO DETECTION
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        layerOutputs = yolo_net.forward(layerNames)

        boxes = []
        confidences = []
        classIDs = []

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
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # calculate coordinates of top left point of box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maximum suppression algorithm and store final results in bounding box list
        nms_boxes = cv2.dnn.NMSBoxes(boxes, confidences, confid, threshold)
        boundingboxes = []
        for i in range(len(boxes)):
            if i in nms_boxes:
                boundingboxes.append(boxes[i])

        # **COORDINATE MAPPING AND DISTANCE CALCULATION** #############################################################
        if len(boundingboxes) == 0:
            frameCount = frameCount + 1
            continue

        bottom_points = transform_box_points(boundingboxes, matrix)

        distance_pairs, box_pairs = violation_detection(boundingboxes, bottom_points, safe_dist)

        risk_count = get_violation_count(distance_pairs)

        # **OUTPUT DISPLAY AND CLEAN UP** #############################################################################

        copy = np.copy(frame)
        bird_view = BEV_output(frame, distance_pairs, bottom_points, scale_w, scale_h)
        street_view = SDD_output(copy, boundingboxes, box_pairs)

        if frameCount != 0:
            street_movie.write(street_view)
            bird_movie.write(bird_view)

            cv2.imshow('Street View', street_view)
            cv2.imshow('Bird Eye View', bird_view)
            # cv2.imwrite(vid_output + "frame%d.jpg" % frameCount, street_view)
            # cv2.imwrite(vid_output + "bird_eye_view/frame%d.jpg" % frameCount, bird_view)

        frameCount = frameCount + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()


# **ARGUMENT PARSER, LOAD YOLO, CALL SDD** ############################################################################

# constructing argument parser and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="videos/test_video4.mp4",
                help="path to input video")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video")
ap.add_argument("-y", "--yolo", default="yolo-coco",
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
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)

# calling SDD function
social_distance_detector(args["input"], args["output"], net, ln, args["confidence"], args["threshold"])
