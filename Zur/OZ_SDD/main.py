# Usage
# python main.py --input videos/test_video2.mp4 --output output/output_01.avi --yolo yolo-coco

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
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]),
                     (0, 0, 0), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        mouse_pts.append((x, y))


# **SOCIAL DISTANCE DETECTOR FUNCTION** ###############################################################################
def social_distance_detector(vid_input, vid_output, yolo_net, layerNames, confid, skip, threshold):
    vs = cv2.VideoCapture(vid_input)
    global image

    # initializing:
    writer = None  # video output writer
    (W, H) = (None, None)  # frame dimensions
    trackers = []  # list to store results of yolo detection
    frameCount = 0  # keep track of number of frames so far
    points = []  # list of mouse input coordinates

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    scale_w, scale_h = float(400 / width), float(600 / height)

    # looping over each frame of video input
    while True:
        (grabbed, frame) = vs.read()

        # break if no frame left to grab
        if not grabbed:
            print("INFO: END OF VIDEO FILE")
            break

        # resizing frames for consistency, regardless of size of input video
        frame = imutils.resize(frame, width=750)

        # grabbing frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize writer for output video
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(vid_output, fourcc, 30, (W, H), True)

        # convert frame color from BGR to RGB, used for dlib trackers
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        perspective_transform = cv2.getPerspectiveTransform(src, dst)

        # warping mouse points 5 - 7 using transformation matrix
        pts = np.float32(np.array([points[4:6]]))
        warped_pt = cv2.perspectiveTransform(pts, perspective_transform)[0]

        # calculating number of pixels that make up approx. 6 feet
        safe_dist = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)

        # drawing the rectangle on the video for the remaining frames
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (0, 0, 0), thickness=2)

        # **PERSON DETECTION AND TRACKING** ###########################################################################

        # initialize status and list of bounding boxes
        status = "Waiting"
        boundingboxes = []

        # Yolo Detection
        if frameCount % skip == 0:
            status = "Detecting"
            trackers = yolo_detection(frame, yolo_net, layerNames, confid, threshold)

        # Object Tracking
        else:
            for tracker in trackers:
                # set status
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                boundingboxes.append((startX, startY, endX, endY))

        # for bbox in boundingboxes:
        # x1, y1, x2, y2 = bbox
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # **COORDINATE MAPPING AND DISTANCE CALCULATION** #############################################################

        if len(boundingboxes) > 0:
            bottom_points = transform_box_points(boundingboxes, perspective_transform)

            distances, checked_boxes = violation_detection(boundingboxes, bottom_points, safe_dist)

            risk_count = get_violation_count(distances)

            frame = SDD_output(frame, boundingboxes, checked_boxes)

        # **OUTPUT DISPLAY AND CLEAN UP** #############################################################################

        # display status
        text = "Status: " + status
        cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # check to see if we should write the frame to output video
        if writer is not None:
            writer.write(frame)

        # increment the total number of frames processed thus far
        frameCount += 1

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    vs.release()

    cv2.destroyAllWindows()


# **ARGUMENT PARSER, LOAD YOLO, CALL SDD** ############################################################################

# constructing argument parser and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
                help="path to input video")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=30,
                help="number of frames between yolo detections")
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
social_distance_detector(args["input"], args["output"], net, ln,
                         args["confidence"], args["skip"], args["threshold"])
