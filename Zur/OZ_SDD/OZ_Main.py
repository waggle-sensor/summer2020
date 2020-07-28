# import packages
import cv2
import numpy as np
import argparse
import datetime
import time
import dlib
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from OZ_CentroidTracker import CentroidTracker
from OZ_NMS import non_max_suppression
import OZ_HelperFunctions


mouse_pts = []


# mouse callback function, takes 8 clicks
# 1-4: rectangle region of interest (red circles)
#   Points must be clicked in the following order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
# 5-7: define 6 feet in vertical and horizontal direction
#   Points 5 and 6 form a horizontal line and points 5 and 7 form a vertical line
# 8: break
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


def social_distance_detector(vid_input, vid_output, yolo_net, layerNames, confid, skip, threshold):
    vs = cv2.VideoCapture(vid_input)
    global image

    # initializing:
    writer = None
    (W, H) = (None, None)
    trackers = []
    frameCount = 0
    points = []

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    scale_w, scale_h = float(400 / width), float(600 / height)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps,
                                 (int(width * scale_w), int(height * scale_h)))

    # instantiate centroid tracker
    ct = CentroidTracker(maxDisappeared=5, maxDistance=50)

    # looping over each frame of video input
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # grabbing frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # pulling first frame of the video and waiting for mouse click inputs
        if frameCount == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break
            points = mouse_pts

        # **IMAGE TRANSFORMATION** ####################################################################################

        # perspective transformation using mouse click inputs
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # calculating number of pixels that make up approx. 6 feet
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

        # **PERSON DETECTION AND TRACKING** ###########################################################################
        status = "Waiting"
        boundingboxes = []

        # Detection
        # updating status and initializing list of trackers
        status = "Detecting"
        trackers = []

        # passing frame through YOLO detector
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initializing lists of detected bounding boxes
        boxes = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filtering detections
                if classID == 0 and confidence > confid:
                    # scale the bounding box coordinates back relative to the size of the image
                    # *Note: YOLO  returns the center (x, y)-coordinates of bbox, then width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype(int)

                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (width / 2))
                    endX = int(centerX + (width / 2))
                    endY = int(centerY + (width / 2))
                    box = (startX, startY, endX, endY)

                    # update list of bounding boxes
                    boxes.append(box)


if __name__ == "__main__":
    # constructing argument parser and parsing arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
                    help="path to input video")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video")
    ap.add_argument("-y", "--yolo", required=True,
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applying non-maximum suppression")
    args = vars(ap.parse_args())

    # deriving the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # loading YOLO object detector
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # calling mouse callback
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)

    # calling SDD function
    social_distance_detector(args["input"], args["output"], net, ln,
                             args["confidence"], args["skip-frames"], args["threshold"])
