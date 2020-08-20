# Usage
# python sdd.py --input videos/test_video2.mp4
# python sdd.py --input videos/pedestrians.mp4
# python sdd.py --input videos/test_video4.mp4

# import packages
import cv2
import numpy as np
import argparse
import os
import imutils
import time
import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict
from helpers import *
from centroid_tracker import CentroidTracker

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
def social_distance_detector(vid_input, yolo_net, layerNames, confid, threshold, disappeared, frames):
    # start timer
    start = time.time()

    # if no video input, use computer webcam
    if vid_input is None:
        print("Loading Webcam...")
        vs = cv2.VideoCapture(0)
        time.sleep(2.0)
    else:
        print("Loading Video...")
        vs = cv2.VideoCapture(vid_input)

    # initializing:
    (H, W) = (None, None)  # frame dimensions
    frameCount = 0  # keep track of number of frames analyzed so far
    points = []  # list of mouse input coordinates
    time_series = OrderedDict()  # ordered dictionary for data collection
    ct = CentroidTracker(maxDisappeared=disappeared)  # centroid tracker

    # initialize output text file
    out = open("out.txt", "w")

    # **LOOP OVER EVERY FRAME OF THE VIDEO** ##########################################################################
    while True:
        (grabbed, frame) = vs.read()
        # frame = frame[1] if vid_input is None else frame

        # break if using video input and no frame left to grab
        if args["input"] is not None and frame is None:
            print("End of Video File")
            break

        # resizing frames for consistency, regardless of size of input video
        frame = imutils.resize(frame, width=1250)

        # grabbing frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # pulling first frame of the video and waiting for mouse click inputs
        if frameCount == 0:
            global image
            while True:
                image = frame
                # text instructions for correct order of input points
                text1 = "Click on 4 ROI points, counter-clockwise starting bottom-left"
                text2 = "Now click on 2 points for 6-foot approximation"
                text3 = "Click anywhere to start the video"
                cv2.putText(image, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) >= 4:
                    cv2.putText(image, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                if len(mouse_pts) == 6:
                    cv2.putText(image, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("image")
                    break
            points = mouse_pts

            # **IMAGE TRANSFORMATION** ################################################################################

            # creating transformation matrix from first four mouse input points
            src = np.float32(np.array(points[:4]))
            dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
            matrix = cv2.getPerspectiveTransform(src, dst)

            # warping mouse points 5 - 6 using transformation matrix
            pts = np.float32(np.array([points[4:6]]))
            warped_pt = cv2.perspectiveTransform(pts, matrix)[0]

            # calculating number of pixels that make up approx. 6 feet
            safe_dist = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)

            # write min safe distance to output text file
            out.write("6 feet = " + str(int(safe_dist)) + " pixels." + "\n" + "\n")

        # drawing the rectangle on the video for the remaining frames
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (0, 0, 0), thickness=2)

        # **YOLO DETECTION** ##########################################################################################

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

        # call update function in centroid tracker, draw unique ID above bounding box
        objects = ct.update(boundingboxes)
        for (objectID, bbox) in objects.items():
            x1, y1, w, h = bbox
            text = "ID: {}".format(objectID)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        boundingboxes = [(k, v) for k, v in objects.items()]

        # **COORDINATE MAPPING AND DISTANCE CALCULATION** #############################################################

        # no need to do any analysis if no people detected
        if len(boundingboxes) == 0:
            frameCount = frameCount + 1
            continue

        # transform bottom center point of each bounding box
        bottom_points = transform_box_points(boundingboxes, matrix)

        # calculate distances between each pair of points, compare to minimum safe distance
        distance_pairs, box_pairs = violation_detection(boundingboxes, bottom_points, safe_dist)

        # count number of violations detected
        violation_count, safe_count = get_violation_count(distance_pairs)

        # **OUTPUT DISPLAY AND CLEAN UP** #############################################################################

        # create blank window for bird's eye view, plot transformed points as green and red circles
        scale_w, scale_h = float(400 / W), float(600 / H)
        bird_view = bird_output(frame, distance_pairs, scale_w, scale_h)

        # draw red or green rectangles on each detected person in frame
        copy = np.copy(frame)
        street_view = street_output(copy, boundingboxes, box_pairs, violation_count, safe_count)

        if frameCount != 0:
            # show current frame
            cv2.imshow('Street View', street_view)
            cv2.imshow('Bird Eye View', bird_view)

        # set current time and calculate social distancing ratio
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sdd_ratio = violation_count / len(boundingboxes)

        # every (default) 20 frames, output current frame data to text file and append to ordered dictionary
        if (frameCount % frames == 0) and (frameCount != 0):
            out.write("Current time: " + str(current_time) + "\n")
            out.write("Pedestrians Detected: " + str(len(boundingboxes)) + "\n")
            for (objectID, point) in bottom_points:
                out.write("\t" + "Object ID: {}".format(objectID) + str(point) + "\n")
            out.write("Safe Pedestrians: " + str(safe_count) + "\n")
            out.write("Violator Pedestrians: " + str(violation_count) + "\n")
            out.write("Social Distancing Ratio: " + str(sdd_ratio) + "\n")
            out.write("\n")

            time_series[current_time] = sdd_ratio

        # update frame count
        frameCount = frameCount + 1

        # press 'q' to end program before video ends
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release video capture
    vs.release()

    """
    # organize ordered dictionary into two numpy arrays
    tuple_list = list(time_series.items())
    x, y = zip(*tuple_list)
    xx = np.array(x)
    yy = np.array(y)
    print(xx)
    print(yy)

    # plotting sdd ratio vs. time
    plt.figure()
    plt.plot(xx, yy, marker='o')
    plt.show()
    """

    # destroy all windows
    cv2.destroyAllWindows()

    # output elapsed time and close output text file
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    out.write("Total Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    out.close()


# **ARGUMENT PARSER, LOAD YOLO, CALL SDD FUNCTION** ###################################################################

# constructing argument parser and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
                help="path to input video")
ap.add_argument("-y", "--yolo", default="yolo-coco",
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maximum suppression")
ap.add_argument("-d", "--disappeared", type=int, default=10,
                help="number of frames that an object ID is not detected before it is de-registered")
ap.add_argument("-f", "--frames", type=int, default=20,
                help="number of frames between data output to text file")
args = vars(ap.parse_args())

# deriving the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# loading YOLO object detector
print("Loading YOLO From Disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# calling mouse callback function
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)

# calling SDD function
social_distance_detector(args["input"], net, ln, args["confidence"], args["threshold"],
                         args["disappeared"], args["frames"])
