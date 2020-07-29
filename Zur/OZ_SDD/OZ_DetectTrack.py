# USAGE
# python OZ_DetectTrack.py --input videos/test_video2.mp4 --output output/output_01.avi --yolo yolo-coco

# importing packages
from OZ_CentroidTracker import CentroidTracker
from OZ_NMS import non_max_suppression
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import datetime
import time
import dlib
import os
import cv2

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

# loading COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# deriving the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# loading YOLO object detector, determining only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initializing:
writer = None  # pointer to output file
(W, H) = (None, None)  # frame dimensions
trackers = []  # list to store dlib correlation trackers
totalFrames = 0  # total number of frames processed so far


# instantiate centroid tracker
ct = CentroidTracker(maxDisappeared=5, maxDistance=50)

# start the frames per second throughput estimator
fps_start_time = datetime.datetime.now()
fps = FPS().start()
FPS = 0

# looping over each frame of the video
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=750)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # initialize writer for output video
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # initialize current status and list of bounding boxes
    status = "Waiting"
    rects = []

    # DETECTION PHASE
    if totalFrames % args["skip_frames"] == 0:
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
                if classID == LABELS.index("person") and confidence > args["confidence"]:
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

        # apply non-maximum suppression algorithm
        bboxes = np.array(boxes).astype(int)
        boxes = non_max_suppression(bboxes, args["threshold"])

        if len(boxes) > 0:
            for box in boxes:
                # construct a dlib rectangle object from the bounding box coordinates
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(box[0], box[1], box[2], box[3])

                # start the dlib correlation tracker and add to list of trackers
                tracker.start_track(rgb, rect)
                trackers.append(tracker)

    # TRACKING PHASE
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
            rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    for(objectID, bbox) in objects.items():
        # draw bounding box and ID
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "ID: {}".format(objectID)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    text = "Status: " + status
    cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        FPS = 0.0
    else:
        FPS = (totalFrames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(FPS)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

cv2.destroyAllWindows()
