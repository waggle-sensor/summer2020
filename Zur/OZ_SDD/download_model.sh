#!/bin/bash -e

mkdir -p yolo-coco
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolo-coco/yolov3.cfg.download
wget https://pjreddie.com/media/files/yolov3.weights -O yolo-coco/yolov3.weights.download

# once *both* files have been successfully downloaded then rename
mv yolo-coco/yolov3.cfg.download yolo-coco/yolov3.cfg
mv yolo-coco/yolov3.weights.download yolo-coco/yolov3.weights