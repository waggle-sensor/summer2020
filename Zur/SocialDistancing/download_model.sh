#!/bin/bash -e

mkdir -p yolo-coco
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O yolo-coco/coco.names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolo-coco/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights -O yolo-coco/yolov3.weights
