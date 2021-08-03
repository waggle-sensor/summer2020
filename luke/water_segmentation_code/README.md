## Water Segmentation Plugin

This plugin is designed to highlight water in a sequence of frames. It can accept a Camera input or a video stream input.
*As of this moment the Camera input feature has not yet been added.*

The model accepts an array of shape [N videos, 60 frames, 800 pixels, 600 pixels, 3 color channels] where N is the 
number of separate videos to process. Note that the 60 frames must be spaced out in time according to the model's framerate.
This means that if you are inferencing using the 5fps model (the model I recommend- it is not too resource intensive to
buffer while still retaining accuracy), then make sure to feed it 60 frames spaced out according to 5fps timing, that is 
0.2s between each frame capture. This means that the model would receive 60 frames over the course of 12s and would output one
water mask from that motion data.

I followed the general algorithm described by [this paper](https://staff.fnwi.uva.nl/p.s.m.mettes/papers/water-detection-cviu-final.pdf).

### Setup

The inference entrypoint of the script is `WaterSegmentationInference.py`. This is where stored models (`tt_classifier_*.model`)
are loaded and arguments are parsed. This script still needs some work; it lacks a function that can buffer frames from 
the camera.
