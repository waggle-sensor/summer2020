import numpy as np
from typing import Dict
from Classifiers import TextureTemporalClassifier
import pickle
import time
import argparse
from collections import deque
from waggle import plugin
from waggle.data.vision import Camera
from pathlib import Path


models = {50: 'tt_classifier_50fps.model',
          5: 'tt_classifier_5fps.model',
          1: 'tt_classifier_1fps.model'}

model_data = {rate: pickle.load(open(filename, 'rb')) for rate, filename in models.items()}
model_data: Dict[TextureTemporalClassifier]
plugin.init()


def get_water_mask(camera: Camera, model: TextureTemporalClassifier):
    N_FRAMES_MAX_BUFFER = 60
    test_framerate = time.time_ns()
    frame_buffer = deque(maxlen=N_FRAMES_MAX_BUFFER)
    for sample in camera.stream():
        frame_buffer.append(sample.data)
        if len(frame_buffer) == N_FRAMES_MAX_BUFFER:
            break
    time_to_accumulate_s = (float(time.time_ns()) - test_framerate) / (10**9)
    fps = 1.0 / (time_to_accumulate_s / N_FRAMES_MAX_BUFFER)
    print("FPS: %f" % fps)


def batch_frame_from_camera(camera: Camera, framerate: int) -> np.ndarray:
    """
    The purpose of this function is to read 60 frames from the camera at a specified framerate. The framerate of this
    buffering is important, because I have trained my TextureTemporal models on a certain framerate of video. The models
    have learned the flickering motion of water from the specific spacing of frames that they have been given, i.e. 50fps,
    5fps, 1fps.

    Returns an ndarray of the size [800, 600]
    """
    pass


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--input', required=True,
                       help='Input video or frame folder to give to the classifier. To inference by using the camera,'
                            'set this option as \'camera.\'')
    parse.add_argument('-o', '--output', required=True,
                       help='Output folder which will hold corresponding water masks')
    parse.add_argument('-r', '--rate', required=False, type=int, default=5,
                       help='Specify the model to use by average FPS: 1, 5, or 50. Note that this does not mean that the'
                            'input video fps has to match the model perfectly; just select the closest option for your'
                            'use case.')
    parse.add_argument('--heatmap', required=False, type=bool, default=False,
                       help='Output the segmentation images as a heatmap.')
    args = parse.parse_args()

    # Run the specified classifier on the image stream once
    if args.input == 'camera':
        print('Running water segmentation with camera...')
        cam = Camera()
        input_frames = batch_frame_from_camera(cam, args.rate)  # TODO This function needs to be written

        # From this point, I do not know the best way to deal with the image output. Should it be published? Stored in
        # a buffer? I don't know. One possible metric that could be extracted from the segmentation image is a "water
        # surface area" measurement. This would be helpful in conjunction with water level analysis to determine flooding
        # seriousness.
        segmentation_array = model_data[args.rate].segment(input_frames, prob_mode=args.heatmap)
