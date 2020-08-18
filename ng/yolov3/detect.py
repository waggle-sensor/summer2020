from __future__ import division

import os
import time
import datetime
import argparse

from torch.utils.data import DataLoader

import yolov3.models as models
import yolov3.utils.utils as utils
import yolov3.evaluate as evaluate
from yolov3.utils.datasets import ImageFolder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder", type=str, default="data/samples", help="path to dataset"
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="config/yolov3.cfg",
        help="path to model definition file",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights/yolov3.weights",
        help="path to weights file",
    )
    parser.add_argument(
        "--class_path",
        type=str,
        default="data/coco.names",
        help="path to class label file",
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.8, help="object confidence threshold"
    )
    parser.add_argument(
        "--nms_thres",
        type=float,
        default=0.4,
        help="iou thresshold for non-maximum suppression",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=0,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--img_size", type=int, default=416, help="size of each image dimension"
    )
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("output", exist_ok=True)

    model = models.get_eval_model(opt.model_def, opt.img_size, opt.weights_path)

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = utils.load_classes(opt.class_path)  # Extracts class labels from file
    print(classes)
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        detections = evaluate.detect(input_imgs, opt.conf_thres, model)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)



        evaluate.save_images(img_paths, detections, opt, classes, False)
