from __future__ import division

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def detect(input_imgs, conf_thres, model, nms_thres=0.4):
    # Configure input
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = Variable(input_imgs.type(Tensor))

    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    return detections


def save_images(imgs, img_detections, opt, best_label_only=False):
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            save_image(detections, path, opt, best_label_only)


def get_most_conf(detections):
    most_conf = None
    for d in detections:
        if most_conf is None or d[5] > most_conf[5]:
            most_conf = d
    return most_conf


def save_image(detections, path, opt, best_label_only=False):
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Create plot
    img = np.array(Image.open(path))
    fig_main = plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Rescale boxes to original image
    detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)

    if best_label_only:
        detections = [get_most_conf(detections)]

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle(
            (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none",
        )
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s="\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()),
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = path.split("/")[-1].split(".")[0]
    plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig_main)
    plt.close(fig)


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

    model = get_eval_model(opt.model_def, opt.img_size, opt.weights_path)

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        detections = detect(input_imgs, opt.conf_thres, model)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    save_images(imgs, img_detections, opt, True)
