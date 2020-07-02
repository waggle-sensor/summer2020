import yolov3
import yolov3.detect as detect
import yolov3.models as models
import yolov3.utils.datasets as datasets
import yolov3.utils.utils as utils

import torch
from torch.utils.data import DataLoader

import pandas as pd
import csv

"""
Runs initial test of object detection models on
the segmented letters from the KAIST dataset
"""

if __name__ == "__main__":
    model = models.get_eval_model(
        "yolov3/config/yolov3.cfg", 416, "checkpoints/yolov3_ckpt_14.pth"
    )

    classes = utils.load_classes("output/chars.names")

    loader = DataLoader(
        datasets.ImageFolder("data/objs/", img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    hits_misses = dict()
    confusion_mat = dict()

    header = "file,actual,detected,conf,hit".split(",")
    output = open("output/benchmark.csv", "w+")
    writer = csv.DictWriter(output, fieldnames=header)

    for (img_paths, input_imgs) in loader:
        props = dict()
        props["file"] = img_paths[0]
        props["actual"] = img_paths[0].split("-")[1][:1]

        detections = detect.detect(input_imgs, 0.5, model)

        # conf is the confident that it's an object
        # cls_conf is the confidence of the classification
        most_conf = detect.get_most_conf(detections)

        if most_conf is not None:
            (_, _, _, _, conf, cls_conf, cls_pred) = most_conf.numpy()[0]

            props["detected"] = classes[int(cls_pred)]
            props["conf"] = cls_conf if cls_conf is not None else 0.00
            props["hit"] = props["actual"] == props["detected"]
        else:
            props["detected"] = ""
            props["conf"] = 0.0
            props["hit"] = False

        writer.writerow(props)

    output.close()
