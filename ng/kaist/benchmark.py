import yolov3.evaluate as evaluate
import yolov3.models as models
import yolov3.utils.datasets as datasets
import yolov3.utils.utils as yoloutils
import yolov3.utils.parse_config as parser
import utils

import os
import torch
import tqdm
from torch.utils.data import DataLoader

import pandas as pd
import csv
import itertools
import sys

"""
Runs initial test of object detection models on
the segmented letters from the KAIST dataset

Contains helper methods to parse generated output data.
"""

OUTPUT = "./output/"
ORIG_DATA = "../yolov3/data/"


class ClassResults:
    def __init__(self, name, output_rows, conf_thresh=0.5):
        self.name = name
        self.condition = ["pos", "neg"]
        self.actual = ["true", "false"]
        self.data = dict()
        self.pop = 0

        for actual in self.actual:
            for cond in self.condition:
                self.data[f"{actual}_{cond}"] = list()

        for row in output_rows:
            row["conf"] = float(row["conf"])
            if row["conf"] >= conf_thresh:
                if row["hit"] == "True":
                    result = "true_pos"
                else:
                    result = "false_pos"
            else:
                if row["hit"] == "True":
                    result = "false_neg"
                else:
                    result = "true_neg"
            self.data[result].append(row)
            self.pop += 1

    def precision(self):
        try:
            predicted_cond_pos = len(self.data["true_pos"]) + len(
                self.data["false_pos"]
            )
            return len(self.data["true_pos"]) / predicted_cond_pos
        except ZeroDivisionError:
            return 0.0

    def accuracy(self):
        return (len(self.data["true_pos"]) + len(self.data["true_neg"])) / self.pop

    def hits_misses(self):
        """Get a split list of hits and misses."""
        all_results = [list(), list()]
        for k, v in self.data.items():
            if k in ("true_pos", "false_neg"):
                all_results[0] += v
            else:
                all_results[1] += v
        return all_results

    def get_all(self):
        return list(itertools.chain.from_iterable(self.hits_misses()))

    def get_confidences(self):
        return [result["conf"] for result in self.get_all()]

    def generate_prec_distrib(self, output, delta=0.05):
        """Generate a spreadsheet of confidence range vs. rolling precision."""
        out = open(output, "w+")
        out.write("conf,rolling precision\n")
        x = delta / 2
        while x < 1.00 + (delta / 2):
            true_pos = len(
                [
                    d
                    for d in self.data["true_pos"]
                    if x + (delta / 2) > d["conf"] >= x - (delta / 2)
                ]
            )
            false_pos = len(
                [
                    d
                    for d in self.data["false_pos"]
                    if x + (delta / 2) > d["conf"] >= x - (delta / 2)
                ]
            )

            try:
                precision = true_pos / (true_pos + false_pos)
                out.write(f"{x},{precision},{true_pos+false_pos}\n")
            except ZeroDivisionError:
                x += delta
                continue

            x += delta
        out.close()


def test(model, classes, img_size, valid_path, check_num):
    """Tests weights against the test data set."""

    class Options(object):
        pass

    opt = Options()
    opt.iou_thres = 0.5
    opt.conf_thres = 0.5
    opt.nms_thres = 0.5
    opt.img_size = img_size

    utils.rewrite_test_list(valid_path, ORIG_DATA)
    utils.save_stdout(
        OUTPUT + f"mAP_{check_num}.txt",
        evaluate.get_results,
        model,
        valid_path.replace(".txt", "-new.txt"),
        opt,
        classes,
    )


def benchmark(
    check_prefix, check_num, config, data_config, classes, sample_dir, out_name=None
):
    options = parser.parse_model_config("config/yolov3.cfg")[0]
    data_opts = parser.parse_data_config("config/chars.data")

    img_size = max(int(options["width"]), int(options["height"]))

    model = models.get_eval_model(
        config, img_size, f"checkpoints/{check_prefix}_ckpt_{check_num}.pth"
    )

    classes = yoloutils.load_classes(classes)

    test(model, classes, img_size, data_opts["valid"], check_num)

    loader = DataLoader(
        datasets.ImageFolder(sample_dir, img_size=img_size),
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    hits_misses = dict()
    confusion_mat = dict()

    header = "file,actual,detected,conf,hit".split(",")
    if out_name is not None:
        output = open(OUTPUT + f"{out_name}_{check_num}.csv", "w+")
    else:
        output = open(OUTPUT + f"benchmark_{check_num}.csv", "w+")
    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()

    for (img_paths, input_imgs) in tqdm.tqdm(loader, "Inferencing on samples"):
        props = dict()
        props["file"] = img_paths[0]
        props["actual"] = img_paths[0].split("-")[1][:1]

        detections = evaluate.detect(input_imgs, 0.5, model)

        # conf is the confidence that it's an object
        # cls_conf is the confidence of the classification
        most_conf = evaluate.get_most_conf(detections)

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


if __name__ == "__main__":
    check_num = int(sys.argv[2])
    benchmark(
        sys.argv[1],
        check_num,
        "config/yolov3.cfg",
        "config/chars.data",
        "config/chars.names",
        "data/images/objs/",
    )
