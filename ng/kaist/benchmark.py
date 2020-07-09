import yolov3.evaluate as evaluate
import yolov3.models as models
import yolov3.utils.datasets as datasets
import yolov3.utils.utils as utils
import yolov3.utils.parse_config as parser

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import pandas as pd
import csv
import itertools
import sys

"""
Runs initial test of object detection models on
the segmented letters from the KAIST dataset

Contains helper methods to parse generated output data.
"""


def load_data(output, by_actual=True):
    samples = dict()
    all_data = list()

    actual = list()
    pred = list()

    with open(output, newline="\n") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            actual.append(row["actual"])
            pred.append(row["detected"])

            key_val = row["actual"] if by_actual else row["detected"]
            if key_val == str():
                continue
            if key_val not in samples.keys():
                samples[key_val] = [row]
            else:
                samples[key_val].append(row)
            all_data.append(row)
    samples = {k: samples[k] for k in sorted(samples)}
    results = [ClassResults(k, v) for k, v in samples.items()]
    mat = confusion_matrix(actual, pred, labels=list(samples.keys()) + [""])

    results.append(ClassResults("All", all_data))

    return results, mat


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
            if float(row["conf"]) >= conf_thresh:
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


if __name__ == "__main__":
    check_num = int(sys.argv[1])

    options = parser.parse_model_config("config/yolov3.cfg")[0]
    img_size = max(int(options["width"]), int(options["height"]))

    model = models.get_eval_model(
        "config/yolov3.cfg", img_size, f"checkpoints/yolov3_ckpt_{check_num}.pth"
    )

    classes = utils.load_classes("config/chars.names")

    loader = DataLoader(
        datasets.ImageFolder("data/objs/", img_size=img_size),
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    hits_misses = dict()
    confusion_mat = dict()

    header = "file,actual,detected,conf,hit".split(",")
    output = open(f"output/benchmark_{check_num}.csv", "w+")
    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()
    for (img_paths, input_imgs) in loader:
        props = dict()
        props["file"] = img_paths[0]
        props["actual"] = img_paths[0].split("-")[1][:1]

        detections = evaluate.detect(input_imgs, 0.5, model)

        # conf is the confident that it's an object
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
