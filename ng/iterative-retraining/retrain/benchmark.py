import retrain.evaluate as evaluate
import retrain.models as models
import yolov3.utils.datasets as datasets
import yolov3.utils.utils as yoloutils
import yolov3.utils.parse_config as yoloparser
import statistics as stats
import argparse

import os
import torch
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader

import pandas as pd
import csv
import itertools
import sys


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
    results = [benchmark.ClassResults(k, v) for k, v in samples.items()]
    mat = confusion_matrix(actual, pred, labels=list(samples.keys()) + [""])

    results.append(benchmark.ClassResults("All", all_data))

    return results, mat


def mean_precision(class_results):
    """Computes mean precision for a least of classes, which shouldn't include All."""
    return stats.mean([res.precision() for res in class_results])


def mean_accuracy(class_results):
    return stats.mean([res.accuracy() for res in class_results])


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


def test(model, classes, img_size, valid_path, check_num, out_folder, silent=False):
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
        out_folder + f"/mAP_{check_num}.txt",
        evaluate.get_results,
        model,
        valid_path.replace(".txt", "-new.txt"),
        opt,
        classes,
        silent=silent,
    )


def benchmark(
    prefix, check_num, config, data_config, classes, sample_dir, out=None, silent=False
):
    benchmark_avg(
        prefix,
        check_num,
        check_num,
        1,
        config,
        data_config,
        classes,
        sample_dir,
        out,
        silent,
    )


def benchmark_avg(
    prefix, start, end, total_check, classes, out,
):
    options = yoloparser.parse_model_config("config/yolov3.cfg")[0]
    data_opts = yoloparser.parse_data_config("config/chars.data")

    img_size = max(int(options["width"]), int(options["height"]))

    classes = yoloutils.load_classes(classes)
    out_folder = "/".join(out.split("/")[:-1]) if out is not None else OUTPUT

    loader = DataLoader(
        datasets.ImageFolder(sample_dir, img_size=img_size),
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    results = pd.DataFrame(
        columns=["file", "confs", "actual", "detected", "conf", "hit"]
    )
    results.set_index("file")

    epochs_tested = int((end - start + 1) / delta)
    epoch_iter = epochs_tested == 1

    orig_prefix = prefix
    for check_num in tqdm(
        range(start, end + 1, delta), "Benchmarking epochs", disable=epoch_iter
    ):

        prefix = "yolov3" if check_num <= 74 else orig_prefix

        model = models.get_eval_model(
            config, img_size, f"checkpoints/{prefix}_ckpt_{check_num}.pth"
        )

        test(
            model, classes, img_size, data_opts["valid"], check_num, out_folder, silent
        )

        for (img_paths, input_imgs) in tqdm(
            loader, "Inferencing on samples", disable=silent
        ):
            path = img_paths[0]
            if path not in results.file:
                actual_class = path.split("-")[1][:1]
                results.loc[path] = [path, dict(), actual_class, None, None, None]

            detections = evaluate.detect(input_imgs, 0.5, model)

            confs = results.loc[path]["confs"]

            for detection in detections:
                if detection is None:
                    continue
                (_, _, _, _, conf, cls_conf, cls_pred) = detection.numpy()[0]

                if cls_pred not in confs.keys():
                    confs[cls_pred] = [cls_conf]

                else:
                    confs[cls_pred].append(cls_conf)

    for i, row in results.iterrows():
        best_class = None
        best_conf = float("-inf")

        for class_name, confs in row["confs"].items():
            avg_conf = sum(confs) / epochs_tested

            if avg_conf > best_conf:
                best_conf = avg_conf
                best_class = class_name

        if best_class is not None:
            row["detected"] = classes[int(best_class)]
            row["conf"] = best_conf
            row["hit"] = row["actual"] == row["detected"]
        else:
            row["detected"] = ""
            row["conf"] = 0.0
            row["hit"] = False

    prefix = out if out is not None else OUTPUT
    suffix = f"{start}.csv" if epochs_tested == 1 else f"avg_{start}_{end}.csv"
    output = open(prefix + suffix, "w+")

    results.to_csv(
        output, columns=["file", "actual", "detected", "conf", "hit"], index=False
    )
