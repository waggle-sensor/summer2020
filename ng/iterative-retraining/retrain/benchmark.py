import yolov3.evaluate as evaluate
import yolov3.models as models
import retrain.utils as utils
import statistics as stats

import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from retrain.dataloader import ListDataset

import pandas as pd
import numpy as np
import csv
import itertools

from sklearn.metrics import confusion_matrix


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

    def get_confidences(self, thresh=0.0):
        return [result["conf"] for result in self.get_all() if result["conf"] >= thresh]

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


def benchmark_avg(img_folder, prefix, start, end, total, config, model_def):
    loader = DataLoader(
        img_folder, batch_size=1, shuffle=False, num_workers=config["n_cpu"],
    )

    results = pd.DataFrame(
        columns=["file", "confs", "actual", "detected", "conf", "hit"]
    )
    results.set_index("file")

    classes = utils.load_classes(config["class_list"])

    checkpoints_i = set(np.linspace(start, end, total, dtype=np.dtype(np.int16)))
    print("Benchmarking on epochs", checkpoints_i)
    checkpoints = list()

    for n in tqdm(checkpoints_i, "Benchmarking epochs"):
        ckpt = f"{config['checkpoints']}/init_ckpt_{n}.pth"
        if not os.path.exists(ckpt):
            ckpt = glob.glob(f"{config['checkpoints']}/{prefix}*_ckpt_{n}.pth")[0]

        model = models.get_eval_model(model_def, config["img_size"], ckpt)

        for (img_paths, input_imgs) in loader:
            path = img_paths[0]
            if path not in results.file:
                actual_class = classes[
                    img_folder.get_classes(utils.get_label_path(path))[0]
                ]
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
            avg_conf = sum(confs) / len(checkpoints_i)

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

    filename = (
        f"{start}.csv" if total == 1 else f"{prefix}_benchmark_avg_{start}_{end}.csv"
    )
    out_path = f"{config['output']}/{filename}"
    output = open(out_path, "w+")

    results.to_csv(
        output, columns=["file", "actual", "detected", "conf", "hit"], index=False
    )
    output.close()

    return out_path
