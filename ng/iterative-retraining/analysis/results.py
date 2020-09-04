from math import sqrt

import csv
import itertools
import statistics as stats
from sklearn.metrics import confusion_matrix

from retrain import utils


def load_data(output, by_actual=True, add_all=True, filter=None, conf_thresh=0.5):
    samples = dict()
    all_data = list()

    actual = list()
    pred = list()

    with open(output, newline="\n") as csvfile:
        reader = csv.DictReader(csvfile)

        if filter is not None:
            filter_list = utils.get_lines(filter)

        for row in reader:
            if filter is not None and row["file"] not in filter_list:
                continue
            actual.append(row["actual"])
            pred.append(row["detected"])
            all_data.append(row)

            key_val = (
                row["actual"]
                if by_actual or row["detected"] == str()
                else row["detected"]
            )

            if key_val in samples.keys():
                samples[key_val].append(row)
            else:
                samples[key_val] = [row]

    samples = {k: samples[k] for k in sorted(samples)}
    results = [ClassResults(k, v, conf_thresh=conf_thresh) for k, v in samples.items()]
    mat = confusion_matrix(actual, pred, labels=list(samples.keys()) + [""])

    if add_all:
        results.append(ClassResults("All", all_data, conf_thresh=conf_thresh))

    return results, mat


def mean_avg_conf(class_results):
    """Compute mean average confidence for a list of classes."""
    if len(class_results) == 0:
        return None
    return stats.mean(stats.mean(res.get_confidences()) for res in class_results)


def mean_conf_std(class_results):
    """Compute the mean standard deviation of the confidences of each class."""
    if len(class_results) == 0:
        return None
    class_vars = [stats.variance(res.get_confidences()) for res in class_results]
    return sqrt(stats.mean(class_vars))


def mean_avg_detect_conf_std(class_results):
    """Compute the mean average standard deviation for each class, based on the standard
    deviations of each image's bounding boxes confidence."""
    if len(class_results) == 0:
        return None
    mean_class_vars = list()
    for res in class_results:
        class_var = [conf ** 2 for conf in res.get_conf_stds()]
        mean_class_vars.append(stats.mean(class_var))
    return sqrt(stats.mean(mean_class_vars))


def mean_metric(class_results, metric):
    """Computes the mean of a given metric for a list of classes.

    Metric strings include precision, accuracy, and recall.
    """
    if len(class_results) == 0:
        return None
    return stats.mean([getattr(res, metric)() for res in class_results])


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
            try:
                row["conf_std"] = float(row["conf_std"])
            except (KeyError, ValueError):
                row["conf_std"] = 0
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

    def __len__(self):
        files = set()
        for row in self.get_all():
            files.add(row["file"])
        return len(files)

    def precision(self):
        predicted_cond_pos = (
            len(self.data["true_pos"]) + len(self.data["false_pos"]) + 1e-16
        )
        return len(self.data["true_pos"]) / predicted_cond_pos

    def recall(self):
        return len(self.data["true_pos"]) / (
            (len(self.data["true_pos"]) + len(self.data["false_neg"]) + 1e-16)
        )

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
        return list(itertools.chain.from_iterable(self.data.values()))

    def get_confidences(self, thresh=0.0):
        return [result["conf"] for result in self.get_all() if result["conf"] >= thresh]

    def get_conf_stds(self):
        return [result["conf_std"] for result in self.get_all()]

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

            if true_pos + false_pos != 0:
                precision = true_pos / (true_pos + false_pos)
                out.write(f"{x},{precision},{true_pos+false_pos}\n")

            x += delta
        out.close()
