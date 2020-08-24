import csv
import statistics as stats
import itertools
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


def mean_conf(class_results):
    """Computes mean average confidence for a list of classes"""
    return stats.mean(stats.mean(res.get_confidences()) for res in class_results)


def mean_precision(class_results):
    """Computes mean precision for a list of classes, which shouldn't include All."""
    return stats.mean([res.precision() for res in class_results])


def mean_accuracy(class_results):
    return stats.mean([res.accuracy() for res in class_results])


def mean_recall(class_results):
    return stats.mean([res.recall() for res in class_results])


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
            row["conf_std"] = float(row["conf_std"])
            if row["conf"] >= conf_thresh:
                if row["hit"] == "True":
                    result = "true_pos"
                else:
                    result = "false_pos"
            else:
                if row["hit"] == "True" or row["detected"] == str():
                    result = "false_neg"
                else:
                    result = "true_neg"
            if result != "true_neg":
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
