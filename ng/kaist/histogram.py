import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix


def load_data(output):
    samples = dict()
    all_data = list()

    actual = list()
    pred = list()

    with open(output, newline="\n") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            actual_class = row["actual"]
            if actual_class not in samples.keys():
                samples[actual_class] = [row]
            else:
                samples[actual_class].append(row)
            all_data.append(row)
            actual.append(actual_class)
            pred.append(row["detected"])
    results = [ClassResults(k, v) for k, v in samples.items()]
    mat = confusion_matrix(actual, pred, labels=list(samples.keys()))

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
        predicted_cond_pos = len(self.data["true_pos"]) + len(self.data["false_pos"])
        return len(self.data["true_pos"]) / predicted_cond_pos

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


def get_conf_data(result_list):
    return [float(row["conf"]) for row in result_list]


def generate_graphs(results):
    num_rows = len(results)
    fig, axs = plt.subplots(num_rows, 3)
    plt.subplots_adjust(hspace=0.35)

    graphs = ["hit", "miss", "all"]
    all_data = dict()
    for name in graphs:
        all_data[name] = list()

    colors = ["lightgreen", "red"]
    for i, res in enumerate(results):
        hit_miss = [get_conf_data(data) for data in res.hits_misses()]

        axs[i][0].hist(hit_miss[0], bins=10, color=colors[0], range=(0.9, 1.0))
        axs[i][1].hist(hit_miss[1], bins=20, color=colors[1], range=(0.0, 1.0))
        axs[i][2].hist(hit_miss, bins=10, color=colors, range=(0.8, 1.0), stacked=True)

        axs[i][1].set_title(
            f"Class: {res.name} (acc={round(res.accuracy(), 3)}, pres={round(res.precision(), 3)}, n={res.pop})"
        )

    fig.set_figheight(20)
    fig.set_figwidth(10)
    fig.savefig("output/hist.pdf", bbox_inches="tight")


if __name__ == "__main__":
    results, mat = load_data(sys.argv[1])
    generate_graphs(results)
    names = [res.name for res in results if res.name != "All"]
    df = pd.DataFrame(mat, index=names, columns=names)
    print(df)
