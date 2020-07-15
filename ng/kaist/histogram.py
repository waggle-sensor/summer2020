import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
import utils
import statistics as stats
import benchmark as bench

OUTPUT = "output/"

HIT_MIN = 0.6
STACKED_MIN = 0.90


def get_conf_data(result_list):
    return [row["conf"] for row in result_list]


def generate_hist(results, filename="hist.pdf", mean_stats=True):
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

        axs[i][0].hist(hit_miss[0], bins=10, color=colors[0], range=(HIT_MIN, 1.0))
        axs[i][1].hist(hit_miss[1], bins=20, color=colors[1], range=(0.0, 1.0))
        axs[i][2].hist(
            hit_miss, bins=10, color=colors, range=(STACKED_MIN, 1.0), stacked=True
        )

        if res.name == "All" and mean_stats:
            acc = round(bench.mean_accuracy(results[:-1]), 3)
            prec = round(bench.mean_precision(results[:-1]), 3)
        else:
            acc = round(res.accuracy(), 3)
            prec = round(res.precision(), 3)

        title = f"Class: {res.name} (acc={acc}, " + f"prec={prec}, n={res.pop})"
        axs[i][1].set_title(title)

    fig.set_figheight(2.5 * num_rows)
    fig.set_figwidth(10)
    fig.savefig(OUTPUT + "/" + filename, bbox_inches="tight")


if __name__ == "__main__":
    """Generates PDF histograms of hit and miss confidences.

    Usage: python3 histogram.py output/benchmark.csv
    """

    if "benchmark_" in sys.argv[1]:
        check_str = sys.argv[1].split("benchmark_")[1][:-4]
        suffix = f"_{check_str}.pdf"
    else:
        check_str = str()
        suffix = ".pdf"

    OUTPUT = "/".join(sys.argv[1].split("/")[:-1])
    print(OUTPUT)

    results, mat = utils.load_data(sys.argv[1], by_actual=False)
    generate_hist(results, filename="hist_by_pred" + suffix)
    results[-1].generate_prec_distrib(f"output/all_prec_{check_str}.csv", 0.01)

    results, _ = utils.load_data(sys.argv[1], by_actual=True)
    generate_hist(results, filename="hist_by_actual" + suffix)

    names = [res.name for res in results if res.name != "All"] + [""]
    df = pd.DataFrame(mat, index=names, columns=names)
    df.to_csv("output/confusion" + suffix[:-4] + ".csv")
    print(df)
