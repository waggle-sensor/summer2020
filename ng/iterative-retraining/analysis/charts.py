import os
from glob import glob

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression

from retrain import utils
import analysis.results as rload


def linear_regression(df):
    x = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    print("R^2:", linear_regressor.score(x, y))
    y_pred = linear_regressor.predict(x)

    plt.scatter(x, y)
    plt.xlabel(f"sample {df.columns[0]}")
    plt.ylabel(f"next iteration benchmark {df.columns[1]}")
    plt.plot(x, y_pred, color="r")
    plt.show()

    x_sm = sm.add_constant(x)
    est = sm.OLS(y, x_sm).fit()
    print(est.summary())


def make_conf_histogram(results, filename):
    num_rows = len(results)
    fig, axs = plt.subplots(num_rows, 3)
    plt.subplots_adjust(hspace=0.35)

    graphs = ["hit", "miss", "all"]
    all_data = dict()
    for name in graphs:
        all_data[name] = list()

    colors = ["lightgreen", "red"]
    for i, res in enumerate(results):
        hit_miss = [[row["conf"] for row in data] for data in res.hits_misses()]

        axs[i][0].hist(hit_miss[0], bins=15, color=colors[0], range=(0, 1))
        axs[i][1].hist(hit_miss[1], bins=15, color=colors[1], range=(0, 1))
        axs[i][2].hist(hit_miss, bins=15, color=colors, stacked=True, range=(0, 1))

        if res.name == "All":
            acc = round(rload.mean_metric(results[:-1], "accuracy"), 3)
            prec = round(rload.mean_metric(results[:-1], "precision"), 3)
        else:
            acc = round(res.accuracy(), 3)
            prec = round(res.precision(), 3)

        title = f"Class: {res.name} (acc={acc}, " + f"prec={prec}, n={res.pop})"
        axs[i][1].set_title(title)

    fig.set_figheight(2.5 * num_rows)
    fig.set_figwidth(10)
    fig.savefig(filename, bbox_inches="tight")
    plt.clf()


def plot_multiline(xy_pairs, xlab=str(), ylab=str(), vert_lines=None):
    fig, ax = plt.subplots()
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    lines = list()
    for (x_coords, y_coords, label) in xy_pairs:
        line = ax.plot(x_coords, y_coords, label=label)
        lines.append(line)

    leg = ax.legend()
    line_dict = dict()
    for leg_line, orig_line in zip(leg.get_lines(), lines):
        leg_line.set_picker(10)
        line_dict[leg_line] = orig_line

    def onpick(event):
        leg_line = event.artist
        orig_line = line_dict[leg_line][0]
        vis = not orig_line.get_visible()
        orig_line.set_visible(vis)
        if vis:
            leg_line.set_alpha(1.0)
        else:
            leg_line.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", onpick)

    if vert_lines is not None:
        for x in vert_lines:
            plt.axvline(x=x, color="black", linestyle="dashed")
    plt.show()


def get_conf_data(result_list):
    return [row["conf"] for row in result_list]


def show_overall_hist(results):
    acc = round(rload.mean_metric(results[:-1], "accuracy"), 3)
    prec = round(rload.mean_metric(results[:-1], "precision"), 3)
    hit_miss = [get_conf_data(data) for data in results[-1].hits_misses()]

    colors = ["lightgreen", "red"]
    plt.hist(hit_miss[0], bins=10, color=colors[0], range=(0.0, 1.0))
    plt.show()
    title = "Misses"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.hist(hit_miss[1], bins=20, color=colors[1], range=(0.0, 1.0))
    plt.show()
    title = (
        f"Confidence Distribution on Sample Batch\n(acc={acc}, "
        + f"prec={prec}, n={results[-1].pop})"
    )
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.hist(hit_miss, bins=10, color=colors, range=(0.0, 1.0), stacked=True)
    plt.show()


def make_conf_matrix(conf_mat, classes, filename):
    classes.append("")
    df = pd.DataFrame(conf_mat, index=classes, columns=classes)
    df.to_csv(filename)


def aggregate_results(config, prefix, metric, delta=2, avg=False, roll=None):
    names = [
        "init",
        "sample",
        "all_iter",
        "all",
    ]
    epoch_splits = utils.get_epoch_splits(config, prefix, True)
    names += [f"cur_iter{i}" for i in range(len(epoch_splits))]

    results = pd.DataFrame(
        columns=["test_set", "epoch", "prec", "acc", "conf", "conf_std", "recall"]
    )

    out_folder = f"{config['output']}/{prefix}-series"
    if avg or roll:
        out_folder += "-roll-avg" if roll else "-avg"
    for name in names:
        for i in range(0, epoch_splits[-1], delta):
            out_name = f"{out_folder}/{name}_{i}.csv"
            if not os.path.exists(out_name):
                continue

            epoch_res, _ = rload.load_data(
                out_name, by_actual=False, conf_thresh=config["pos_thres"]
            )
            new_row = {
                "test_set": name,
                "epoch": i,
                "prec": rload.mean_metric(epoch_res, "precision"),
                "acc": rload.mean_metric(epoch_res, "accuracy"),
                "conf": rload.mean_avg_conf(epoch_res),
                "conf_std": rload.mean_avg_conf_std(epoch_res),
                "recall": rload.mean_metric(epoch_res, "recall"),
            }
            results = results.append(new_row, ignore_index=True)

    results.to_csv(f"{out_folder}/{prefix}-series-stats.csv")

    xy_pairs = list()
    for name in names:
        if "cur_iter" in name:
            if name == "cur_iter0":
                filtered_data = results[results["test_set"].str.contains("cur_iter")]
                name = "cur_iter"
            else:
                continue
        else:
            filtered_data = results[results["test_set"] == name]
        xy_pairs.append((filtered_data["epoch"], filtered_data[metric], name))

    plot_multiline(
        xy_pairs, xlab="Epoch", ylab=f"Avg. {metric}", vert_lines=epoch_splits
    )


def visualize_conf(prefix, benchmark, sample_filter=False, pos_thres=0.5):
    kwargs = dict()
    if sample_filter:
        folder = "/".join(benchmark.split("/")[:-1])
        epoch = utils.get_epoch(benchmark)
        sampled_imgs = glob(f"{folder}/{prefix}*_sample_{epoch}.txt")[0]
        kwargs["filter"] = sampled_imgs

    results, conf_mat = rload.load_data(
        benchmark, by_actual=False, conf_thresh=pos_thres, **kwargs
    )

    conf_mat_file = benchmark[:-4] + "_conf.csv"
    classes = [result.name for result in results]
    classes.remove("All")
    make_conf_matrix(conf_mat, classes, conf_mat_file)

    hist_filename = benchmark[:-4] + "_viz.pdf"
    make_conf_histogram(results, hist_filename)
    show_overall_hist(results)


def tabulate_batch_samples(config, prefix, silent=False, filter_samp=False, roll=False):
    """Analyze accuracy/precision relationships and training duration
    for each batched sample using existing testing data."""
    bench_str = f"{config['output']}/{prefix}*_benchmark"
    bench_str += "_roll*.csv" if roll else "_avg*.csv"

    benchmarks = utils.sort_by_epoch(bench_str)
    checkpoints = utils.sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")
    data = pd.DataFrame(
        columns=["batch", "prec", "acc", "conf", "conf_std", "recall", "epochs_trained"]
    )

    for i, benchmark in enumerate(benchmarks):
        kwargs = dict()
        if filter_samp and prefix != "init":
            sampled_imgs = glob(f"{config['output']}/{prefix}{i}_sample*")
            if len(sampled_imgs) == 0:
                continue
            kwargs["filter"] = sampled_imgs[0]
        results, _ = rload.load_data(
            benchmark,
            by_actual=False,
            add_all=False,
            conf_thresh=config["pos_thres"],
            **kwargs,
        )

        if i == len(benchmarks) - 1:
            train_len = utils.get_epoch(checkpoints[-1]) - utils.get_epoch(benchmark)
        else:
            train_len = utils.get_epoch(benchmarks[i + 1]) - utils.get_epoch(benchmark)

        data.loc[i] = [
            i,
            rload.mean_metric(results, "precision"),
            rload.mean_metric(results, "accuracy"),
            rload.mean_avg_conf(results),
            rload.mean_avg_conf_std(results),
            rload.mean_metric(results, "recall"),
            train_len,
        ]

    if not silent:
        print("=== Metrics on Batch ===")
        print(data)

    return data


def compare_benchmarks(
    config,
    prefixes,
    metric,
    metric2=None,
    roll=False,
    compare_init=False,
    filter_sample=False,
):
    """Compares benchmarks on sample sets (before retraining) for sample methods."""
    df = pd.DataFrame()
    print("Avg.", metric)
    for prefix in prefixes:
        results = tabulate_batch_samples(
            config, prefix, silent=True, filter_samp=filter_sample, roll=roll
        )[metric]
        if prefix != "init" and compare_init:
            df[prefix] = results - df["init"]
        else:
            df[prefix] = results
    print(df.transpose())

    if metric2 is not None:
        if "init" in prefixes:
            prefixes.remove("init")
        df = pd.DataFrame(
            columns=["Method", f"avg. {metric}", f"batch avg. {metric2}"]
        ).set_index("Method")

        init_vals = tabulate_batch_samples(
            config, "init", silent=True, filter_samp=False
        )
        print(f"Baseline avg {metric2}: ", init_vals[metric2][1:].mean())

        for prefix in prefixes:
            indep_var = tabulate_batch_samples(
                config, prefix, silent=True, filter_samp=True
            )
            dep_var = tabulate_batch_samples(
                config, prefix, silent=True, filter_samp=False
            )

            y_series = dep_var[metric2]
            if compare_init:
                y_series -= init_vals[metric2]

            df.loc[prefix] = [
                indep_var[metric][:-1].mean(),
                y_series[1:].mean(),
            ]

        print(df.sort_values(df.columns[0]))
        linear_regression(df)
