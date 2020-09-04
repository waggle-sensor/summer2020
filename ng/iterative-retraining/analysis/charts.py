import os
from glob import glob

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
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
    plt.ylabel(f"batch test set {df.columns[1]} incr.")
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
        f"Confidence Distribution on Binned Normal PDF\n(acc={acc}, "
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


def get_avg_metric_dict(results):
    return {
        "prec": rload.mean_metric(results, "precision"),
        "acc": rload.mean_metric(results, "accuracy"),
        "conf": rload.mean_avg_conf(results),
        "conf_std": rload.mean_conf_std(results),
        "detect_conf_std": rload.mean_avg_detect_conf_std(results),
        "recall": rload.mean_metric(results, "recall"),
    }


def display_series(config, opt):
    names = [
        "init",
        "sample",
        "all_iter",
        "all",
    ]
    epoch_splits = utils.get_epoch_splits(config, opt.prefix, True)
    names += [f"cur_iter{i}" for i in range(len(epoch_splits))]
    if opt.batch_test is not None:
        names.append("batch_test")

    results = list()

    out_folder = f"{config['output']}/{opt.prefix}-series"
    if opt.avg or opt.roll_avg:
        out_folder += "-roll-avg" if opt.roll_avg else "-avg"
    for name in names:
        is_baseline = opt.prefix == "init" or "baseline" in opt.prefix
        start_epoch = 1 if is_baseline else epoch_splits[0]
        for i in range(start_epoch, epoch_splits[-1], opt.delta):
            out_name = f"{out_folder}/{name}_{i}.csv"

            if not os.path.exists(out_name):
                print(f"Skipping epoch {i} due to missing benchmark")
                continue

            epoch_res, _ = rload.load_data(
                out_name, by_actual=False, conf_thresh=config["pos_thres"]
            )
            new_row = {"test_set": name, "epoch": i, **get_avg_metric_dict(epoch_res)}
            results.append(new_row)

    results = pd.DataFrame.from_dict(results, orient="columns")

    results.to_csv(f"{out_folder}/{opt.prefix}-series-stats.csv")

    xy_pairs = list()
    for name in names:
        if "cur_iter" in name:
            # Combine the current iteration sets into one line
            if name == "cur_iter0":
                filtered_data = results[results["test_set"].str.contains("cur_iter")]
                name = "cur_iter"
            else:
                continue
        else:
            filtered_data = results[results["test_set"] == name]
        xy_pairs.append((filtered_data["epoch"], filtered_data[opt.metric], name))

    plot_multiline(
        xy_pairs, xlab="Epoch", ylab=f"Avg. {opt.metric}", vert_lines=epoch_splits
    )


def visualize_conf(prefix, benchmark, filter_sample=False, pos_thres=0.5):
    kwargs = dict()
    if filter_sample:
        folder = "/".join(benchmark.split("/")[:-1])
        epoch = utils.get_epoch(benchmark)
        sampled_imgs = glob(f"{folder}/{prefix}*_sample_{epoch}.txt")[0]
        kwargs["filter"] = sampled_imgs

    results, conf_mat = rload.load_data(
        benchmark, by_actual=False, conf_thresh=pos_thres, **kwargs
    )

    results[-1].generate_prec_distrib(benchmark[:-4] + "_prec.csv")

    conf_mat_file = benchmark[:-4] + "_conf.csv"
    classes = [result.name for result in results]
    classes.remove("All")
    make_conf_matrix(conf_mat, classes, conf_mat_file)

    hist_filename = benchmark[:-4] + "_viz.pdf"
    make_conf_histogram(results, hist_filename)
    show_overall_hist(results)


def tabulate_batch_samples(
    config, prefix, bench_suffix=None, silent=False, filter_samp=False
):
    """Analyze accuracy/precision relationships and training duration
    for each batched sample using existing testing data."""
    bench_str = f"{config['output']}/{prefix}*_benchmark" + bench_suffix

    benchmarks = utils.sort_by_epoch(bench_str)
    checkpoints = utils.sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")

    data = list()
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

        new_row = {
            "batch": i,
            **get_avg_metric_dict(results),
            "epochs_trained": train_len,
        }

        data.append(new_row)
    data = pd.DataFrame.from_dict(data, orient="columns")
    # data.set_index("batch")

    if not silent:
        print("=== Metrics on Batch ===")
        print(data.to_string(index=False))

    return data


def compare_benchmarks(
    config,
    prefixes,
    metric,
    metric2=None,
    bench_suffix=None,
    compare_init=False,
    filter_sample=False,
    use_median=False,
):
    """Compares benchmarks on sample sets (before retraining) for sample methods."""
    sample_results = dict()
    print(metric)

    if "init" in prefixes:
        prefixes.remove("init")

    sample_results["init"] = tabulate_batch_samples(
        config, "init", bench_suffix=bench_suffix, silent=True, filter_samp=False
    )

    max_len = 1
    for prefix in prefixes:
        results = tabulate_batch_samples(
            config,
            prefix,
            bench_suffix=bench_suffix,
            silent=True,
            filter_samp=filter_sample,
        )[metric]

        if "test" in bench_suffix:
            max_len = max(max_len, len(results))
        sample_results[prefix] = results

    init_vals = pd.DataFrame(np.repeat(sample_results["init"].values, max_len, axis=0))
    init_vals.columns = sample_results["init"].columns
    sample_results["init"] = init_vals[metric]

    if compare_init and metric2 is None:
        for prefix, result in sample_results.items():
            if prefix == "init":
                continue
            sample_results[prefix] = result - sample_results["init"]

    df = pd.DataFrame.from_dict(sample_results, orient="index")
    print(df)

    if metric2 is not None:
        agg_str = "median" if use_median else "mean"
        df = pd.DataFrame(
            columns=["Method", f"{agg_str} {metric}", f"{agg_str} {metric2}"]
        ).set_index("Method")

        print(f"Baseline avg {metric2}: ", init_vals[metric2][1:].mean())

        for prefix in prefixes:
            indep_var = tabulate_batch_samples(
                config,
                prefix,
                bench_suffix="_avg_1_*.csv",
                silent=True,
                filter_samp=True,
            )
            dep_var = tabulate_batch_samples(
                config,
                prefix,
                bench_suffix=bench_suffix,
                silent=True,
                filter_samp=False,
            )

            y_series = dep_var[metric2]
            if compare_init:
                y_series -= init_vals[metric2]

            df.loc[prefix] = [
                getattr(indep_var[metric][:-1], agg_str)(),
                getattr(y_series[1:], agg_str)(),
            ]

        print(df.sort_values(df.columns[0]))
        linear_regression(df)


def display_benchmark(file, config):
    results, _ = rload.load_data(
        file, by_actual=False, add_all=False, conf_thresh=config["pos_thres"],
    )

    df = pd.DataFrame(
        columns=["Class", "N", "Prec", "Acc", "Recall", "Avg. Conf", "Conf Std"]
    ).set_index("Class")
    for result in results:
        df.loc[result.name] = [
            len(result),
            result.precision(),
            result.accuracy(),
            result.recall(),
            np.mean(result.get_confidences()),
            np.std(result.get_confidences(), ddof=1),
        ]

    df.loc["Overall"] = [df["N"].sum(), *df.loc[:, "Prec":"Conf Std"].mean(axis=0)]
    print(df)
    df.to_csv(file[:-4] + "_stats.csv")
