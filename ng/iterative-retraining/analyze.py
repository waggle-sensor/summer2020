import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import os
from tqdm import tqdm
from retrain.dataloader import LabeledSet
import retrain.utils as utils
import retrain.benchmark as bench


def get_epoch_splits(config, prefix, incl_last_epoch=False):
    splits = [
        get_epoch(file)
        for file in sort_by_epoch(f"{config['output']}/{prefix}*sample*.txt")
    ]
    if incl_last_epoch:
        last_checkpoint = sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")[-1]
        splits.append(get_epoch(last_checkpoint))
    return splits


def series_benchmark(config, prefix, delta=2, avg=False, roll=None):
    # 1. Find the number of batches for the given prefix
    # 2. Find the starting/ending epochs of each split
    # 3. Benchmark that itertion's test set with the average method
    #    (Could plot this, but may not be meaningful due to differing test sets)
    # 4. Benchmark the overall test set with the same average method (and save results)
    #    4a. plot the overall test set performance as a function of epoch number
    # 5. (optional) serialize results of the overall test set as JSON for improved speed
    #    when using averages

    out_dir = config["output"]
    num_classes = len(utils.load_classes(config["class_list"]))
    epoch_splits = get_epoch_splits(config, prefix)

    # Initial test set
    init_test_set = f"{out_dir}/init_test.txt"
    init_test_folder = LabeledSet(init_test_set, num_classes)

    # Only data from the (combined) iteration test sets (75% sampling + 25% seen data)
    iter_test_sets = [
        f"{out_dir}/{prefix}{i}_test.txt" for i in range(len(epoch_splits))
    ]
    iter_img_files = list()
    for file in iter_test_sets:
        iter_img_files += utils.get_lines(file)
    all_iter_sets = LabeledSet(iter_img_files, num_classes)

    # Test sets filtered for only sampled images
    sampled_imgs = [img for img in iter_img_files if config["sample_set"] in img]
    sample_test = LabeledSet(sampled_imgs, num_classes)

    # Data from all test sets
    all_test = LabeledSet(sampled_imgs, num_classes)
    all_test += init_test_folder

    test_sets = {
        "init": init_test_folder,
        "all_iter": all_iter_sets,
        "sample": sample_test,
        "all": all_test,
    }

    epoch_splits = get_epoch_splits(config, prefix, True)

    # Begin benchmarking
    out_folder = f"{out_dir}/{prefix}-series"
    if avg or roll:
        out_folder += "-roll-avg" if roll else "-avg"
    os.makedirs(out_folder, exist_ok=True)
    for i, split in enumerate(epoch_splits):
        # Get specific iteration set
        if i != 0:
            test_sets[f"cur_iter{i}"] = LabeledSet(iter_test_sets[i - 1], num_classes)

        start = epoch_splits[i - 1] if i else 0

        for epoch in tqdm(range(start, split + 1, delta)):
            for name, img_folder in test_sets.items():
                # Benchmark both iterations sets at the split mark
                if not epoch or (epoch == start and "cur_iter" not in name):
                    continue

                out_name = f"{out_folder}/{name}_{epoch}.csv"

                if not os.path.exists(out_name):
                    if roll:
                        result_df = bench.benchmark_avg(
                            img_folder, prefix, 1, epoch, roll, config, roll=True
                        )
                    elif avg:
                        result_df = bench.benchmark_avg(
                            img_folder, prefix, 1, epoch, 5, config
                        )
                    else:
                        result_df = bench.benchmark(img_folder, prefix, epoch, config)
                    bench.save_results(result_df, out_name)
        if i != 0:
            test_sets.pop(f"cur_iter{i}")


def aggregate_results(config, prefix, metric, delta=2, avg=False, roll=None):
    names = [
        "init",
        "sample",
        "all_iter",
        "all",
    ]
    epoch_splits = get_epoch_splits(config, prefix, True)
    names += [f"cur_iter{i}" for i in range(len(epoch_splits))]

    results = pd.DataFrame(
        columns=["test_set", "epoch", "prec", "acc", "conf", "recall"]
    )

    out_folder = f"{config['output']}/{prefix}-series"
    if avg or roll:
        out_folder += "-roll-avg" if roll else "-avg"
    for name in names:
        for i in range(0, epoch_splits[-1], delta):
            out_name = f"{out_folder}/{name}_{i}.csv"
            if not os.path.exists(out_name):
                continue

            epoch_res, _ = bench.load_data(out_name, by_actual=True)
            new_row = {
                "test_set": name,
                "epoch": i,
                "prec": bench.mean_precision(epoch_res),
                "acc": bench.mean_accuracy(epoch_res),
                "conf": bench.mean_conf(epoch_res),
                "recall": bench.mean_recall(epoch_res),
            }
            results = results.append(new_row, ignore_index=True)

    results.to_csv(f"{out_folder}/{prefix}-series-stats.csv")

    plt.xlabel("Epoch")
    plt.ylabel(f"Avg. {metric}")
    for name in names:
        filtered_data = results[results["test_set"] == name]
        plt.plot(filtered_data["epoch"], filtered_data[metric], label=name)
    plt.legend()
    for split in epoch_splits:
        plt.axvline(x=split, color="black", linestyle="dashed")
    plt.show()


def get_epoch(filename):
    return int(filename.split("_")[-1].split(".")[0])


def sort_by_epoch(pattern):
    files = sorted(glob.glob(pattern))
    return sorted(files, key=get_epoch)


def visualize_conf(prefix, benchmark, sample_filter=False):
    if sample_filter:
        folder = "/".join(benchmark.split("/")[:-1])
        epoch = get_epoch(benchmark)
        sampled_imgs = glob.glob(f"{folder}/{prefix}*_sample_{epoch}.txt")[0]
        results, _ = bench.load_data(benchmark, by_actual=True, filter=sampled_imgs)
    else:
        results, _ = bench.load_data(benchmark, by_actual=True)

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
            acc = round(bench.mean_accuracy(results[:-1]), 3)
            prec = round(bench.mean_precision(results[:-1]), 3)
        else:
            acc = round(res.accuracy(), 3)
            prec = round(res.precision(), 3)

        title = f"Class: {res.name} (acc={acc}, " + f"prec={prec}, n={res.pop})"
        axs[i][1].set_title(title)

    fig.set_figheight(2.5 * num_rows)
    fig.set_figwidth(10)
    fig.savefig(benchmark[:-4] + "_viz.pdf", bbox_inches="tight")


def tabulate_batch_samples(config, prefix, silent=False, filter=False, roll=False):
    """Analyze accuracy/precision relationships and training duration
    for each batched sample using existing testing data."""
    bench_str = f"{config['output']}/{prefix}*_benchmark"
    bench_str += "_roll*.csv" if roll else "_avg*.csv"

    benchmarks = sort_by_epoch(bench_str)
    checkpoints = sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")

    data = pd.DataFrame(
        columns=["batch", "prec", "acc", "conf", "recall", "epochs trained"]
    )

    for i, benchmark in enumerate(benchmarks):
        if filter and prefix != "init":
            sampled_imgs = glob.glob(f"{config['output']}/{prefix}{i}_sample*")[0]
            results, _ = bench.load_data(
                benchmark, by_actual=True, add_all=False, filter=sampled_imgs
            )

        else:
            results, _ = bench.load_data(benchmark, by_actual=True, add_all=False)

        if i == len(benchmarks) - 1:
            train_len = get_epoch(checkpoints[-1]) - get_epoch(benchmark)
        else:
            train_len = get_epoch(benchmarks[i + 1]) - get_epoch(benchmark)

        data.loc[i] = [
            i,
            bench.mean_precision(results),
            bench.mean_accuracy(results),
            bench.mean_conf(results),
            bench.mean_recall(results),
            train_len,
        ]

    if not silent:
        print("=== Metrics on Batch ===")
        print(data)

    return data


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


def compare_benchmarks(prefixes, metric, metric2=None, roll=False):
    """Compares benchmarks on sample sets (before retraining) for sample methods."""
    df = pd.DataFrame()
    for prefix in prefixes:
        results = tabulate_batch_samples(
            config, prefix, silent=True, filter=opt.filter_sample, roll=roll
        )[metric]
        df[prefix] = results
    print(df.transpose())

    if metric2:
        if "init" in prefixes:
            prefixes.remove("init")
        df = pd.DataFrame(
            columns=["Method", f"avg. {opt.metric}", f"avg. {opt.metric2}",]
        ).set_index("Method")

        for prefix in prefixes:
            indep_var = tabulate_batch_samples(config, prefix, silent=True, filter=True)
            dep_var = tabulate_batch_samples(config, prefix, silent=True, filter=False)

            df.loc[prefix] = [
                indep_var[metric][:-1].mean(),
                dep_var[metric2][1:].mean(),
            ]

        print(df)
        linear_regression(df)


def benchmark_batch_set(prefix, config, roll=None):
    """See initial training performance on batch splits."""
    out_dir = config["output"]
    num_classes = len(utils.load_classes(config["class_list"]))
    batch_sets = sorted(glob.glob(f"{out_dir}/sample*.txt"))

    epoch_splits = get_epoch_splits(config, prefix, True)
    if prefix == "init":
        epoch_splits *= len(batch_sets)

    for i, batch_set in enumerate(batch_sets):
        batch_folder = LabeledSet(batch_set, num_classes)
        if len(batch_folder) < config["sampling_batch"]:
            break

        end_epoch = epoch_splits[i]
        num_ckpts = roll if roll is not None else config["conf_check_num"]
        filename = f"{out_dir}/{prefix}{i}_benchmark_"
        filename += "roll_" if roll else "avg_"
        filename += f"{end_epoch}.csv"

        if os.path.exists(filename):
            continue
        if roll is not None:
            results = bench.simple_benchmark_avg(
                batch_folder, prefix, 1, end_epoch, num_ckpts, config, roll=True
            )
        else:
            results = bench.simple_benchmark_avg(
                batch_folder, prefix, 1, end_epoch, num_ckpts, config,
            )

        bench.save_results(results, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="init", help="prefix of model to test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--in_list", default=None)
    parser.add_argument("--avg", action="store_true", default=False)
    parser.add_argument("--roll_avg", type=int, default=None)
    parser.add_argument("--tabulate", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--visualize_conf", default=None)
    parser.add_argument("--filter_sample", action="store_true", default=False)
    parser.add_argument("--metric", default="prec")
    parser.add_argument("--metric2", default=None)
    opt = parser.parse_args()

    config = utils.parse_retrain_config(opt.config)

    prefixes = [
        "median-below-thresh",
        "median-thresh",
        "normal",
        "iqr",
        "mid-thresh",
        "mid-below-thresh",
        "mid-normal",
        "bin-quintile",
        "bin-normal",
        "init",
    ]

    if opt.benchmark:
        for prefix in prefixes:
            benchmark_batch_set(prefix, config, opt.roll_avg)
    if opt.tabulate:
        if opt.prefix != "init":
            tabulate_batch_samples(
                config, opt.prefix, filter=opt.filter_sample, roll=opt.roll_avg
            )
        else:
            compare_benchmarks(prefixes, opt.metric, opt.metric2, roll=opt.roll_avg)

    elif opt.visualize_conf:
        visualize_conf(opt.prefix, opt.visualize_conf, opt.filter_sample)

    elif opt.prefix != "init":
        tabulate_batch_samples(config, opt.prefix, roll=opt.roll)
        series_benchmark(config, opt.prefix, avg=opt.avg, roll=opt.roll_avg)
        aggregate_results(
            config, opt.prefix, opt.metric, avg=opt.avg, roll=opt.roll_avg
        )
