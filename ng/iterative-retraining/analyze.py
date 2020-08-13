import argparse
import glob
import pandas as pd

import os
from tqdm import tqdm
from retrain.dataloader import LabeledSet
import retrain.utils as utils

import analysis.benchmark as bench
from analysis import charts


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
    epoch_splits = utils.get_epoch_splits(config, prefix)

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

    epoch_splits = utils.get_epoch_splits(config, prefix, True)

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
    epoch_splits = utils.get_epoch_splits(config, prefix, True)
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

            epoch_res, _ = bench.load_data(
                out_name, by_actual=True, conf_thresh=config["pos_thres"]
            )
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

    charts.plot_multiline(
        xy_pairs, xlab="Epoch", ylab=f"Avg. {metric}", vert_lines=epoch_splits
    )


def visualize_conf(prefix, benchmark, sample_filter=False):
    if sample_filter:
        folder = "/".join(benchmark.split("/")[:-1])
        epoch = utils.get_epoch(benchmark)
        sampled_imgs = glob.glob(f"{folder}/{prefix}*_sample_{epoch}.txt")[0]
        results, _ = bench.load_data(benchmark, by_actual=True, filter=sampled_imgs)
    else:
        results, _ = bench.load_data(benchmark, by_actual=True)

    filename = benchmark[:-4] + "_viz.pdf"
    charts.make_conf_histogram(results, filename)


def tabulate_batch_samples(config, prefix, silent=False, filter=False, roll=False):
    """Analyze accuracy/precision relationships and training duration
    for each batched sample using existing testing data."""
    bench_str = f"{config['output']}/{prefix}*_benchmark"
    bench_str += "_roll*.csv" if roll else "_avg*.csv"

    benchmarks = utils.sort_by_epoch(bench_str)
    checkpoints = utils.sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")

    data = pd.DataFrame(
        columns=["batch", "prec", "acc", "conf", "recall", "epochs trained"]
    )

    for i, benchmark in enumerate(benchmarks):
        if filter and prefix != "init":
            sampled_imgs = glob.glob(f"{config['output']}/{prefix}{i}_sample*")[0]
            results, _ = bench.load_data(
                benchmark,
                by_actual=True,
                add_all=False,
                filter=sampled_imgs,
                conf_thresh=config["pos_thres"],
            )

        else:
            results, _ = bench.load_data(
                benchmark,
                by_actual=True,
                add_all=False,
                conf_thresh=config["pos_thres"],
            )

        if i == len(benchmarks) - 1:
            train_len = utils.get_epoch(checkpoints[-1]) - utils.get_epoch(benchmark)
        else:
            train_len = utils.get_epoch(benchmarks[i + 1]) - utils.get_epoch(benchmark)

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
            columns=["Method", f"avg. {opt.metric}", f"avg. {opt.metric2}"]
        ).set_index("Method")

        for prefix in prefixes:
            indep_var = tabulate_batch_samples(config, prefix, silent=True, filter=True)
            dep_var = tabulate_batch_samples(config, prefix, silent=True, filter=False)

            df.loc[prefix] = [
                indep_var[metric][:-1].mean(),
                dep_var[metric2][1:].mean(),
            ]

        print(df)
        charts.linear_regression(df)


def benchmark_batch_set(prefix, config, roll=None):
    """See initial training performance on batch splits."""
    out_dir = config["output"]
    num_classes = len(utils.load_classes(config["class_list"]))
    batch_sets = sorted(glob.glob(f"{out_dir}/sample*.txt"))

    epoch_splits = utils.get_epoch_splits(config, prefix, True)
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
        filename += f"1_{end_epoch}.csv"

        if os.path.exists(filename):
            continue
        if roll is not None:
            results = bench.benchmark_avg(
                batch_folder, prefix, 1, end_epoch, num_ckpts, config, roll=True
            )
        else:
            results = bench.benchmark_avg(
                batch_folder, prefix, 1, end_epoch, num_ckpts, config,
            )

        bench.save_results(results, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=None, help="prefix of model to test")
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
        "random",
        "init",
    ]

    if opt.benchmark and opt.prefix is None:
        for prefix in prefixes:
            benchmark_batch_set(prefix, config, opt.roll_avg)
            series_benchmark(config, prefix, avg=opt.avg, roll=opt.roll_avg)
    if opt.tabulate:
        if opt.prefix != "init":
            tabulate_batch_samples(
                config, opt.prefix, filter=opt.filter_sample, roll=opt.roll_avg
            )
        else:
            compare_benchmarks(prefixes, opt.metric, opt.metric2, roll=opt.roll_avg)

    elif opt.visualize_conf:
        visualize_conf(opt.prefix, opt.visualize_conf, opt.filter_sample)

    elif opt.prefix is not None:
        benchmark_batch_set(opt.prefix, config, opt.roll_avg)
        tabulate_batch_samples(config, opt.prefix, roll=opt.roll_avg)
        if opt.prefix != "init":
            series_benchmark(config, opt.prefix, avg=opt.avg, roll=opt.roll_avg)
            aggregate_results(
                config, opt.prefix, opt.metric, avg=opt.avg, roll=opt.roll_avg
            )
