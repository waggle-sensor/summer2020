import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from retrain.dataloader import LabeledSet
import retrain.utils as utils
import retrain.benchmark as bench


def get_epoch_splits(config, prefix, incl_last_epoch=False, avg=False):
    splits = list(
        map(get_epoch, sort_by_epoch(f"{config['output']}/{prefix}*sample*.txt"))
    )
    if incl_last_epoch:
        last_checkpoint = sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")[-1]
        splits.append(get_epoch(last_checkpoint))
    return splits


def series_benchmark(config, prefix, delta=2, avg=False):
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
    if avg:
        out_folder += "-avg"
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
                    if avg:
                        result_file = bench.benchmark_avg(
                            img_folder, prefix, 1, epoch, 5, config
                        )
                    else:
                        result_file = bench.benchmark(img_folder, prefix, epoch, config)
                    os.rename(result_file, out_name)
        if i != 0:
            test_sets.pop(f"cur_iter{i}")


def aggregate_results(config, prefix, delta=2, avg=False):
    names = ["init", "sample", "all_iter", "all"]
    epoch_splits = get_epoch_splits(config, prefix, True)
    #names += [f"cur_iter{i}" for i in range(len(epoch_splits))]

    results = pd.DataFrame(columns=["test_set", "epoch", "prec", "acc", "mean_conf"])

    out_folder = f"{config['output']}/{prefix}-series"
    if avg:
        out_folder += "-avg"
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
                "mean_conf": bench.mean_conf(epoch_res),
            }
            results = results.append(new_row, ignore_index=True)

    results.to_csv(f"{out_folder}/{prefix}-series-stats.csv")

    plt.xlabel("Epoch")
    plt.ylabel("Avg. Precision")
    for name in names:
        filtered_data = results[results["test_set"] == name]
        plt.plot(filtered_data["epoch"], filtered_data["prec"], label=name)
    plt.legend()
    for split in epoch_splits:
        plt.axvline(x=split, color="black", linestyle="dashed")
    plt.show()


def get_epoch(filename):
    return int(filename.split("_")[-1].split(".")[0])


def sort_by_epoch(pattern):
    files = glob.glob(pattern)
    return sorted(files, key=get_epoch)


def tabulate_batch_samples(config, prefix, silent=False, filter=False):
    """Analyze accuracy/precision relationships and training duration
    for each batched sample using existing testing data."""

    benchmarks = sort_by_epoch(f"{config['output']}/{prefix}_bench*.csv")
    checkpoints = sort_by_epoch(f"{config['checkpoints']}/{prefix}*.pth")

    data = pd.DataFrame(
        columns=["Batch", "Prec", "Acc", "Conf", "Recall", "Epochs Trained"]
    )

    for i, benchmark in enumerate(benchmarks):
        if filter:
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
        print(data)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--start", required=True, type=int, help="starting benchmark epoch",
    # )
    # parser.add_argument(
    #     "--end", required=True, type=int, help="ending benchmark epoch",
    # )
    # parser.add_argument(
    #     "--delta", type=int, help="interval to plot", default=3, required=False
    # )
    parser.add_argument("--prefix", default="init", help="prefix of model to test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--in_list", default=None)
    parser.add_argument("--avg", action="store_true", default=False)
    parser.add_argument("--tabulate", action="store_true", default=False)
    opt = parser.parse_args()

    config = utils.parse_retrain_config(opt.config)

    if opt.tabulate:
        prefixes = ["median-below-thresh", "median-thresh", "normal", "iqr"]
        df = pd.DataFrame()
        for prefix in prefixes:
            results = tabulate_batch_samples(config, prefix, silent=True, filter=False)[
                "Recall"
            ]
            df[prefix] = results
        print(df)

    else:
        tabulate_batch_samples(config, opt.prefix)
        series_benchmark(config, opt.prefix, avg=opt.avg)
        aggregate_results(config, opt.prefix, avg=opt.avg)

    # images = utils.get_lines(opt.in_list)
    # img_folder = ImageFolder(images, config["img_size"], prefix=opt.prefix)

    # bench.series_benchmark_loss(
    #     img_folder, opt.prefix, opt.start, opt.end, opt.delta, config, opt.out
    # )
