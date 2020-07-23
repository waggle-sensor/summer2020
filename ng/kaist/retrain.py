import sample
import benchmark
import utils
import sys
import os
import subprocess as sp
import random
import argparse


def get_retrain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_epoch",
        type=int,
        help="benchmark epoch for retraining",
        default=None,
        required=True,
    )
    parser.add_argument("--from_file", type=str, help="load from file", default=None)
    parser.add_argument(
        "--early_stop",
        type=bool,
        default=False,
        action="store_true",
        help="use early stop instead of a set number of epochs",
    )
    parser.add_argument(
        "--sample_iterations",
        type=int,
        default=1,
        help="iterations of sampling/retraining to undergo",
    )
    parser.add_argument(
        "--retrain_length",
        type=int,
        default=25,
        help="(max) number of epochs to retrain per sample iteration",
    )
    parser.add_argument("--data_config", default="config/chars.data", type=str)
    parser.add_argument("--prefix", default="yolov3", type=str)
    opt = parser.parse_args()

    return opt


def get_benchmark_results(opt):
    if opt.from_file is not None:
        results, _ = utils.load_data(opt.from_file, by_actual=False)
    else:
        if not os.path.exists(f"output/benchmark_{opt.start_epoch}.csv"):
            benchmark.benchmark(
                "yolov3",
                opt.epoch,
                "config/yolov3.cfg",
                opt.data_config,
                "config/chars.names",
                "data/images/objs/",
            )

        results, _ = utils.load_data(
            f"output/benchmark_{opt.start_epoch}.csv", by_actual=False
        )

    return results


if __name__ == "__main__":
    random.seed("sage")

    opt = get_retrain_args()

    methods = {
        "median_thresh": sample.median_thresh_sample,
        "normal": sample.normal_sample,
        "iqr": sample.iqr_sample,
    }

    results = get_benchmark_results(opt)

    for name, func in methods.items():
        start_epoch = opt.start_epoch
        epoch_num = start_epoch + opt.retrain_length + 1
        prefix = opt.prefix

        for i in range(opt.sample_iterations):
            sample.create_sample(data_file, results, False, name, func)

            train_list = f"output/configs-retrain/{name}/train.txt"

            aug_cmd = (
                f"python3 ../char-cleanup/augment.py --train_list {train_list} --balance --compose "
                + "--imgs_per_class 10000"
            )
            aug_cmd = aug_cmd.split(" ")
            sp.run(aug_cmd, check=True)

            data_config = f"output/configs-retrain/{name}/chars.data"
            bench_weights = f"checkpoints/{prefix}_ckpt_{opt.start_epoch}.pth"

            train_cmd = (
                f"python3 ../yolov3/train.py --epochs {epoch_num} --data_config {opt.data_config} "
                + f"--pretrained_weights {bench_weights} --img_size 416 --resume {start_epoch} "
                + f"--prefix {name} --clip 1.0 --batch_size 16 "
            )

            if opt.early_stop:
                train_cmd += "--early_stop"

            train_cmd = train_cmd.split(" ")
            sp.run(train_cmd, check=True)

            start_epoch = epoch_num - 1
            prefix = name
