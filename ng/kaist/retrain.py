import sample
import benchmark
import utils
import sys
import os
import subprocess as sp
import random
import argparse


"""
This is a really hacky script right now, meant for testing.
"""

if __name__ == "__main__":
    random.seed("sage")

    methods = {
        "median_thresh": sample.median_thresh_sample,
        "normal": sample.normal_sample,
        "iqr": sample.iqr_sample,
    }
    data_file = "config/chars.data"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        help="benchmark epoch for retraining",
        default=None,
        required=True,
    )
    parser.add_argument("--from_file", type=str, help="load from file", default=None)
    opt = parser.parse_args()

    if opt.from_file is not None:
        results, _ = utils.load_data(opt.from_file, by_actual=False)

    else:
        if not os.path.exists(f"output/benchmark_{opt.epoch}.csv"):
            benchmark.benchmark(
                "yolov3",
                opt.epoch,
                "config/yolov3.cfg",
                data_file,
                "config/chars.names",
                "data/images/objs/",
            )

        results, _ = utils.load_data(
            f"output/benchmark_{opt.epoch}.csv", by_actual=False
        )

    for name, func in methods.items():
        sample.create_sample(data_file, results, False, name, func)

        train_list = f"output/configs-retrain/{name}/train.txt"

        aug_cmd = f"python3 ../char-cleanup/augment.py --train_list {train_list} --balance --compose --imgs_per_class 10000"
        aug_cmd = aug_cmd.split(" ")
        sp.run(aug_cmd, check=True)

        data_config = f"output/configs-retrain/{name}/chars.data"
        bench_weights = f"checkpoints/yolov3_ckpt_{opt.epoch}.pth"

        epoch_num = opt.epoch + 26
        train_cmd = (
            f"python3 ../yolov3/train.py --epoch {epoch_num} --data_config {data_config} "
            + f"--pretrained_weights {bench_weights} --img_size 416 --resume {opt.epoch} "
            + f"--prefix {name} --clip 1.0 --batch_size 16"
        )
        train_cmd = train_cmd.split(" ")
        sp.run(train_cmd, check=True)
