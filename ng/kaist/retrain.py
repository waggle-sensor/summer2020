import sample
import benchmark
import utils
import sys
import os
import subprocess as sp
import random


"""
This is a really hacky script right now, meant for testing.

TODO generalize more
"""

if __name__ == "__main__":
    random.seed("sage")

    check_num = int(sys.argv[1])

    methods = {"normal": sample.normal_sample, "iqr": sample.iqr_sample}
    # methods = {"median_thresh": sample.median_thresh_sample}
    data_file = "config/chars.data"

    if not os.path.exists(f"output/benchmark_{check_num}.csv"):
        benchmark.benchmark(
            "yolov3",
            check_num,
            "config/yolov3.cfg",
            data_file,
            "config/chars.names",
            "data/images/objs/",
        )

    results, _ = utils.load_data(f"output/benchmark_{check_num}.csv", by_actual=False)

    for name, func in methods.items():
        sample.create_sample(data_file, results, False, name, func)

        train_list = f"output/configs-retrain/{name}/train.txt"

        aug_cmd = f"python3 ../char-cleanup/augment.py --train_list {train_list} --balance --compose --imgs_per_class 10000"
        aug_cmd = aug_cmd.split(" ")
        sp.run(aug_cmd, check=True)

        data_config = f"output/configs-retrain/{name}/chars.data"
        bench_weights = f"checkpoints/yolov3_ckpt_{check_num}.pth"

        epoch_num = check_num + 26
        train_cmd = (
            f"python3 ../yolov3/train.py --epoch {epoch_num} --data_config {data_config} "
            + f"--pretrained_weights {bench_weights} --img_size 416 --resume {check_num} "
            + f"--prefix {name} --clip 1.0 --batch_size 16"
        )
        train_cmd = train_cmd.split(" ")
        sp.run(train_cmd, check=True)
