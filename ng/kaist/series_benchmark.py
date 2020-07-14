import benchmark
import csv
import os
import utils
from shutil import copyfile
import argparse

"""
Helper script to plot average precision as a function of
epoch/checkpoint number
"""


def copy_test(test_file):
    """Small helper function to copy test set into separate folder"""
    with open(test_file, "r") as file:
        lines = file.read().split("\n")[-1]
    for f in lines:
        idx = int(f.split("Sample0")[1].split("/")[0]) - 1
        new_name = f"data/temp/Class-{generate_all_classes()[idx]}/{f.split('/')[-1]}"
        os.makedirs(os.path.dirname(new_name), exist_ok=True)
        if os.path.exists(new_name):
            new_name = new_name[:-4] + "B.png"
        copyfile(f, new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", required=True, type=int, help="starting benchmark epoch",
    )
    parser.add_argument(
        "--end", required=True, type=int, help="ending benchmark epoch",
    )
    parser.add_argument(
        "--delta", type=int, help="interval to plot", default=3, required=False
    )
    parser.add_argument("--prefix", required=True, help="prefix of model to test")
    parser.add_argument("--output", required=False, default="./output/")
    opt = parser.parse_args()

    for test in ("", "_test"):
        output = open(f"{opt.output}/val_precision{test}_time.csv", "w+")
        output.write("epoch,all_precision\n")
        for i in range(opt.start, opt.end, opt.delta):
            if not os.path.exists(f"{opt.output}/benchmark{test}_{i}.csv"):
                benchmark.benchmark(
                    opt.prefix,
                    i,
                    "config/yolov3.cfg",
                    "config/chars.data",
                    "config/chars.names",
                    "data/images/objs/" if test == str() else "data/temp/",
                    f"{opt.output}/benchmark{test}",
                )

            results, _ = utils.load_data(f"{opt.output}/benchmark{test}_{i}.csv", True)
            output.write(f"{i},{benchmark.mean_precision(results[:-1])}\n")

        output.close()
