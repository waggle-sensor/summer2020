import benchmark
import csv
import os
import utils
from shutil import copyfile

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
    test = "_test"

    output = open(f"output/val_precision{test}_time.csv", "w+")
    output.write("epoch,all_precision\n")
    for i in range(99, 150, 3):
        if not os.path.exists(f"output/benchmark{test}_{i}.csv"):
            benchmark.benchmark(
                "median_thresh",
                i,
                "config/yolov3.cfg",
                "config/chars.data",
                "config/chars.names",
                "data/images/objs/" if test == str() else "data/temp/",
                f"benchmark{test}",
            )

        results, _ = utils.load_data(f"output/benchmark{test}_{i}.csv", True)
        output.write(f"{i},{results[-1].precision()}\n")

    output.close()
