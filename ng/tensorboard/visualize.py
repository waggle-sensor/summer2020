import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from scipy.interpolate import make_interp_spline, BSpline, interp1d

"""
This script was primarily written to convert Tensorboard log data from the YOLOv3
scripts to a parsable format, but it can be adapted to other purposes
"""

BATCHES_PER_EPOCH = 2269


def add_plot(data, title, color="b", label="", header=None, spline=None, x_scale=1):
    x = np.array([], dtype=np.float128)
    y = np.array([], dtype=np.float128)

    with open(data, "r") as file:
        plots = csv.reader(file, delimiter=",")
        csv_header = next(plots, None)
        if header is None:
            header = csv_header

        prev = -1
        for row in plots:
            x_val = np.float128(np.float128(row[0]))
            y_val = np.float128(row[1])

            if x_val == prev or x_val in x or math.isnan(y_val) or math.isinf(y_val):
                continue
            prev = x_val
            x = np.append(x, x_val)
            y = np.append(y, y_val)

    if spline is not None:
        x_new = np.linspace(x.min(), x.max(), spline)
        spl = interp1d(x, y, kind=3)
        y = spl(x_new)
        x = x_new
    x = x_scale * x
    plt.plot(x, y, color, label=label)

    plt.title(title)
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    return x, y


def plot_diff(data, data2, title):
    x, y1 = add_plot(data, title)
    _, y2 = add_plot(data2, title, color="r")

    y3 = [y1[i] - y2[i] for i in range(len(x))]
    plt.axhline(y=1.0, color="black", linestyle="dashed")
    plt.plot(x, y3, "g", label="Difference")

    plt.legend()
    plt.show()


def plot(data, title, spline=None, x_scale=1, ylab=""):
    add_plot(data, title, header=["Epoch", ylab], spline=spline, x_scale=x_scale)
    plt.show()


if __name__ == "__main__":
    title = sys.argv[2] if len(sys.argv) >= 3 else "Title"
    x_scale = 1 if "val_" in sys.argv[1] else 1 / BATCHES_PER_EPOCH
    ylab = sys.argv[1].split(".csv")[0]
    plot(sys.argv[1], title, 50, x_scale=x_scale, ylab=ylab)
