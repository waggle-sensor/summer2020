import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import os
import argparse
from scipy.interpolate import make_interp_spline, BSpline, interp1d
import cutecharts.charts
import webview

"""
This script was primarily written to convert Tensorboard log data from the YOLOv3
scripts to a parsable format, but it can be adapted to other purposes
"""

BATCHES_PER_EPOCH = 1  # 2269


def add_plot(
    data, title, color="b", label="", header=None, spline=None, x_scale=1, cute=False
):
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

    if cute:
        chart = cutecharts.charts.Line(title)
        chart.set_options(
            labels=[round(float(xs), 3) for xs in x],
            x_label=header[0],
            y_label=header[1],
            y_tick_count=5,
        )
        chart.add_series("Retrained Precision (%)", [float(ys * 100) for ys in y])
        return x, y, chart

    plt.plot(x, y, color, label=label)

    plt.title(title)
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    return x, y


def plot_diff(data, data2, title, spline=None, x_scale=1, cute=False):

    x, y1 = add_plot(data, title, spline=spline, x_scale=x_scale)
    _, y2 = add_plot(data2, title, spline=spline, x_scale=x_scale, color="r")

    y3 = [y1[i] - y2[i] for i in range(len(x))]

    if cute:
        _, _, chart = add_plot(data, title, spline=spline, x_scale=x_scale, cute=True)
        chart.add_series("Baseline Precision (%)", [float(ys * 100) for ys in y2])
        chart.add_series("Difference", [float(ys * 100) for ys in y3])
        show_cute_chart(chart)
        return

    plt.axhline(y=0.0, color="black", linestyle="dashed")
    plt.plot(x, y3, "g", label="Difference")

    plt.legend()
    plt.show()


def show_cute_chart(chart):
    chart.render(dest="output/chart.html")
    webview.create_window("title", f"{os.getcwd()}/output/chart.html")
    webview.start()
    os.remove("output/chart.html")


def plot(data, title, spline=None, x_scale=1, xlab="", ylab="", cute=False):
    _, _, chart = add_plot(
        data, title, header=[xlab, ylab], spline=spline, x_scale=x_scale, cute=cute
    )
    if cute:
        show_cute_chart(chart)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--title", default="Title")
    parser.add_argument("--xlab", default="xlab")
    parser.add_argument("--ylab", default="ylab")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--xscale", default=1.0, type=float)
    parser.add_argument("--spline", default=None, type=int)
    parser.add_argument("--cute", default=False, action="store_true")
    args = parser.parse_args()

    x_scale = 1 / args.xscale

    if args.compare is None:
        plot(
            args.file,
            args.title,
            args.spline,
            x_scale=x_scale,
            xlab=args.xlab,
            ylab=args.ylab,
            cute=args.cute,
        )
    else:
        plot_diff(args.file, args.compare, args.title, args.spline, x_scale, args.cute)
