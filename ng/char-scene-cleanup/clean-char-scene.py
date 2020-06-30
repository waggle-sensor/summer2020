#!/usr/bin/env python3
import cleanup

DATA = "./data/"
OUTPUT = "./output/"


def annot_parser(txt_path):
    labels = list()

    with open(txt_path, "r") as file:
        lines = file.read().split("\n")

    for line in lines:
        if line == str():
            continue
        cols = line.split(";")[:-1]

        points = cols[0].split(",")[:-1]

        max_x = float("-inf")
        max_y = float("-inf")
        min_x = float("inf")
        min_y = float("inf")

        for point in points:
            (x, y) = map(int, point.split(":"))

            max_x = max(x, max_x)
            min_x = min(x, min_x)
            max_y = max(y, max_y)
            min_y = min(y, min_y)

        label = dict()
        label["class"] = cols[1]
        label["minXY"] = (min_x, min_y)
        label["maxXY"] = (max_x, max_y)

        labels.append(label)

    return labels


def annot_path_to_img(annot_path):
    return annot_path[:-4]


if __name__ == "__main__":
    exts = [".JPG", ".gif", ".jpg", ".bmp"]
    cleanup.clean(DATA, OUTPUT, exts, ".txt", annot_parser, annot_path_to_img)
