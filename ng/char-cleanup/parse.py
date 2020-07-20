#!/usr/bin/env python3

"""
Hacky script created to view bounding box annotations for augmented images
"""

import sys
import os
sys.path.insert(0, '../kaist')

import cleanup

DATA = "./data/aug-images"
OUTPUT = "./output/"


def annot_parser(txt):
    str_labels = open(txt, "r").read().rstrip(" \n").split("\n")
    crude_labels = [list(map(float, label.split(" "))) for label in str_labels]
    dict_labels = list()

    for label in crude_labels:
        dict_label = dict()
        dict_label["class"] = int(label[0])
        dict_label["minXY"] = (label[1] - label[3] / 2, label[2] - label[4] / 2)
        dict_label["maxXY"] = (label[1] + label[3] / 2, label[2] + label[4] / 2)
        dict_labels.append(dict_label)

    return dict_labels


def annot_path_to_img(annot_path):
    name = annot_path[:-4] + ".JPG"
    if os.path.exists(name):
        return name
    return annot_path[:-4] + ".jpg"


def main():
    exts = [".png"]

    img_paths = cleanup.get_img_paths(DATA, exts)
    print("Parsing annotations...")
    annots = cleanup.parse_annots(img_paths, ".txt", annot_parser, normalize=True)

    for annot in annots:
        annot.draw_bounding_boxes()


if __name__ == "__main__":
    main()
