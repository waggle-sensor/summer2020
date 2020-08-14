#!/usr/bin/env python3
import string
import os
import glob
import random
from math import log, ceil
import cv2
import imutils

"""
Cleans up a set of annotated scene images for training.

Provides utility functions for analyzing class frequency, organizing images by classes, 
creating Darknet labels, visualizing bounding boxes, and splitting data into train/test sets.

You only need to provide your own annotation parser and a way to extract
an image path from an annotation.

In this repo, this file is symbolically linked between char-scene-cleanup/ and kaist/.
"""


def generate_all_classes():
    classes = [str(i) for i in range(10)]
    for c in string.ascii_uppercase:
        classes.append(c)
    for c in string.ascii_lowercase:
        classes.append(c)

    return classes


# Implementation of iterative stratification
# Based on https://link.springer.com/content/pdf/10.1007%2F978-3-642-23808-6_10.pdf
def split_test_train(annots, classes, prop_train, output):
    random.seed("sage")

    train = open(output + "train.txt", "w+")
    test = open(output + "test.txt", "w+")

    remaining_examples = sort_freq_dict(get_freq(annots), False)
    for k in list(remaining_examples.keys()):
        if k not in classes:
            del remaining_examples[k]
    print(remaining_examples)
    train_desire = dict()
    test_desire = dict()

    for k, v in remaining_examples.items():
        train_num = round(prop_train * v)
        train_desire[k] = train_num
        test_desire[k] = v - train_num

    for i in range(len(classes)):
        cur_label = list(remaining_examples.keys())[i]
        available = list()

        for img in annots:
            for label in img.labels:
                if label["class"] == cur_label:
                    available.append(img)
                    break

        random.shuffle(available)

        for img in available:

            if train_desire[cur_label] > test_desire[cur_label] or (
                train_desire[cur_label] == test_desire[cur_label]
                and random.choice([True, False])
            ):
                train.write(img.img_path + "\n")
                chosen_set = train_desire
            else:
                test.write(img.img_path + "\n")
                chosen_set = test_desire

            for label in img.labels:
                if label["class"] in classes:
                    chosen_set[label["class"]] -= 1
                    remaining_examples[label["class"]] -= 1
            annots.remove(img)

        remaining_examples = sort_freq_dict(remaining_examples, False)

    train.close()
    test.close()
    print(train_desire)


def move_rename_images(annot_ext, data, annot_path_to_img):
    os.makedirs(data + "/images/labeled", exist_ok=True)
    annot_paths = glob.glob(data + "labels/**/*" + annot_ext, recursive=True)

    num_digits = ceil(log(len(annot_paths), 10))

    count = 1
    for annot_path in annot_paths:
        new_path = data + f"images/labeled/IMG{str(count).zfill(num_digits)}"
        img_path = annot_path_to_img(annot_path)
        img_ext = img_path[-4:].lower()
        try:
            os.replace(img_path, new_path + img_ext)
            os.replace(annot_path, new_path + annot_ext)
            count += 1
        except FileNotFoundError:
            print("Image not found: Skipping " + annot_path)


def sort_freq_dict(freq, rev=True):
    return {
        k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=rev)
    }


def get_freq(annots):
    freq = dict()
    for a in annots:
        for label in a.labels:
            name = label["class"]
            if name in freq.keys():
                freq[name] += 1
            else:
                freq[name] = 1

    return freq


def parse_annots(img_path, ext, annot_parser, normalize=False, **kwargs):
    return [
        Annotation(path, ext, annot_parser, normalize, **kwargs) for path in img_path
    ]


class Annotation:
    def __init__(self, img_path, annot_ext, annot_parser, normalize=False, **kwargs):
        self.img_path = img_path

        annot_path = img_path[:-4] + annot_ext
        self.labels = annot_parser(annot_path, **kwargs)
        if normalize:
            self.normalize_labels()

    def normalize_labels(self):
        normalized_labels = list()
        img = cv2.imread(self.img_path)
        h, w, _ = img.shape
        resize = lambda coord: (round(w * coord[0]), round(h * coord[1]))
        for label in self.labels:
            label["minXY"] = resize(label["minXY"])
            label["maxXY"] = resize(label["maxXY"])
            normalized_labels.append(label)
        self.labels = normalized_labels

    def make_darknet_label(self, class_list):
        out_path = self.img_path.replace("images", "labels")[:-4] + ".txt"
        lines = list()
        for label in self.labels:
            if label["class"] not in class_list:
                continue
            idx = class_list.index(label["class"])

            (x0, y0) = label["minXY"]
            (x1, y1) = label["maxXY"]

            rect_h = y1 - y0
            rect_w = x1 - x0
            x_center = rect_w / 2 + x0
            y_center = rect_h / 2 + y0

            img = cv2.imread(self.img_path)
            h, w, _ = img.shape

            line = f"{idx} {x_center / w} {y_center / h} {rect_w / w} {rect_h / h}"
            lines.append(line)

        if len(lines) == 0:
            return

        with open(out_path, "w+") as out:
            out.write("\n".join(lines))

    def draw_bounding_boxes(self):
        img = cv2.imread(self.img_path)
        print(self.img_path)
        for label in self.labels:
            cv2.rectangle(img, label["minXY"], label["maxXY"], (0, 255, 0), 3)
            cv2.putText(
                img, label["class"], label["minXY"], cv2.FONT_HERSHEY_SIMPLEX, 3.0, 2, 5
            )
        if img is None:
            return
        disp_img = imutils.resize(img, width=2048)
        cv2.imshow("Bounding box", disp_img)
        cv2.waitKey(0)

    def crop_labels(self, class_list, output_path):
        num_digits = ceil(log(len(class_list), 10))
        for label in self.labels:
            if label["class"] not in class_list:
                continue
            idx = class_list.index(label["class"])

            folder = f"{output_path}/Class{str(idx).zfill(num_digits)}-{label['class']}"
            os.makedirs(folder, exist_ok=True)
            name = str(len(os.listdir(folder)) + 1) + ".png"

            img = cv2.imread(self.img_path)

            (x0, y0) = label["minXY"]
            (x1, y1) = label["maxXY"]
            cropped_img = img[y0:y1, x0:x1]
            if len(cropped_img) != 0:
                cv2.imwrite(folder + "/" + name, cropped_img)


def get_img_paths(data, img_exts):
    img_paths = list()
    for img_ext in img_exts:
        img_paths += glob.glob(data + "/**/*" + img_ext, recursive=True)

    return img_paths


def clean(
    data, output, img_exts, annot_ext, annot_parser, annot_path_to_img, top_classes=None
):
    os.makedirs(output, exist_ok=True)

    if not os.path.exists(data + "/images/labeled"):
        move_rename_images(annot_ext, data, annot_path_to_img)

    # Filter out augmented images
    img_paths = [
        img
        for img in get_img_paths(data + "/images/labeled", img_exts)
        if "_" not in img
    ]

    annots = parse_annots(img_paths, annot_ext, annot_parser)

    freq = sort_freq_dict(get_freq(annots))
    with open(output + "freq.txt", "w+") as out:
        for k, v in freq.items():
            out.write(f"{k}: {v}\n")

    if top_classes is None:
        classes = list(freq.keys())
    else:
        classes = list(freq.keys())[:top_classes]
    with open(output + "chars.names", "w+") as out:
        out.write("\n".join(classes) + "\n")

    if not os.path.exists(data + "labels/labeled"):
        os.makedirs(data + "labels/labeled")
        for a in annots:
            for label in a.labels:
                if label["class"] in classes:
                    a.make_darknet_label(classes)
                    break

    split_test_train(annots, classes, 0.75, output)
