#!/usr/bin/env python3
import string
import os
import glob
import cv2
import random
import imutils
from math import log, ceil

DATA = "./data/"
OUTPUT = "./output/"
NUM_CLASSES = 8


def generate_all_classes():
    classes = [str(i) for i in range(10)]
    for c in string.ascii_uppercase:
        classes.append(c)
    for c in string.ascii_lowercase:
        classes.append(c)

    return classes


# Implementation of iterative stratification
# Based on https://link.springer.com/content/pdf/10.1007%2F978-3-642-23808-6_10.pdf
def split_test_train(annots, classes, prop_train):
    train = open(OUTPUT + "train.txt", "w+")
    test = open(OUTPUT + "test.txt", "w+")

    for class_name, imgs in img_dict.items():
        random.shuffle(imgs)

        train_num = round(prop_train * len(imgs))
        train_list = imgs[:train_num]
        test_list = imgs[train_num : len(imgs)]

        train.write("\n".join(train_list) + "\n")
        test.write("\n".join(test_list) + "\n")

    train.close()
    test.close()


def move_rename_images():
    os.makedirs(DATA + "images/labeled", exist_ok=True)
    txt_paths = glob.glob(DATA + "images/**/*.txt", recursive=True)

    num_digits = ceil(log(len(txt_paths), 10))

    count = 1
    for txt_path in txt_paths:
        new_path = DATA + f"images/labeled/IMG{str(count).zfill(num_digits)}"
        img_path = txt_path[:-4]
        ext = img_path[-4:]
        os.replace(txt_path, new_path + ".txt")
        try:
            os.replace(img_path, new_path + ext)
            count += 1
        except FileNotFoundError:
            os.remove(new_path + ".txt")


def get_freq(annots):
    freq = dict()
    for a in annots:
        for label in a.labels:
            name = label["class"]
            if name in freq.keys():
                freq[name] += 1
            else:
                freq[name] = 1
    sorted_freq = {
        k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)
    }
    return sorted_freq


def parse_annots(img_path):
    return [Annotation(path) for path in img_path]


class Annotation:
    def __init__(self, img_path):
        self.img_path = img_path

        txt_path = img_path[:-4] + ".txt"
        self.parse_txt(txt_path)

    def parse_txt(self, txt_path):
        self.labels = list()

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

            self.labels.append(label)

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

        for label in self.labels:
            cv2.rectangle(img, label["minXY"], label["maxXY"], (0, 255, 0), 5)

        disp_img = imutils.resize(img, width=1024)
        cv2.imshow("Bounding box", disp_img)
        cv2.waitKey(0)


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    if not os.path.exists(DATA + "images/labeled"):
        move_rename_images()

    exts = [".JPG", ".gif", ".jpg", ".bmp"]
    img_paths_raw = list()
    for ext in exts:
        img_paths_raw += glob.glob(DATA + "images/labeled/*" + ext)

    # Filter out augmented images
    img_paths = [img for img in img_paths_raw if "_" not in img]
    annots = parse_annots(img_paths)

    # annots[1].draw_bounding_boxes()

    all_classes = generate_all_classes()

    freq = get_freq(annots)
    with open(OUTPUT + "freq.txt", "w+") as out:
        for k, v in freq.items():
            out.write(f"{k}: {v}\n")

    classes = list(freq.keys())[:NUM_CLASSES]
    with open(OUTPUT + "chars.names", "w+") as out:
        out.write("\n".join(classes) + "\n")

    if not os.path.exists(DATA + "labels/labeled"):
        os.makedirs(DATA + "labels/labeled")
        for a in annots:
            for label in a.labels:
                if label["class"] in classes:
                    a.make_darknet_label(classes)
                    break

    # split_test_train(annots, classes, 0.75)


if __name__ == "__main__":
    main()
