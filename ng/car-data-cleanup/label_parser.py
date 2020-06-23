#!/usr/bin/env python3
import os
import glob
import csv
import cv2
import imutils
import lxml.etree as ET

DATA = "/media/sng/My Passport/images/make_model_dataset/"
OUTPUT = "./outputs/"


def get_root(xml_file):
    tree = ET.parse(xml_file)
    return tree.getroot()


def get_freq(label_xml):
    root = get_root(label_xml)

    make_models = dict()

    for annot in root.findall(".//annotation"):
        filename = annot.find(".//filename").text
        obj = annot.find(".//object")
        make, model = obj.find(".//name").text.split(" ", 1)
        conf = float(obj.find(".//confidence").text)

        name = make + " " + model
        if name not in make_models.keys():
            make_models[name] = 1
        else:
            make_models[name] += 1

    with open(OUTPUT + "names2.txt", "w+") as out:
        for k, v in make_models.items():
            out.write(f"{k.lower()}: {v}\n")


def combine_names(f1, f2):
    master_dict = dict()

    for f in (f1, f2):
        for line in open(f).read().split("\n"):
            if line == "":
                continue
            make_model, count = line.split(": ")
            count = int(count)

            if make_model not in master_dict.keys():
                master_dict[make_model] = count
            else:
                master_dict[make_model] += count

    with open(OUTPUT + "names_combine.txt", "w+") as out:
        for k, v in sorted(master_dict.items(), key=lambda item: item[1], reverse=True):
            out.write(f"{k.lower()}: {v}\n")


def main():
    # get_freq(DATA + "labels.xml")
    combine_names(OUTPUT + "names.txt", OUTPUT + "names2.txt")


if __name__ == "__main__":
    main()
