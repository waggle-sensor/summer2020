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


def parse_labels(label_xml):
    root = get_root(label_xml)

    labels = list()

    for annot in root.findall(".//annotation"):
        label = dict()
        label["filename"] = annot.find("filename").text
        
        obj = annot.find(".//object")
        label["make"], label["model"] = obj.find(".//name").text.split(" ", 1)
        label["conf"] = float(obj.find(".//confidence").text)
        
        labels.append(label)

    return labels

def get_freq(labels):
    make_models = dict()

    for label in labels:
        name = label["make"] + " " + label["model"]

        if name not in make_models.keys():
            make_models[name] = 1
        else:
            make_models[name] += 1

    with open(OUTPUT + "names2.txt", "w+") as out:
        for k, v in make_models.items():
            out.write(f"{k.lower()}: {v}\n")


def generate_txt_labels(labels, classes):
    filepaths = list()
    for label in labels:
        name = label["make"] + " " + label["model"]

        if name.lower() in classes:
            filepaths.append(DATA + label["filename"])
            txt_filename = label["filename"].replace(".jpg", ".txt")
            txt_path = OUTPUT + "labels/" + txt_filename

            idx = classes.index(name.lower())
            line = f"{idx} 0.5 0.5 1 1"
            
            with open(txt_path, "w+") as out:
                out.write(line)
    
    with open(OUTPUT + "cars.data", "w+") as out:
        out.write("\n".join(filepaths))


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
    label_dicts = parse_labels(DATA + "labels.xml")
    get_freq(label_dicts)
    combine_names(OUTPUT + "names.txt", OUTPUT + "names2.txt")

    classes = open("topclasses.names", "r").read().split("\n")
    generate_txt_labels(label_dicts, classes)


if __name__ == "__main__":
    main()
