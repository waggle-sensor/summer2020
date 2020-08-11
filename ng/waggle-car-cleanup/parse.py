#!/usr/bin/env python3
# Analyze KAIST scene text dataset, from
# http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database

import lxml.etree as ET
import os
import cleanup
import tqdm

DATA = "./data/"
OUTPUT = "./output/"


def get_root(xml_file):
    tree = ET.parse(xml_file)
    return tree.getroot()


def get_attr_dict(obj_xml, fields=None):
    attr_str = obj_xml.find(".//attributes").text

    if attr_str == "" or attr_str is None:
        return None

    attr_str = attr_str.replace("_ ", "_")
    attributes = attr_str.split(" ")

    attr_dict = dict()

    for i, attr in enumerate(attributes):
        if "_" in attr:
            k, v = attr.split("_", 1)
            if fields is not None and k not in fields:
                continue
            attr_dict[k] = v
            if k in ("make", "model", "type", "use"):
                if i + 3 >= len(attributes):
                    continue

                conf_pair = attributes[i + 3].split("_")
                if len(conf_pair) != 2 or conf_pair[1] == "" or conf_pair[0][0] != "(":
                    continue

                try:
                    conf_val = int(conf_pair[1])
                    attr_dict[k + "_conf"] = conf_val
                except ValueError:
                    pass
    return attr_dict


def annot_parser(xml):
    labels = list()
    try:
        root = ET.parse(xml).getroot()
    except ET.XMLSyntaxError:
        return labels

    valid_names = open("valid_names.txt", "r").read().split("\n")

    fields = [
        "file",
        "vehicle",
        "perspective",
        "make",
        "make_conf",
        "model",
        "model_conf",
        "type",
        "type_conf",
        "use",
        "use_conf",
    ]

    useless = True

    for obj in root.findall(".//object"):
        name_obj = obj.find(".//name")
        if name_obj is not None:
            name = name_obj.text
            if "vehicle" in name or name in valid_names:
                useless = False

        attr_dict = get_attr_dict(obj, fields)

        if attr_dict is None:
            continue
        label = dict()
        if "make" in attr_dict.keys() and "model" in attr_dict.keys():
            label["class"] = attr_dict["make"] + " " + attr_dict["model"]
        else:
            continue

        poly = obj.find(".//polygon")

        if poly is None:
            continue

        max_x = float("-inf")
        max_y = float("-inf")
        min_x = float("inf")
        min_y = float("inf")

        for pt in poly.findall(".//pt"):
            x = int(pt.find(".//x").text)
            y = int(pt.find(".//y").text)

            max_x = max(x, max_x)
            min_x = min(x, min_x)
            max_y = max(y, max_y)
            min_y = min(y, min_y)
        label["minXY"] = (min_x, min_y)
        label["maxXY"] = (max_x, max_y)

        labels.append(label)

    return labels


def annot_path_to_img(annot_path):
    return annot_path[:-4] + ".jpg"


def main():
    exts = [".JPG", ".jpg"]

    print("Cleaning up data...")
    cleanup.clean(DATA, OUTPUT, exts, ".xml", annot_parser, annot_path_to_img)
    img_paths = cleanup.get_img_paths(DATA + "/images/labeled", exts)

    print("Parsing annotations...")
    annots = cleanup.parse_annots(img_paths, ".xml", annot_parser)

    classes = open("output/chars.names", "r").read().split("\n")[:-1]
    for a in tqdm.tqdm(annots, "Cropping letters"):
        a.crop_labels(classes, DATA + "images/objs/")


if __name__ == "__main__":
    main()
