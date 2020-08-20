#!/usr/bin/env python3
# Analyze KAIST scene text dataset, from
# http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database

import lxml.etree as ET
import os
import cleanup
import tqdm
import random
from glob import glob

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


key_map = {
    "c": "Cab",
    "n": "Convertible",
    "u": "Coupe",
    "h": "Hatchback",
    "m": "Minivan",
    "s": "SUV",
    "d": "Sedan",
    "v": "Van",
    "w": "Wagon",
}


def increment(mapping, make_model, car_type):
    if isinstance(mapping[make_model], str):
        return
    mapping[make_model][car_type] += 1
    car_type_count = mapping[make_model][car_type]
    if (
        car_type_count >= 10
        and car_type_count / sum(mapping[make_model].values()) >= 0.9
    ):
        mapping[make_model] = car_type


def get_classes(label_file):
    return [int(line.split(" ")[0]) for line in open(label_file).read().split("\n")]


def relabel(annot, car_types, mapping):
    labels = annot.labels
    filename = f"output/labels/{annot.img_path.split('/')[-1][:-4]}.txt"
    remove_list = list()
    if os.path.exists(filename):
        label_classes = get_classes(filename)
        for txt_class, annot_label in zip(label_classes, labels):
            increment(mapping, annot_label["class"], car_types[txt_class])
        return

    i = 0
    while i < len(labels):
        label = labels[i]
        annot.labels = [label]
        make_model = label["class"]
        if make_model in car_types:
            i += 1
            continue
        print(make_model)
        if make_model != "unknown unknown" and isinstance(mapping[make_model], str):
            label["class"] = mapping[make_model]
            continue

        choice = annot.draw_bounding_boxes()
        if choice == ord("x"):
            remove_list.append(label)
            i += 1
            continue
        for key, car_type in key_map.items():
            if choice == ord(key):
                label["class"] = car_type
                break
        if label["class"] in key_map.values():
            increment(mapping, make_model, car_type)
            i += 1

    annot.labels = [lab for lab in labels if lab not in remove_list]
    annot.make_darknet_label(car_types, filename)


def count_types(folder, car_types):
    freq = {car: 0 for car in car_types}
    labels = glob(folder + "/*.txt")
    for label in labels:
        for car_type in get_classes(label):
            freq[car_types[car_type]] += 1

    return freq


def annot_path_to_img(annot_path):
    return annot_path[:-4] + ".jpg"


def main():
    exts = [".JPG", ".jpg"]

    print("Cleaning up data...")
    # cleanup.clean(DATA, OUTPUT, exts, ".xml", annot_parser, annot_path_to_img)
    img_paths = cleanup.get_img_paths(DATA + "/images/labeled", exts)

    annots = cleanup.parse_annots(img_paths, ".xml", annot_parser)

    old_classes = open("output/make_model.names", "r").read().split("\n")[:-1]
    new_classes = open("output/cars.names", "r").read().split("\n")[:-1]

    mapping = {old: {new: 0 for new in new_classes} for old in old_classes}

    for i, a in enumerate(annots):
        print(i)
        relabel(a, new_classes, mapping)

    # for a in tqdm.tqdm(annots, "Cropping..."):
    #     # a.draw_bounding_boxes()
    #     try:
    #         a.crop_labels(classes, "data/images/obj/")
    #     finally:
    #         pass


if __name__ == "__main__":
    main()
