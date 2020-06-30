#!/usr/bin/env python3
# Analyze KAIST scene text dataset, from http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database

import glob
import lxml.etree as ET
import os
import cleanup

DATA = "./data/"
OUTPUT = "./output/"


def annot_parser(xml):
    labels = list()
    try:
        root = ET.parse(xml).getroot()
    except ET.XMLSyntaxError:
        return labels

    try:
        words = root.find("image").find("words").findall(".//word")
    except AttributeError:
        return labels

    for word in words:
        chars = word.findall(".//character")
        for char in chars:
            label = dict()

            label["class"] = char.attrib["char"]
            min_x = int(char.attrib["x"])
            min_y = int(char.attrib["y"])
            label["minXY"] = (min_x, min_y)
            label["maxXY"] = (
                min_x + int(char.attrib["width"]),
                min_y + int(char.attrib["height"]),
            )

            labels.append(label)

    return labels


def annot_path_to_img(annot_path):
    name = annot_path[:-4] + ".JPG"
    if os.path.exists(name):
        return name
    return annot_path[:-4] + ".jpg"


def main():
    exts = [".JPG", ".jpg"]
    cleanup.clean(DATA, OUTPUT, exts, ".xml", annot_parser, annot_path_to_img)


if __name__ == "__main__":
    main()
