#!/usr/bin/env python3
import glob
import xml.etree.ElementTree as ET

DATA = "/media/sng/My Passport/images/"
OUTPUT = "./outputs/"


def get_metadata():
    md_path = DATA + "maingate_meta/*.xml"
    return glob.glob(md_path)

def parse_metadata(files):
    name_counts = dict()

    for md_xml in files:
        tree = ET.parse(md_xml)
        root = tree.getroot()

        for obj in root.findall(".//object"):
            name_obj = obj.find(".//name")
            if name_obj is not None:
                name = name_obj.text
                if "vehicle" in name:
                    name = "vehicle"
            else:
                continue

            if name not in name_counts.keys():
                name_counts[name] = 1
            else:
                name_counts[name] += 1

    print(name_counts)

def main():
    md = get_metadata()
    parse_metadata(md)


if __name__ == "__main__":
    main()
