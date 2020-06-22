#!/usr/bin/env python3
import os
import glob
import csv
import lxml.etree as ET

DATA = "/media/sng/My Passport/images/"
OUTPUT = "./outputs/"


def get_metadata():
    md_path = DATA + "maingate_meta/*.xml"
    return glob.glob(md_path)

# Filters files to those with vehicle data
def filter_metadata(files, valid_names):
    # name_counts = dict()

    good_files = list()

    for md_xml in files:
        tree = ET.parse(md_xml)
        root = tree.getroot()

        useless = True

        for obj in root.findall(".//object"):
            name_obj = obj.find(".//name")
            if name_obj is not None:
                name = name_obj.text
                if "vehicle" in name or name in valid_names:
                    useless = False

            # if name not in name_counts.keys():
            #     name_counts[name] = 1
            # else:
            #     name_counts[name] += 1

        if not useless:
            good_files.append(md_xml)

    with open(OUTPUT + "good_meta.txt", "w+") as out:
        out.write("\n".join(good_files))

    return good_files


def parse_metadata(files):
    count = 0

    out_csv = open(OUTPUT + "make_model.csv", "w+", newline="")
    fields = "file,vehicle,perspective,make,make_conf,model,model_conf,type,type_conf,use,use_conf".split(",")
    writer = csv.DictWriter(out_csv, fieldnames=fields)

    writer.writeheader()

    make_models = dict()

    for md_xml in files:
        tree = ET.parse(md_xml)
        root = tree.getroot()

        
        for obj in root.findall(".//object"):
            attr_str = obj.find(".//attributes").text
            
            if attr_str == "" or attr_str is None:
                continue

            attr_str = attr_str.replace("_ ", "_")
            attributes = attr_str.split(" ")

            attr_dict = {"file": md_xml}

            for i, attr in enumerate(attributes):
                if "_" in attr and len(attr.split("_")) == 2:
                    k, v = attr.split("_")
                    if k not in fields:
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

            if len(attr_dict.keys()) > 1:
                writer.writerow(attr_dict)
            if "make" in attr_dict.keys() and "model" in attr_dict.keys():
                name = attr_dict["make"] + " " + attr_dict["model"]
                if name not in make_models.keys():
                    make_models[name] = 1
                else:
                    make_models[name] += 1

    out_csv.close()

    with open(OUTPUT + "names.txt", "w+") as out:
        for k, v in make_models.items():
            out.write(f"{k}: {v}\n")


def read_file_as_list(filename):
    with open(filename, "r") as file_obj:
        return file_obj.read().split("\n")

def main():
    md = get_metadata()
    
    valid_names = read_file_as_list("valid_names.txt")
    
    if os.path.exists(OUTPUT + "good_meta.txt"):
        vehicle_md = read_file_as_list(OUTPUT + "good_meta.txt")
    else:
        vehicle_md = filter_metadata(md, valid_names)

    parse_metadata(vehicle_md)


if __name__ == "__main__":
    main()
