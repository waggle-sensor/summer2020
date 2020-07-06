#!/usr/bin/env python3
import os
import glob
import csv
import cv2
import imutils
import lxml.etree as ET

DATA = "/media/sng/My Passport/images/"
OUTPUT = "./outputs/"


def get_metadata():
    md_path = DATA + "maingate_meta/*.xml"
    return glob.glob(md_path)


def get_root(xml_file):
    tree = ET.parse(xml_file)
    return tree.getroot()


# Filters files to those with vehicle data
def filter_metadata(files, valid_names):
    # name_counts = dict()

    good_files = list()

    for md_xml in files:
        root = get_root(md_xml)

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


def parse_metadata(files):
    out_csv = open(OUTPUT + "make_model.csv", "w+", newline="")
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
    writer = csv.DictWriter(out_csv, fieldnames=fields)

    writer.writeheader()
    make_models = dict()

    for md_xml in files:
        root = get_root(md_xml)

        for obj in root.findall(".//object"):
            attr_dict = get_attr_dict(obj, fields)

            if attr_dict is None:
                continue

            attr_dict["file"] = md_xml
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

    with open(OUTPUT + "cars.names", "w+") as out:
        for k in make_models.keys():
            out.write(f"{k}\n")


def get_rect_bounds(obj_xml):
    poly = obj_xml.find(".//polygon")

    if poly is None:
        return None, None

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

    return (min_x, min_y), (max_x, max_y)


def draw_bounding_box(md_file):
    root = get_root(md_file)
    path = DATA + root.find("folder").text + "/" + root.find("filename").text

    img = cv2.imread(path)
    for obj in root.findall(".//object"):
        (x0, y0), top_right = get_rect_bounds(obj)
        attr_dict = get_attr_dict(obj, ["make", "model", "model_conf"])

        if x0 is None:
            continue
        if "model_conf" not in attr_dict.keys():
            continue
        name = (
            attr_dict["make"]
            + " "
            + attr_dict["model"]
            + " "
            + str(attr_dict["model_conf"])
        )
        cv2.putText(
            img, name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 10, 0), 4
        )
        cv2.rectangle(img, (x0, y0), top_right, (0, 255, 0), 5)

    disp_img = imutils.resize(img, width=1500)
    cv2.imshow("Bounding box", disp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_file_as_list(filename):
    with open(filename, "r") as file_obj:
        return file_obj.read().split("\n")


def generate_txt_labels(vehicle_md, classes):
    os.makedirs(OUTPUT + "labels", exist_ok=True)
    paths = list()
    for md_file in vehicle_md:
        root = get_root(md_file)
        path = DATA + root.find("folder").text + "/" + root.find("filename").text

        img = cv2.imread(path)
        h, w, _ = img.shape
        class_labels = list()

        for obj in root.findall(".//object"):
            attr = get_attr_dict(obj)

            if attr is None or not ("make" in attr.keys() and "model" in attr.keys()):
                continue

            class_name = f"{attr['make']} {attr['model']}"
            if class_name not in classes:
                continue

            (x0, y0), (x1, y1) = get_rect_bounds(obj)
            rect_h = y1 - y0
            rect_w = x1 - x0
            x_center = rect_w / 2 + x0
            y_center = rect_h / 2 + y0

            idx = classes.index(class_name)
            line = f"{idx} {x_center / w} {y_center / h} {rect_w / w} {rect_h / h}"
            class_labels.append(line)

        txt_filename = root.find("filename").text.replace(".jpg", ".txt")
        txt_path = OUTPUT + "labels/" + txt_filename

        if len(class_labels) > 0:
            print(path)
            paths.append(path)
            with open(txt_path, "w+") as out:
                out.write("\n".join(class_labels))

    with open(OUTPUT + "cars.data", "a+") as out:
        out.write("\n".join(paths))


def main():
    md = get_metadata()

    valid_names = read_file_as_list("valid_names.txt")

    if os.path.exists(OUTPUT + "good_meta.txt"):
        vehicle_md = read_file_as_list(OUTPUT + "good_meta.txt")
    else:
        vehicle_md = filter_metadata(md, valid_names)

    for i in range(0, 100):
        draw_bounding_box(vehicle_md[i])

    if not os.path.exists(OUTPUT + "cars.names"):
        parse_metadata(vehicle_md)

    classes = read_file_as_list("topclasses.names")
    generate_txt_labels(vehicle_md, classes)


if __name__ == "__main__":
    main()
