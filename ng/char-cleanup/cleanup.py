#!/usr/bin/env python3
import string
import os
import glob
import random

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


def create_labels(img_paths, classes, all_classes):
    for path in img_paths:
        orig_idx = int(path.split("Sample0")[1][:2]) - 1
        if all_classes[orig_idx] not in classes:
            continue

        txt_path = path.replace("images", "labels").replace(".png", ".txt")

        folder_path = "/".join(txt_path.split("/")[:-1])
        os.makedirs(folder_path, exist_ok=True)

        idx = classes.index(all_classes[orig_idx])
        line = f"{idx} 0.5 0.5 1 1"

        with open(txt_path, "w+") as out:
            out.write(line)


def split_test_train(img_dict, prop_train):
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


def get_img_dict(img_paths, classes):
    """Create a dictionary of images by class."""
    img_dict = dict()

    for c in classes:
        img_dict[c] = list()
    for path in img_paths:
        idx = int(path.split("Sample0")[1][:2]) - 1
        img_dict[classes[idx]].append(path)

    return img_dict


def filter_img_dict(img_dict, classes):
    filtered_dict = dict()
    for k, v in img_dict.items():
        if k in classes:
            filtered_dict[k] = v
    return filtered_dict


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    img_paths = glob.glob(DATA + "images/**/*.png", recursive=True)

    all_classes = generate_all_classes()
    img_dict = get_img_dict(img_paths, all_classes)

    freq = {k: len(v) for k, v in img_dict.items()}
    sorted_freq = {
        k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)
    }

    with open(OUTPUT + "freq.txt", "w+") as out:
        for k, v in sorted_freq.items():
            out.write(f"{k}: {v}\n")

    classes = list(sorted_freq.keys())[:NUM_CLASSES]

    with open(OUTPUT + "chars.names", "w+") as out:
        out.write("\n".join(classes) + "\n")

    create_labels(img_paths, classes, all_classes)

    filtered_dict = filter_img_dict(img_dict, classes)
    split_test_train(filtered_dict, 0.75)


if __name__ == "__main__":
    main()
