#!/usr/bin/env python3
import string
import os
import glob
import random
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


def create_labels(img_dict, classes):
    for class_name, imgs in img_dict.items():
        idx = classes.index(class_name)
        for path in imgs:
            txt_path = path.replace("images", "labels").replace(".png", ".txt")

            folder_path = "/".join(txt_path.split("/")[:-1])
            os.makedirs(folder_path, exist_ok=True)

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

def move_rename_images():
    os.makedirs(DATA + "images/labeled", exist_ok=True)
    txt_paths = glob.glob(DATA + "images/**/*.txt", recursive=True)
    #print(txt_paths)
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
    print(len(img_paths))

    all_classes = generate_all_classes()
    # img_dict = get_img_dict(img_paths, all_classes)

    # freq = {k: len(v) for k, v in img_dict.items()}
    # sorted_freq = {
    #     k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)
    # }

    # with open(OUTPUT + "freq.txt", "w+") as out:
    #     for k, v in sorted_freq.items():
    #         out.write(f"{k}: {v}\n")

    # classes = list(sorted_freq.keys())[:NUM_CLASSES]

    # with open(OUTPUT + "chars.names", "w+") as out:
    #     out.write("\n".join(classes) + "\n")

    # filtered_dict = filter_img_dict(img_dict, classes)

    # create_labels(filtered_dict, classes)
    # split_test_train(filtered_dict, 0.75)


if __name__ == "__main__":
    main()
