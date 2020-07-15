#!/usr/bin/env python3
import albumentations as alb
import albumentations.augmentations.transforms as trans
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from shutil import copyfile
import tqdm
import os
import argparse

AUGS_PER_IMG = 120


def get_augmentations():
    return {
        "blur": trans.GaussianBlur(7, always_apply=True),
        "noise": trans.GaussNoise((20.0, 40.0), always_apply=True),
        "shift-scale-rot": trans.ShiftScaleRotate(shift_limit=0.05, always_apply=True),
        "crop": trans.RandomResizedCrop(
            100, 100, scale=(0.8, 0.95), ratio=(0.8, 1.2), always_apply=True
        ),
        "bright-contrast": trans.RandomBrightnessContrast(0.4, 0.4, always_apply=True),
        "hsv": trans.HueSaturationValue(30, 40, 50, always_apply=True),
        "rgb": trans.RGBShift(30, 30, 30, always_apply=True),
        "distort": trans.OpticalDistortion(0.2, always_apply=True),
        "elastic": trans.ElasticTransform(
            alpha=0.8,
            alpha_affine=10,
            sigma=40,
            border_mode=cv2.BORDER_CONSTANT,
            always_apply=True,
        ),
    }


def multi_aug(augs):
    return alb.Compose(
        [
            alb.OneOf(
                [
                    augs["shift-scale-rot"],
                    augs["crop"],
                    augs["elastic"],
                    augs["distort"],
                ],
                p=0.9,
            ),
            alb.OneOf(
                [
                    augs["blur"],
                    augs["noise"],
                    augs["bright-contrast"],
                    augs["hsv"],
                    augs["rgb"],
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    augs["blur"],
                    augs["noise"],
                    augs["bright-contrast"],
                    augs["hsv"],
                    augs["rgb"],
                ],
                p=0.2,
            ),
        ],
        p=1.0,
    )


def get_incr_factors(imgs, imgs_per_class):
    class_counts = dict()
    img_classes = dict()

    for img in imgs:
        txt_path = img.replace("images", "labels")[:-4] + ".txt"
        with open(txt_path, "r") as txt:
            class_num = int(txt.read().split(" ")[0])
            img_classes[img] = class_num
            if class_num not in class_counts.keys():
                class_counts[class_num] = 1
            else:
                class_counts[class_num] += 1

    imgs_per_class = max(imgs_per_class, max(class_counts.values()))
    img_factors = dict()
    for img, class_num in img_classes.items():
        img_factors[img] = round(imgs_per_class / class_counts[class_num])

    return img_factors


def parse_label(label_path):
    labels = open(label_path, "r").read().split("\n")

    boxes = list()
    field_ids = list()

    for label in labels:
        box = list()
        for i, info in enumerate(label.split(" ")):
            if i == 0:
                field_ids.append(int(info))
            else:
                box.append(float(info))
        boxes.append(box)

    return boxes, field_ids


def augment(train_list, compose, balance, imgs_per_class):
    augs = get_augmentations()
    imgs = open(train_list, "r").read().split("\n")[:-1]
    orig_imgs = imgs.copy()

    if compose:
        compose_aug = multi_aug(augs)
        augs = {"comp": compose_aug}

    if balance:
        incr_factors = get_incr_factors(orig_imgs, imgs_per_class)

    for k, img_path in enumerate(tqdm.tqdm(orig_imgs, "Augmenting images")):

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = img_path[:-4]

        txt_path = img_path.replace("images", "labels")[:-4] + ".txt"
        boxes, field_ids = parse_label(txt_path)
        bbox_params = {"format": "yolo", "min_visibility": 0.50}

        if balance:
            incr_factor = incr_factors[img_path]
        else:
            incr_factor = AUGS_PER_IMG

        augs_per_trans = int(incr_factor / len(augs.keys()))

        for name, aug in augs.items():
            for i in range(augs_per_trans):
                result = aug(
                    image=img,
                    # bboxes=boxes,
                    category_id=field_ids,
                    bbox_params=bbox_params,
                )
                aug_img = result["image"]
                new_bbox = result["bboxes"]

                print(new_bbox)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                aug_path = f"{base_name.replace('images', 'aug-images')}_{name}-{i}.png"
                os.makedirs(os.path.dirname(aug_path), exist_ok=True)
                cv2.imwrite(aug_path, aug_img)

                imgs.append(aug_path)

                new_txt_path = aug_path.replace("images", "labels")[:-4] + ".txt"
                os.makedirs(os.path.dirname(new_txt_path), exist_ok=True)
                copyfile(txt_path, new_txt_path)

    with open(train_list[:-4] + "-aug.txt", "w+") as out:
        out.write("\n".join(imgs))


if __name__ == "__main__":
    random.seed("sage")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balance",
        action="store_true",
        default=False,
        help="balance augmentation based on class size",
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        default=False,
        help="use a composition of augmentations, not individual transforms",
    )
    parser.add_argument(
        "--train_list", required=True, help="text file of training images"
    )
    parser.add_argument("--imgs_per_class", default=15000, required=False, type=int)
    opt = parser.parse_args()

    augment(opt.train_list, opt.compose, opt.balance, opt.imgs_per_class)
