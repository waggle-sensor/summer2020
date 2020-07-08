#!/usr/bin/env python3
import albumentations as alb
import albumentations.augmentations.transforms as trans
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from shutil import copyfile

AUGS_PER_TRANSFORM = 120
COMPOSE = True


def get_augmentations():
    return {
        "blur": trans.GaussianBlur(7, always_apply=True),
        "noise": trans.GaussNoise((20.0, 40.0), always_apply=True),
        "shift-scale-rot": trans.ShiftScaleRotate(shift_limit=0.05, always_apply=True),
        "crop": trans.RandomResizedCrop(
            100, 100, scale=(0.7, 0.95), ratio=(0.8, 1.2), always_apply=True
        ),
        "bright-contrast": trans.RandomBrightnessContrast(0.4, 0.4, always_apply=True),
        "hsv": trans.HueSaturationValue(30, 40, 50, always_apply=True),
        "rgb": trans.RGBShift(30, 30, 30, always_apply=True),
        "distort": trans.OpticalDistortion(0.2, always_apply=True),
        "elastic": trans.ElasticTransform(
            alpha=0.6, border_mode=cv2.BORDER_WRAP, always_apply=True
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


def augment(train_list):
    random.seed("sage")

    augs = get_augmentations()
    imgs = open(train_list, "r").read().split("\n")[:-1]
    orig_imgs = imgs.copy()

    if COMPOSE:
        compose_aug = multi_aug(augs)
        augs = {"comp": compose_aug}

    for k, img_path in enumerate(orig_imgs):
        if (k + 1) % 10 == 0:
            print(f"{k+1}/{len(orig_imgs)}")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = img_path[:-4]

        for name, func in augs.items():
            for i in range(AUGS_PER_TRANSFORM):
                aug_path = f"{base_name}_{name}-{i}.png"
                aug_img = func(image=img)["image"]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_path, aug_img)

                imgs.append(aug_path)

                txt_path = img_path.replace("images", "labels")[:-4] + ".txt"
                new_txt_path = aug_path.replace("images", "labels")[:-4] + ".txt"
                copyfile(txt_path, new_txt_path)

    with open(train_list[:-4] + "-aug.txt", "w+") as out:
        out.write("\n".join(imgs))


if __name__ == "__main__":
    augment(sys.argv[1])
