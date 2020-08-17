#!/usr/bin/env python3
import os
from collections import Counter

import albumentations as alb
import albumentations.augmentations.transforms as trans
import cv2
from tqdm import tqdm

from retrain.utils import get_label_path


def get_augmentations():
    return {
        "major": {
            "shift-scale-rot": trans.ShiftScaleRotate(
                shift_limit=0.05,
                rotate_limit=35,
                border_mode=cv2.BORDER_REPLICATE,
                always_apply=True,
            ),
            "crop": trans.RandomResizedCrop(
                100, 100, scale=(0.8, 0.95), ratio=(0.8, 1.2), always_apply=True
            ),
            # "elastic": trans.ElasticTransform(
            #     alpha=0.8,
            #     alpha_affine=10,
            #     sigma=40,
            #     border_mode=cv2.BORDER_REPLICATE,
            #     always_apply=True,
            # ),
            "distort": trans.OpticalDistortion(0.2, always_apply=True),
        },
        "minor": {
            "blur": trans.GaussianBlur(7, always_apply=True),
            "noise": trans.GaussNoise((20.0, 40.0), always_apply=True),
            "bright-contrast": trans.RandomBrightnessContrast(
                0.4, 0.4, always_apply=True
            ),
            "hsv": trans.HueSaturationValue(30, 40, 50, always_apply=True),
            "rgb": trans.RGBShift(always_apply=True),
            "flip": trans.HorizontalFlip(always_apply=True),
        },
    }


def multi_aug(augs, major=True, bbox_params=None):
    major_augs = list(augs["major"].values())
    minor_augs = list(augs["minor"].values())
    return alb.Compose(
        [
            alb.OneOf(major_augs, p=0.9 if major else 0.0),
            alb.OneOf(minor_augs, p=1.0,),
            alb.OneOf(minor_augs, p=0.2,),
        ],
        p=1.0,
        bbox_params=bbox_params,
        label_fields=["classes"],
    )


class Augmenter:
    def __init__(self, img_folder):
        self.img_folder = img_folder

    def get_incr_factors(self, imgs_per_class):
        desired = {i: imgs_per_class for i in range(self.img_folder.num_classes)}
        incr_factors = dict()

        imgs_by_label_count = dict(
            sorted(
                self.img_folder.make_img_dict().items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        # This algorithm could be optimized by sorting by the labels
        # by the most desired and normalizing classes relative to each other
        while sum(desired.values()) > 0:
            for img, labels in imgs_by_label_count.items():
                overshoot = False
                label_counts = Counter(labels)
                for label, count in label_counts.items():
                    if desired[label] - count < 0:
                        overshoot = True
                        break
                if overshoot:
                    continue
                for label, count in label_counts.items():
                    desired[label] -= count
                if img in incr_factors.keys():
                    incr_factors[img] += 1
                else:
                    incr_factors[img] = 1
        return incr_factors

    def augment(self, imgs_per_class, major_aug, min_visibility=0.75):
        incr_factors = self.get_incr_factors(imgs_per_class)

        bbox_params = alb.BboxParams("yolo", min_visibility=min_visibility)
        aug = multi_aug(get_augmentations(), major_aug, bbox_params)

        classes = [i for i in range(len(incr_factors.keys()))]
        pbar = tqdm(desc="Augmenting training images", total=sum(incr_factors.values()))
        for img, count in incr_factors.items():
            augment_img(aug, "compose", img, classes, count=count)
            new_imgs = {
                f"{img[:-4].replace('images', 'aug-images')}_compose-{i}.png"
                for i in range(count)
            }
            self.img_folder.imgs.update(new_imgs)
            self.img_folder.labels.update({get_label_path(img) for img in new_imgs})
            pbar.update(count)

        pbar.close()


def augment_img(aug, suffix, img_path, classes, count=1):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base_name = img_path[:-4]
    label_path = get_label_path(img_path)

    boxes, field_ids = parse_label(label_path)

    i = 0
    while i < count:
        aug_path = f"{base_name.replace('images', 'aug-images')}_{suffix}-{i}.png"
        new_txt_path = get_label_path(aug_path)
        os.makedirs(os.path.dirname(new_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(aug_path), exist_ok=True)

        if os.path.exists(aug_path) and os.path.exists(new_txt_path):
            i += 1
            continue

        try:
            result = aug(image=img, bboxes=boxes, classes=classes)
        except IndexError:
            continue
        aug_img = result["image"]

        new_bboxes = [" ".join(map(str, bbox)) for bbox in result["bboxes"]]
        if len(new_bboxes) != len(boxes):
            continue

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(aug_path, aug_img)

        with open(new_txt_path, "w+") as out:
            for box_i, bbox_str in enumerate(new_bboxes):
                out.write(f"{field_ids[box_i]} {bbox_str}\n")
        i += 1


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
