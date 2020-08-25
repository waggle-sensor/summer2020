"""
Module to augment a set of labeled images, with bounding box transformations.

Modify get_augmentations() and multi_aug() as necessary for your specific
deployment needs with the albumentations library.
"""

import os
from collections import Counter

import cv2
import numpy as np
import albumentations as alb
import albumentations.augmentations.transforms as trans
from tqdm import tqdm

from retrain.utils import get_label_path


def get_augmentations():
    """Get a list of 'major' and 'minor' augmentation functions for the pipeline in a dictionary."""
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
    """Get a composite augmentation function incorporation 'major' and 'minor' transformations.

    Modify this function as needed.
    """
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
    )


class Augmenter:
    """Wrapper class for an ImageFolder object to begin augmenting images."""

    def __init__(self, img_folder):
        self.img_folder = img_folder

    def get_incr_factors(self, imgs_per_class):
        """Get a dictionary of images in the image folder and the number of times each
        item should be agumented."""
        desired = {i: imgs_per_class for i in range(self.img_folder.num_classes)}
        img_dict = self.img_folder.make_img_dict()
        incr_factors = {img: 0 for img in img_dict.keys()}

        imgs_by_label_count = dict(
            sorted(img_dict.items(), key=lambda x: len(x[1]), reverse=True,)
        )

        # This algorithm could be optimized by sorting by the labels
        # by the most desired and normalizing classes relative to each other
        while sum(desired.values()) > 0:
            for img, labels in imgs_by_label_count.items():
                label_counts = Counter(labels)

                if any(desired[label] < count for label, count in label_counts.items()):
                    continue
                for label, count in label_counts.items():
                    desired[label] -= count
                incr_factors[img] += 1

        return incr_factors

    def augment(self, imgs_per_class, major_aug, min_visibility=0.75):
        """Augment all images in the image folder, adding the augmentations to the folder.

        Parameters:
            imgs_per_class   Target number of images. If there are more samples in the folder
                             than this number for a class, no augmentation will be performed.
            major_aug        A boolean variable determining if 'major' transformations will be used.
            min_visibility   Minimum visibility of the resultant bounding boxes after augmentation.
                             This is a value in (0.0, 1.0] relative to the area of the bounding box.
        """
        incr_factors = self.get_incr_factors(imgs_per_class)

        bbox_params = alb.BboxParams(
            "yolo", min_visibility=min_visibility, label_fields=["classes"]
        )
        aug = multi_aug(get_augmentations(), major_aug, bbox_params)

        pbar = tqdm(desc="Augmenting training images", total=sum(incr_factors.values()))
        for img, count in incr_factors.items():
            augment_img(aug, "compose", img, count=count)
            new_imgs = {
                f"{img[:-4].replace('images', 'aug-images')}_compose-{i}.png"
                for i in range(count)
            }
            self.img_folder.imgs.update(new_imgs)
            self.img_folder.labels.update({get_label_path(img) for img in new_imgs})
            pbar.update(count)

        pbar.close()


def augment_img(aug, suffix, img_path, count=1):
    """Iteratively augment a single image with a given augmentation function.
    Generates transformed labels for each augmentation

    Parameters:
        aug      Augmentation function from albumentations
        suffix   String suffix appended to each augmentation file and label
        img_path Filesystem path to the image to augment
        count    Number of augmentations to perform
    """
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
            result = aug(image=img, bboxes=boxes, classes=field_ids)
        except IndexError:
            continue
        aug_img = result["image"]

        new_bboxes = [" ".join(map(str, bbox)) for bbox in result["bboxes"]]

        # Check if bounding boxes have been removed due to visibility criteria
        if len(new_bboxes) != len(boxes):
            continue

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_path, aug_img)

        with open(new_txt_path, "w+") as out:
            for box_i, bbox_str in enumerate(new_bboxes):
                out.write(f"{field_ids[box_i]} {bbox_str}\n")
        i += 1


def parse_label(label_path):
    """Parse a Darknet format label, returning bounding boxes and class IDs."""
    labels = open(label_path, "r").read().split("\n")

    boxes = list()
    field_ids = list()

    for label in labels:
        box = list()
        for i, info in enumerate(label.split(" ")):
            if i == 0:
                field_ids.append(int(info))
            else:
                box.append(np.float64(info))
        boxes.append(box)

    return boxes, field_ids
