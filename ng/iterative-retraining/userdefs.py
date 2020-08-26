"""
A collection of essential user-defined functions used in various stages
of the sampling and retraining pipeline. Modify these files as needed, to
fit the needs and structure of your data.

This file and a configuration file should be finalized before deploying at the edge.
"""

import os
import cv2
import albumentations as alb
import albumentations.augmentations.transforms as trans
import retrain.sampling as sample


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
    """Get a composite augmentation function incorporation 'major' and 'minor' transformations."""
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


def label_sample_set(img_path, classes):
    """Sample function to label an image path with its ground truth with a list of labels.

    This function is customizable (e.g. including a GUI to annotate) depending on your needs.
    It should return a list of tuples, with each tuple representing a label with the values
    (class_label, bounding_box_x_center, bb_y_center, bb_width, bb_height)
    These coordinates should also be normalized according to the image's width and height.
    """
    path = img_path.replace("images", "classes")[:-4] + ".txt"
    if os.path.exists(path):
        labels = map(lambda x: map(float, x.split(" ")), open(path).read().split("\n"))
        for label in labels:
            label[0] = classes[int(label[0])]
        return labels
    return []


def get_sample_methods():
    """Get a dictionary of sampling methods, containing prefix names as keys and function-parameter
    tuples as values.

    Refer to the README for documentation on implementing your own sampling functions.
    """
    return {
        "median-below-thresh": (sample.median_below_thresh_sample, {"thresh": 0.0}),
        "median-thresh": (sample.median_thresh_sample, {"thresh": 0.0}),
        "bin-quintile": (
            sample.bin_sample,
            {"stratify": False, "num_bins": 5, "curve": sample.const, "thresh": 0.0},
        ),
        "random": (sample.in_range_sample, {"min_val": 0.0, "max_val": 1.0}),
        "true-random": (
            sample.in_range_sample,
            {"stratify": False, "min_val": 0.0, "max_val": 1.0},
        ),
        "bin-normal": (
            sample.bin_sample,
            {
                "stratify": False,
                "num_bins": 5,
                "curve": sample.norm,
                "mean": 0.5,
                "std": 0.25,
            },
        ),
        "mid-below-thresh": (sample.in_range_sample, {"min_val": 0.0, "max_val": 0.5}),
        "iqr": (sample.iqr_sample, {"thresh": 0.0}),
        "normal": (sample.normal_sample, {"thresh": 0.0}),
        "mid-normal": (
            sample.normal_sample,
            {"thresh": 0.0, "avg": 0.5, "stdev": 0.25},
        ),
        "mid-thresh": (sample.in_range_sample, {"min_val": 0.5, "max_val": 1.0}),
    }
