"""
Module containing data objects to extract and manipulate images and labels for training,
sampling, and evaluation.

Some code derived from loader module in the YOLOv3 Docker plugin.
"""

import os
import math
import random

import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from retrain import sampling
from retrain.augment import Augmenter
from retrain.utils import get_label_path, get_lines
from yolov3.utils import pad_to_square, resize


class ImageFolder(Dataset):
    """Dataset representation of a (potentially unlabeled) set of images.

    Iterable provides an image path and an image tensor of the specified size.
    """

    def __init__(self, src, img_size, prefix=str()):
        """
        Parameters:
            src (str): source of the image folder. Can be a list or set of image paths,
                a text file of image paths, or a Darknet-labeled folder
            img_size (int): square resolution to pad or downsize images to
            prefix (str): string to represent the image folder. used in output filenames
        """
        if isinstance(src, (list, set)):
            self.imgs = set(src)
        elif ".txt" in src:
            if os.path.isfile(src):
                self.imgs = get_lines(src)
            else:
                print(f"{src} is an invalid file. Ignoring...")
        elif os.path.isdir(src):
            self.imgs = get_images(src)
        else:
            raise TypeError("ImageFolder source must be file list or folder")

        self.prefix = prefix
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = list(self.imgs)[index % len(self.imgs)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.imgs)

    def __iadd__(self, img_folder):
        # Take the union of the two image sets
        self.imgs.update(img_folder.imgs)

    def to_dataset(self, **args):
        return ListDataset(list(self.imgs), img_size=self.img_size, **args)

    def save_img_list(self, output):
        """Save list of images (for splits) as a text file."""
        with open(output, "w+") as out:
            out.write("\n".join(self.imgs))

    def get_batch_splits(self, batch_size, output):
        """Output text files lists of random image batches of the specified size.

        Does not stratify by class, and the last batch may have fewer images than desired.
        """
        batches = math.ceil(len(self) / batch_size)
        batch_files = [f"{output}/{self.prefix}{i}.txt" for i in range(batches)]
        if all(os.path.exists(path) for path in batch_files):
            batch_splits = [get_lines(path) for path in batch_files]
            print("Previous batch splits found")
        else:
            batch_splits = self.split_batch(batch_size)
            for i, path in enumerate(batch_files):
                with open(path, "w+") as out:
                    out.write("\n".join(batch_splits[i]))
        return self.convert_splits(batch_splits)

    def convert_splits(self, splits):
        """Convert image lists into a list of ImageFolders."""
        return [
            ImageFolder(img_list, self.img_size, prefix=f"{self.prefix}{i}")
            for i, img_list in enumerate(splits)
        ]

    def split_batch(self, batch_size):
        """Split an ImageFolder into multiple batches of a finite size.

        This function is meant to be used for simulations at the inferencing/sampling stage.
        """
        random.shuffle(list(self.imgs))
        splits = list()
        for i in range(0, len(self), batch_size):
            upper_bound = min(len(self), i + batch_size)
            splits.append(set(list(self.imgs)[i:upper_bound]))
        return splits

    def label(self, classes, ground_truth_func):
        """Label images in the folder in accordance to a provided ground truth function.

        Ground truth function should return a list of labels, each in a tuple format with
        class name and normalized bounding box dimensions in Darknet format.

        The input class list should be ordered and consistent with the ground truth labels.
        """
        for img in self.imgs:
            labels = ground_truth_func(img, classes)
            if len(labels) == 0:
                return
            text_label = open(get_label_path(img), "w+")
            for i, (class_display_name, x_cent, y_cent, w, h) in enumerate(labels):
                class_num = classes.index(class_display_name)
                if i > 0:
                    text_label.write("\n")
                text_label.write(f"{class_num} {x_cent} {y_cent} {w} {h}")
            text_label.close()


def get_images(path):
    """Recursively extract a list of images from a path."""
    extensions = (".jpg", ".png", ".gif", ".bmp")
    raw_imgs = sorted(glob.glob(f"{path}/images/**/*.*", recursive=True))
    imgs = {file for file in raw_imgs if file[-4:].lower() in extensions}

    return imgs


class LabeledSet(ImageFolder):
    """A Dataset object inheriting from ImageFolder containing labeled images only."""

    def __init__(self, src, num_classes, img_size=416, prefix=str(), **args):
        super().__init__(src, img_size, prefix, **args)
        self.num_classes = num_classes

        self.filter_images()
        self.labels = self.get_labels()
        self.sets = ("train", "valid", "test")

    def filter_images(self):
        """Remove non-labeled images from the folder."""
        labeled_imgs = set()

        for img in self.imgs:
            label_path = get_label_path(img)
            if os.path.exists(label_path):
                labeled_imgs.add(img)
        self.imgs = labeled_imgs

    def get_classes(self, label_path):
        """Get a list of classes from a Darknet label."""
        labels = get_lines(label_path)
        classes = [int(lab.split(" ")[0]) for lab in labels if lab != ""]
        return [c for c in classes if c in range(self.num_classes)]

    def get_labels(self):
        """Get a set of labels corresponding to images in the folder."""
        return {get_label_path(img) for img in self.imgs}

    def make_img_dict(self):
        """Get a dictionary of image paths and their corresponding classes."""
        img_dict = dict()
        for img in self.imgs:
            classes = self.get_classes(get_label_path(img))
            if len(classes) != 0:
                img_dict[img] = classes
        return img_dict

    def group_by_class(self):
        """Get a dictionary of classes for the folder's images, with a list of corresponding
        images as labels."""
        class_dict = dict()
        for img, class_list in self.make_img_dict().items():
            for c in class_list:
                if c not in class_dict.keys():
                    class_dict[c] = set()
                class_dict[c].add(img)
        return class_dict

    def __iadd__(self, labeled_set):
        """Add two labeled sets, creating a union of their train, test, and validation sets."""
        super().__iadd__(labeled_set)
        self.num_classes = max(self.num_classes, labeled_set.num_classes)

        for attr in ("labels", "train", "test", "valid"):
            self_attr = getattr(self, attr, None)
            other_attr = getattr(labeled_set, attr, None)
            if other_attr is not None:
                if self_attr is not None:
                    if isinstance(self_attr, LabeledSet):
                        self_attr += other_attr
                    else:
                        self_attr.update(other_attr)
                else:
                    setattr(self, attr, other_attr)

        return self

    def load_or_split(self, output, train_prop, valid_prop, save=True, sample_dir=None):
        """Split a LabeledSet into test, train, and validation sets if and only if existing
        splits do not exist as text files.

        Returns a boolean value indicating if new splits were generated.
        """
        print(f"Getting splits for {self.prefix}")

        if self.load_splits(output):
            train_imgs = sum(
                round(train_prop * len(v)) for v in self.group_by_class().values()
            )
            train_len = len(self.train)

            # Case where we use load splits from the mixed set of sampled
            # and known images
            if sample_dir is not None:
                train_imgs = (len(self.valid) + train_len + len(self.test)) * train_prop

            if abs(train_len - train_imgs) <= 10:
                print("Previous splits found and validated")
                return False

            print("Train list mismatch found... Ignoring....")

        print("Generating new splits")
        self.split_img_set(train_prop, valid_prop)
        if save:
            self.save_splits(output)
        return True

    def save_splits(self, folder):
        """Save the train, test, and validation splits of the current folder into text files."""
        for name in self.sets:
            img_set = getattr(self, name, None)
            if img_set is not None:
                filename = f"{folder}/{self.prefix}_{name}.txt"
                img_set.save_img_list(filename)

    def load_splits(self, folder):
        """Load previously-generated text file splits into the current folder if found.

        Returns a boolean value indicating if the operation was successful.
        """
        split_paths = [f"{folder}/{self.prefix}_{name}.txt" for name in self.sets]
        if all(os.path.exists(path) for path in split_paths):
            file_lists = [get_lines(path) for path in split_paths]
            labeled_sets = self.convert_splits(file_lists)
            for name, split_set in zip(self.sets, labeled_sets):
                setattr(self, name, split_set)
            return True
        return False

    def split_img_set(self, prop_train, prop_valid):
        """Split labeled images in an image folder into train, validation, and test sets.

        Assumes labels are consistent with the provided class list and labels are in
        YOLOv3 (Darknet) format.

        This relies on a modified implementation of iterative stratification from
        Sechidis et. al 2011, as images may contain multiple labels/classes.
        """

        img_dict = self.make_img_dict()

        prop_test = 1 - prop_train - prop_valid
        proportions = [prop_train, prop_valid, prop_test]

        img_lists = sampling.iterative_stratification(img_dict, proportions)

        split_sets = self.convert_splits(img_lists)
        self.train, self.valid, self.test = split_sets

        return split_sets

    def convert_splits(self, splits):
        return [
            LabeledSet(
                set(img_list),
                self.num_classes,
                self.img_size,
                prefix=f"{self.prefix}{i}",
            )
            for i, img_list in enumerate(splits)
        ]

    def split_batch(self, batch_size):
        """Split an LabeledSet into multiple batches of a finite size.

        This function is meant to be used for simulations at the inferencing/sampling stage.
        It does not support stratified splitting at the train, validation, and
        test set levels.
        """
        full_batches = int(len(self) / batch_size)
        imgs_remaining = len(self) - full_batches * batch_size
        if full_batches == 0:
            raise ValueError("Batch size exceeds folder size")

        weights = [batch_size] * full_batches
        weights.append(imgs_remaining)
        normal_weights = [weight / sum(weights) for weight in weights]

        splits = sampling.iterative_stratification(self.make_img_dict(), normal_weights)

        return self.convert_splits(splits)

    def augment(self, imgs_per_class, compose=True):
        """Augment the images in the current folder by a specified factor."""
        aug = Augmenter(self)
        aug.augment(imgs_per_class, compose)
        self.img_dict = self.make_img_dict()


class ListDataset(Dataset):
    """Final wrapper Dataset object for loading images into the Darknet model."""

    def __init__(
        self, img_list, img_size=416, multiscale=True, normalized_labels=True,
    ):

        self.img_files = img_list

        self.label_files = [get_label_path(path) for path in self.img_files]

        self.img_size = img_size
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        img = None

        i = 0

        # Prevent corrupted augmentation images
        while img is None and i < len(self.img_files):
            try:
                img_path = self.img_files[(index + i) % len(self.img_files)].rstrip()
                img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
            except OSError:
                if os.path.exists(img_path):
                    os.remove(img_path)
                i += 1

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets

        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
