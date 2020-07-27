import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import retrain.sampling as sampling


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def get_label_path(self, img):
    return img[:-4].replace("images", "labels") + ".txt"


class ImageFolder:
    def __init__(self, src, num_classes, from_path=True):
        self.num_classes = num_classes

        if from_path:
            self.path = src
            self.imgs = self.get_images()
        else:
            self.imgs = src
        self.labels = self.get_labels()
        self.img_dict = self.make_img_dict()

    def get_images(self):
        extensions = (".jpg", ".png", ".gif", ".bmp")
        raw_imgs = sorted(glob.glob(f"{self.path}/**/*.*", recursive=True))
        raw_imgs = [file for file in raw_imgs if file[:-4].lower() in extensions]
        labeled_imgs = list()

        for img in raw_imgs:
            label_path = get_label_path(img)
            if os.path.exists(label_path):
                labeled_imgs.append(img)

        return labeled_imgs

    def get_labels(self):
        return [get_label_path(img) for img in self.imgs]

    def make_img_dict(self):
        return {
            img: self.get_classes(self.labels[i]) for i, img in enumerate(self.imgs)
        }

    def get_classes(self, label_path):
        """Get a list of classes from a Darknet label."""
        with open(label_path, "r") as label_file:
            labels = label_file.read().split("\n")
            classes = [int(lab.split(" ")[0]) for lab in labels if lab != ""]
            return [c for c in classes if c in range(self.num_classes)]

    def append(self, img_folder):
        self.num_classes = max(self.num_classes, img_folder)
        self.imgs += img_folder.imgs
        self.labels += img_folder.labels

    def to_dataset(self, **args):
        return ListDataset(self.imgs, **args)

    def split_img_set(self, prop_train, prop_valid, prop_test):
        """Split labeled images in an image folder into train, validation, and test sets.

        Assumes labels are consistent with the provided class list and labels are in
        YOLOv3 (Darknet) format.

        This is a modified implementation of iterative stratification based on
        Sechidis et. al 2011, as images may contain multiple labels/classes.
        """

        img_dict = self.make_img_dict()
        proportions = [prop_train, prop_valid, prop_test]
        img_lists = sampling.iterative_stratification(img_dict, proportions)
        return [
            ImageFolder(img_list, self.num_classes, from_path=False)
            for img_list in img_lists
        ]

    def augment(self, imgs_per_class, compose=True):
        pass
        # TODO: write this


class ListDataset(Dataset):
    def __init__(
        self,
        img_list,
        img_size=416,
        aug_compose=False,
        multiscale=True,
        normalized_labels=True,
    ):

        self.img_files = img_list

        self.label_files = [get_label_path(path) for path in self.img_files]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

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
