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
from retrain.augment import Augmenter
from retrain.utils import get_label_path


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


class ImageFolder(Dataset):
    def __init__(self, src, num_classes, img_size=416, prefix=str(), from_path=True):
        self.num_classes = num_classes

        if from_path:
            self.path = src
            self.imgs = self.get_images()
        else:
            self.imgs = src

        self.labels = self.get_labels()
        self.img_dict = self.make_img_dict()
        self.prefix = prefix
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = list(self.imgs)[index % len(self.imgs)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.imgs)

    def get_images(self):
        extensions = (".jpg", ".png", ".gif", ".bmp")
        raw_imgs = sorted(glob.glob(f"{self.path}/images/**/*.*", recursive=True))
        raw_imgs = {file for file in raw_imgs if file[-4:].lower() in extensions}
        labeled_imgs = set()

        for img in raw_imgs:
            label_path = get_label_path(img)
            if os.path.exists(label_path):
                labeled_imgs.add(img)

        return labeled_imgs

    def get_labels(self):
        return {get_label_path(img) for img in self.imgs}

    def make_img_dict(self):
        img_dict = dict()
        for img in self.imgs:
            classes = self.get_classes(get_label_path(img))
            if len(classes) != 0:
                img_dict[img] = classes
        return img_dict

    def get_classes(self, label_path):
        """Get a list of classes from a Darknet label."""
        with open(label_path, "r") as label_file:
            labels = label_file.read().split("\n")
            classes = [int(lab.split(" ")[0]) for lab in labels if lab != ""]
            return [c for c in classes if c in range(self.num_classes)]

    def __iadd__(self, img_folder):
        self.num_classes = max(self.num_classes, img_folder)

        for attr in ("imgs", "labels", "train", "test", "valid"):
            self_attr = getattr(self, attr, None)
            other_attr = get_attr(img_folder, attr, None)
            if other_attr is not None:
                if self_attr is not None:
                    self_attr += other_attr
                else:
                    setattr(self, attr, other_attr)

    # def append(self, img_folder, prop_new=1.0):
    #     """Append a certain proportion of an image folder to the current one."""
    #     for name in ("train", "valid", "test"):
    #         folder = getattr(img_folder, name)
    #         new_imgs = getattr(folder, "imgs")[:prop_new * len(new_imgs)]
    #         getattr(self, name) += ImageFolder(new_imgs, img_folder.num_classes, from_path=False)

    def to_dataset(self, **args):
        return ListDataset(self.imgs, **args)

    def save_splits(self, folder):
        for i, name in enumerate(("train", "valid", "test")):
            img_set = getattr(self, name, None)
            if img_set is not None:
                filename = f"{folder}/{self.prefix}_{name}.txt"
                img_set.save_img_list(filename)

    def save_img_list(self, output):
        """Save list of images (for splits) as a text file."""
        with open(output, "w+") as out:
            out.write("\n".join(self.imgs))

    def augment(self, imgs_per_class, compose=True):
        aug = Augmenter(self)
        aug.augment(imgs_per_class, compose)
        self.img_dict = self.make_img_dict()

    def split_img_set(self, prop_train, prop_valid, prop_test):
        """Split labeled images in an image folder into train, validation, and test sets.

        Assumes labels are consistent with the provided class list and labels are in
        YOLOv3 (Darknet) format.

        This relies on a modified implementation of iterative stratification from
        Sechidis et. al 2011, as images may contain multiple labels/classes.
        """
        sets = ("train", "valid", "test")

        img_dict = self.make_img_dict()
        proportions = [prop_train, prop_valid, prop_test]
        img_lists = sampling.iterative_stratification(img_dict, proportions)
        splits = [
            ImageFolder(
                img_list,
                self.num_classes,
                prefix=f"{self.prefix}_{sets[i]}",
                from_path=False,
            )
            for i, img_list in enumerate(img_lists)
        ]

        self.train, self.valid, self.test = splits
        return splits

    def split_batch(self, batch_size):
        """Split an ImageFolder into multiple batches of a finite size.

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

        return [
            ImageFolder(
                img_list, self.num_classes, prefix=f"{self.prefix}{i}", from_path=False,
            )
            for i, img_list in enumerate(splits)
        ]

    def group_by_class(self):
        class_dict = dict()
        for img, class_list in self.make_img_dict():
            for c in class_list:
                if c not in class_dict.keys():
                    class_dict[c] = set()
                class_dict[c].add(img)
        return class_dict


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
