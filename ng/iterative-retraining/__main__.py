import argparse
import random
import os

import retrain.utils as utils
from retrain.train import train_initial, get_free_gpus
from retrain.dataloader import LabeledSet, ImageFolder, split_set
import retrain.sampling as sample
from retrain import retrain


def label_sample_set(img_path):
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
    return {
        "median-below-thresh": (sample.median_below_thresh_sample, {"thresh": 0.0}),
        "median-thresh": (sample.median_thresh_sample, {"thresh": 0.0}),
        "bin-quintile": (
            sample.bin_sample,
            {"stratify": False, "num_bins": 5, "curve": sample.const, "thresh": 0.0},
        ),
        "random": (sample.in_range_sample, {"min_val": 0.0, "max_val": 1.0}),
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


if __name__ == "__main__":
    random.seed("sage")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configuration for retraining")
    parser.add_argument(
        "--reload_baseline",
        default=None,
        help="bypass initial training with a checkpoint",
    )
    opt = parser.parse_args()
    config = utils.parse_retrain_config(opt.config)

    classes = utils.load_classes(config["class_list"])

    init_images = LabeledSet(
        config["initial_set"], len(classes), config["img_size"], prefix="init"
    )
    split_set(init_images, config["output"], config["train_init"], config["valid_init"])

    # Run initial training
    if opt.reload_baseline is None:
        init_end_epoch = train_initial(init_images, config)
        print(f"Initial training ended on epoch {init_end_epoch}")
    else:
        init_end_epoch = utils.get_epoch(opt.reload_baseline)

    # Sample
    all_samples = ImageFolder(config["sample_set"], config["img_size"], prefix="sample")

    # Simulate a video feed at the edge
    batched_samples = all_samples.get_batch_splits(
        config["sampling_batch"], config["output"]
    )

    # Remove the last batch if incomplete
    if len(batched_samples[-1]) != len(batched_samples[0]):
        batched_samples = batched_samples[:-1]

    sample_methods = get_sample_methods()

    if len(get_free_gpus(config)) <= 1:
        for name, (func, kwargs) in sample_methods.items():
            retrain.sample_retrain(
                name,
                batched_samples,
                config,
                init_end_epoch,
                init_images,
                label_sample_set,
                func,
                kwargs,
            )
    else:
        retrain.parallel_retrain(
            sample_methods, config, batched_samples, init_end_epoch, init_images
        )
