import argparse
import random
import os

import retrain.utils as utils
from retrain.train import train_initial, get_free_gpus
from retrain.dataloader import LabeledSet, ImageFolder, split_set
import retrain.sampling as sample
from retrain import retrain


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

    retrain.parallel_retrain(config, batched_samples, init_end_epoch, init_images)
