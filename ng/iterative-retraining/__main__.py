import argparse
import random
import os
import copy
from torch import cuda

import retrain.utils as utils
from retrain.train import train
from retrain.dataloader import LabeledSet, ImageFolder
import analysis.benchmark as bench
import retrain.sampling as sample


def label_sample_set(img_path):
    """Sample function to label an image path with its ground truth with a list of labels.

    This function is customizable (e.g. including a GUI to annotate) depending on your needs.
    It should return a list of tuples, with each tuple representing a label with the values
    (class_label, bounding_box_x_center, bb_y_center, bb_width, bb_height)
    These coordinates should also be normalized according to the image's width and height.
    """
    class_letter = img_path.split("-")[1].split("/")[0]
    label = (class_letter, 0.5, 0.5, 1.0, 1.0)
    return [label]


def get_sample_methods():
    return {
        "median-thresh": (sample.median_thresh_sample, {"thresh": 0.0}),
        "median-below-thresh": (sample.median_below_thresh_sample, {"thresh": 0.0}),
        "iqr": (sample.iqr_sample, {"thresh": 0.0}),
        "normal": (sample.normal_sample, {"thresh": 0.0}),
        "mid-normal": (
            sample.normal_sample,
            {"thresh": 0.0, "avg": 0.5, "stdev": 0.25},
        ),
        "mid-thresh": (sample.in_range_sample, {"min_val": 0.5, "max_val": 1.0}),
        "mid-below-thresh": (sample.in_range_sample, {"min_val": 0.0, "max_val": 0.5}),
        "bin-quintile": (
            sample.bin_sample,
            {"stratify": False, "num_bins": 5, "curve": sample.const, "thresh": 0.0},
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
        "random": (sample.in_range_sample, {"min_val": 0.0, "max_val": 1.0}),
    }


def split_set(labeled_set, output, train_prop, valid_prop, save=True, sample_dir=None):
    print(f"Getting splits for {labeled_set.prefix}")

    if labeled_set.load_splits(output):
        train_imgs = sum(
            round(train_prop * len(v)) for v in labeled_set.group_by_class().values()
        )
        train_len = len(labeled_set.train)

        # Case where we use load splits from the mixed set of sampled
        # and known images
        if sample_dir is not None:
            train_imgs = (
                len(labeled_set.valid) + train_len + len(labeled_set.test)
            ) * train_prop

        if abs(train_len - train_imgs) <= 10:
            print("Previous splits found and validated")
            return False
        else:
            print("Train list mismatch found... Ignoring....")

    print("Generating new splits")
    labeled_set.split_img_set(train_prop, valid_prop)
    if save:
        labeled_set.save_splits(output)
    return True


def train_initial(init_folder, config):
    config["start_epoch"] = 1

    init_folder.train.augment(config["images_per_class"])
    end_epoch = train(init_folder, config)
    return end_epoch


def sample_retrain(name, batches, config, last_epoch, seen_images, sample_func, kwargs):
    classes = utils.load_classes(config["class_list"])
    seen_images = copy.deepcopy(seen_images)
    for i, sample_folder in enumerate(batches):
        sample_folder.label(classes, label_sample_set)
        sample_labeled = LabeledSet(
            sample_folder.imgs, len(classes), config["img_size"],
        )

        sample_filename = f"{config['output']}/{name}{i}_sample_{last_epoch}.txt"
        if os.path.exists(sample_filename):
            print("Loading existing samples")
            retrain_files = open(sample_filename, "r").read().split("\n")

        else:
            # Benchmark data at the edge
            bench_file = (
                f"{config['output']}/{name}{i}_benchmark_avg_1_{last_epoch}.csv"
            )

            if not os.path.exists(bench_file):
                results_df = bench.benchmark_avg(
                    sample_labeled,
                    name,
                    1,
                    last_epoch,
                    config["conf_check_num"],
                    config,
                )

                bench.save_results(results_df, bench_file)

            # Create samples from the benchmark
            results, _ = bench.load_data(bench_file, by_actual=False)

            print(f"===== {name} ======")
            retrain_files = sample.create_sample(
                results, config["bandwidth"], func, **kwargs
            )

            with open(sample_filename, "w+") as out:
                out.write("\n".join(retrain_files))

        # Receive raw sampled data in the cloud
        # This process simulates manually labeling/verifying all inferences
        retrain_obj = LabeledSet(
            retrain_files, len(classes), config["img_size"], prefix=f"{name}{i}"
        )

        new_splits = split_set(
            retrain_obj,
            config["output"],
            config["train_sample"],
            config["valid_sample"],
            save=False,
            sample_dir=config["sample_set"],
        )

        if new_splits:
            # If reloaded, splits have old images already incorporated
            for set_name in retrain_obj.sets:
                # Calculate proportion of old examples needed
                number_desired = (1 / config["retrain_new"] - 1) * len(
                    getattr(retrain_obj, set_name)
                )
                if round(number_desired) == 0:
                    continue
                print(set_name, number_desired)
                extra_images = getattr(seen_images, set_name).split_batch(
                    round(number_desired)
                )[0]
                orig_set = getattr(retrain_obj, set_name)
                orig_set += extra_images

        seen_images += retrain_obj

        retrain_obj.save_splits(config["output"])
        retrain_obj.train.augment(config["images_per_class"])

        config["start_epoch"] = last_epoch + 1
        checkpoint = utils.find_checkpoint(config, name, last_epoch)
        last_epoch = train(retrain_obj, config, checkpoint)


def get_cuda_devices():
    torch.cuda.device_count()


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

    for name, (func, kwargs) in sample_methods.items():
        sample_retrain(
            name, batched_samples, config, init_end_epoch, init_images, func, kwargs
        )
