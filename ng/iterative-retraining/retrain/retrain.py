"""
Module with functions for the retraining and sampling steps in the pipeline
"""

import os
import copy

import userdefs
from retrain import sampling as sample
from retrain import utils, train
from retrain.dataloader import LabeledSet
import yolov3.utils as yoloutils
from yolov3 import parallelize
import analysis.benchmark as bench
import analysis.results as resloader


def benchmark_sample(sample_method, imgs, config, batch_num, last_epoch):
    """Simulate benchmarking and sampling at the edge, returning a list of samples."""
    name, (sample_func, kwargs) = sample_method
    bench_file = (
        f"{config['output']}/{name}{batch_num}_benchmark_avg_1_{last_epoch}.csv"
    )

    if not os.path.exists(bench_file):
        results_df = bench.benchmark_avg(
            imgs, name, 1, last_epoch, config["conf_check_num"], config,
        )

        bench.save_results(results_df, bench_file)

    # Create samples from the benchmark
    results, _ = resloader.load_data(bench_file, by_actual=False)

    print(f"===== {name} ======")
    sample_files = sample.create_sample(
        results, config["bandwidth"], sample_func, **kwargs
    )

    return sample_files


def sample_retrain(
    sample_method, batches, config, last_epoch, seen_images, label_func, device=None,
):
    """Run the sampling and retraining pipeline for a particular sampling function."""
    name, _ = sample_method
    classes = utils.load_classes(config["class_list"])
    seen_images = copy.deepcopy(seen_images)
    for i, sample_folder in enumerate(batches):
        sample_folder.label(classes, label_func)
        sample_labeled = LabeledSet(
            sample_folder.imgs, len(classes), config["img_size"],
        )

        sample_filename = f"{config['output']}/{name}{i}_sample_{last_epoch}.txt"
        if os.path.exists(sample_filename):
            print("Loading existing samples")
            retrain_files = open(sample_filename, "r").read().split("\n")

        else:
            retrain_files = benchmark_sample(
                sample_method, sample_labeled, config, i, last_epoch
            )
            with open(sample_filename, "w+") as out:
                out.write("\n".join(retrain_files))

        # Receive raw sampled data in the cloud
        # This process simulates manually labeling/verifying all inferences
        retrain_obj = LabeledSet(
            retrain_files, len(classes), config["img_size"], prefix=f"{name}{i}"
        )

        new_splits_made = retrain_obj.load_or_split(
            config["output"],
            config["train_sample"],
            config["valid_sample"],
            save=False,
            sample_dir=config["sample_set"],
        )

        if new_splits_made:
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
        last_epoch = train.train(retrain_obj, config, checkpoint, device=device)


def retrain(config, sample_methods, sample_batches, base_epoch, init_imgs):
    """Sample images and retrain for all sample methods given."""
    free_gpus = yoloutils.get_free_gpus(yoloutils.get_memory_needed(config))

    grouped_args = list()
    for i, sample_method in enumerate(sample_methods.items()):
        device = free_gpus[i % len(free_gpus)]
        method_args = (
            sample_method,
            sample_batches,
            config,
            base_epoch,
            init_imgs,
            userdefs.label_sample_set,
            device,
        )
        grouped_args.append(method_args)

        if not config["parallel"]:
            sample_retrain(*method_args)

    if config["parallel"]:
        parallelize.run_parallel(sample_retrain, grouped_args)
