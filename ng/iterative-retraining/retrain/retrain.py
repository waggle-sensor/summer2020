from retrain import sampling as sample
from retrain import utils
from retrain import train
import os
import copy
from retrain.dataloader import LabeledSet, split_set
import analysis.benchmark as bench
from multiprocessing import Pool


def sample_retrain(
    name, batches, config, last_epoch, seen_images, label_func, sample_func, kwargs
):
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
                results, config["bandwidth"], sample_func, **kwargs
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
        last_epoch = train.train(retrain_obj, config, checkpoint)


def parallel_retrain(
    sample_methods, config, batched_samples, init_end_epoch, init_images
):
    with Pool(len(train.get_free_gpus(config))) as pool:
        for name, (func, kwargs) in sample_methods.items():
            pool.apply_async(
                sample_retrain,
                (
                    name,
                    batched_samples,
                    config,
                    init_end_epoch,
                    init_images,
                    func,
                    kwargs,
                ),
            )
