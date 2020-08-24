import os
import copy
import traceback

import multiprocessing as mp
import multiprocessing.pool as pool

import analysis.benchmark as bench
import userdefs
from retrain import sampling as sample
from retrain import utils, train
from retrain.dataloader import LabeledSet, split_set


def sample_retrain(
    name,
    batches,
    config,
    last_epoch,
    seen_images,
    label_func,
    sample_func,
    kwargs,
    device=None,
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
        last_epoch = train.train(retrain_obj, config, checkpoint, device=device)


class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class ExceptionLogger(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception:
            with open("error.err", "a+") as out:
                out.write(traceback.format_exc())
            raise Exception(traceback.format_exc())
        return result


class NoDaemonPool(pool.Pool):
    Process = NoDaemonProcess

    def starmap_async(self, func, iterable, **kwargs):
        return pool.Pool.starmap_async(self, ExceptionLogger(func), iterable, **kwargs)


def parallel_retrain(config, batched_samples, init_end_epoch, init_images):
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    mp.set_start_method("spawn")
    sample_methods = userdefs.get_sample_methods()
    free_gpus = train.get_free_gpus(config)
    with NoDaemonPool(max(1, len(free_gpus)) * 10) as pool:
        grouped_args = list()
        for i, (name, (func, kwargs)) in enumerate(sample_methods.items()):
            device = free_gpus[i % len(free_gpus)]
            method_args = (
                name,
                batched_samples,
                config,
                init_end_epoch,
                init_images,
                userdefs.label_sample_set,
                func,
                kwargs,
                device,
            )
            grouped_args.append(method_args)

        pool.starmap_async(sample_retrain, grouped_args)
        pool.close()
        pool.join()
