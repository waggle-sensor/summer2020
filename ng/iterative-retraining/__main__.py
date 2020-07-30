import argparse
import random
import retrain.utils as utils
from retrain.train import train
from retrain.dataloader import LabeledSet, ImageFolder
import retrain.benchmark as bench
import retrain.sampling as sample


def train_initial(init_folder, config):
    config["start_epoch"] = 1

    init_folder.train.augment(config["images_per_class"])
    init_folder.save_splits(config["output"])

    end_epoch = train(init_folder, config)
    return end_epoch


def get_num_classes(config):
    class_names = utils.load_classes(config["class_list"])
    return len(class_names)


def get_epoch_num(checkpoint):
    return int(checkpoint.split("_")[-1][:-4])


def label_sample_set(img_path):
    """Sample function of labeling an image given ground truth."""
    return img_path.split("-")[1].split("/")[0]


if __name__ == "__main__":
    random.seed("sage")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain_config", required=True, help="configuration for retraining"
    )
    parser.add_argument(
        "--reload_baseline",
        default=None,
        help="bypass initial training with a checkpoint",
    )
    opt = parser.parse_args()

    config = utils.parse_retrain_config(opt.retrain_config)

    classes = utils.load_classes(config["class_list"])
    num_classes = get_num_classes(config)

    init_images = LabeledSet(config["initial_set"], num_classes, prefix="init")
    test_prop = 1 - config["train_init"] - config["valid_init"]
    init_images.split_img_set(config["train_init"], config["valid_init"], test_prop)

    # Run initial training
    if opt.reload_baseline is None:
        init_end_epoch = train_initial(init_images, config)
        print(f"Initial training ended on epoch {init_end_epoch}")
        opt.reload_baseline = f"{config['checkpoints']}/init_ckpt_{init_end_epoch}.pth"
    else:
        init_end_epoch = get_epoch_num(opt.reload_baseline)

    # Sample
    all_samples = ImageFolder(config["sample_set"], img_size=config["img_size"])

    # Simulate a video feed at the edge
    batched_samples = all_samples.split_batch(config["sampling_batch"])

    config["train_split"] = config["train_sample"]
    config["valid_split"] = config["valid_sample"]

    sample_methods = {
        "median-thresh": sample.median_thresh_sample,
        "iqr": sample.iqr_sample,
        "normal": sample.normal_sample,
    }

    seen_images = init_images

    for name, func in sample_methods.items():
        last_epoch = init_end_epoch
        for i, sample_folder in enumerate(batched_samples):

            # TODO make this applicable for multiple labels
            sample_folder.label(classes, label_sample_set)
            sample_labeled = LabeledSet(
                sample_folder.imgs,
                num_classes,
                img_size=config["img_size"],
                from_path=False,
            )

            # Benchmark data at the edge
            bench_file = bench.benchmark_avg(
                sample_labeled, name, 1, last_epoch, config["conf_check_num"], config
            )

            # Create samples from the benchmark
            results, _ = bench.load_data(bench_file, by_actual=False)
            retrain_list = sample.create_sample(
                results, name, config["bandwidth"], func, thresh=0.0
            )

            retrain_files = [data["file"] for data in retrain_list]

            # Receive raw sampled data in the cloud, with ground truth annotations
            retrain_obj = LabeledSet(
                retrain_files, num_classes, prefix=f"{name}{i}", from_path=False
            )

            test_prop = 1 - config["train_sample"] - config["valid_sample"]
            retrain_obj.split_img_set(
                config["train_sample"], config["valid_sample"], test_prop
            )

            seen_images += retrain_obj

            for name in ("train", "valid", "test"):
                # Calculate proportion of old examples needed
                number_desired = (1 / config["retrain_new"] - 1) * len(
                    getattr(retrain_obj, name)
                )
                print(number_desired)
                extra_images = getattr(seen_images, name).split_batch(
                    round(number_desired)
                )[0]
                orig_set = getattr(retrain_obj, name)
                orig_set += extra_images

            seen_images += retrain_obj
            
            retrain_obj.train.augment(config["images_per_class"])
            retrain_obj.save_splits(config["output"])

            config["start_epoch"] = last_epoch + 1
            checkpoint = utils.find_checkpoint(config, name, last_epoch)
            last_epoch = train(retrain_obj, config, checkpoint)
