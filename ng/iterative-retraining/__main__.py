import argparse
import random
import retrain.utils as utils
from retrain.train import train
from retrain.dataloader import ImageFolder
import retrain.benchmark as bench
import retrain.sampling as sample


def train_initial(config):
    config["train_split"] = config["train_init"]
    config["valid_split"] = config["valid_init"]
    config["start_epoch"] = 1
    config["prefix"] = "init"

    img_folder = ImageFolder(config["initial_set"], get_num_classes(config))
    end_epoch = train(img_folder, config, model_config)
    return img_folder, end_epoch


def get_num_classes(config):
    class_names = utils.load_classes(config["class_list"])
    return len(class_names)


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
    model_config = utils.parse_model_config(config["model_config"])
    num_classes = get_num_classes(config)

    # Run initial training
    if opt.reload_baseline is None:
        initial_folder, init_end_epoch = train_initial(config)
        seen_images = initial_folder
        print(f"Initial training ended on epoch {init_end_epoch}")
        opt.reload_baseline = f"{config['checkpoints']}/init_ckpt_{init_end_epoch}.pth"
    else:
        seen_images = ImageFolder(config["initial_set"], num_classes)
        init_end_epoch = int(opt.reload_baseline.split("_")[-1][:-4])

    # Sample
    all_samples = ImageFolder(config["sample_set"], num_classes)
    batched_samples = all_samples.split_batch(config["sampling_batch"])

    config["train_split"] = config["train_sample"]
    config["valid_split"] = config["valid_sample"]

    sample_methods = {
        "median-thresh": sample.median_thresh_sample,
        "iqr": sample.iqr_sample,
        "normal": sample.normal_sample,
    }

    for name, func in sample_methods:
        last_epoch = init_end_epoch
        for i, sample in enumerate(batched_samples):
            bench_file = bench.benchmark_avg(
                sample,
                name,
                1,
                last_epoch,
                config["conf_check_num"],
                config,
                model_config,
            )

            results, _ = bench.load_data(bench_file, by_actual=False)

            retrain_list = sample.create_sample(
                sample, results, config["bandwidth"], name, func, thresh=0.0
            )

            retrain_obj = ImageFolder(retrain_list, num_classes, from_path=False)
            num_sample_imgs = round(config["images_per_class"] * config["retrain_new"])
            num_old_imgs = config["images_per_class"] - num_sample_imgs

            retrain_obj.augment(num_sample_imgs)

            old_examples = list()
            for _, imgs in seen_images.class_dict().items():
                old_examples += imgs[:num_old_imgs]

            seen_images.append(retrain_obj)
            retrain_obj.append_list(old_examples)

            config["prefix"] = name + str(i)
            config["start_epoch"] = last_epoch + 1
            last_epoch = train(retrain_obj, config, model_config)
