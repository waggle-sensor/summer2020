import argparse
import random
import retrain.utils as utils
from retrain.train import train


def train_initial(config):
    config["train_split"] = config["train_init"]
    config["valid_split"] = config["valid_init"]
    config["start_epoch"] = 1
    config["prefix"] = "init"

    end_epoch = train(config["initial_set"], config, model_config)
    return end_epoch


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

    # Run initial training
    if opt.reload_baseline is None:
        end_epoch = train_initial(config)
        print(f"Initial training ended on epoch {end_epoch}")
        opt.reload_baseline = f"{config['checkpoints']}/init_ckpt_{end_epoch}.pth"

    # Sample

    config["conf_check_num"]

    # Retrain for each sample
    # config["train_split"] = config["train_sample"]
    # config["valid_split"] = config["valid_sample"]
    # config["start_epoch"] =
    # train(config["initial_set"], config, model_config)
