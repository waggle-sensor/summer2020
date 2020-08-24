from __future__ import division

import os
import time
import datetime

import torch
from torch import cuda
from torch.autograd import Variable

from terminaltables import AsciiTable

from yolov3 import evaluate
from yolov3.logger import Logger
import yolov3.utils as yoloutils
from yolov3 import models

import retrain.utils as utils


def train_initial(init_folder, config):
    config["start_epoch"] = 1

    init_folder.train.augment(config["images_per_class"])
    end_epoch = train(init_folder, config)
    return end_epoch


def train(img_folder, opt, load_weights=None, device=None):
    """Trains a given image set, with an early stop.

    Precondition: img_folder has been split into train, test, and validation sets.
    """
    os.makedirs(opt["checkpoints"], exist_ok=True)
    os.makedirs(opt["output"], exist_ok=True)

    model = models.get_train_model(opt)

    free_gpus = get_free_gpus(opt, model)

    if device is None:
        device_str = "cuda" if len(free_gpus) != 0 else "cpu"
    else:
        if device not in free_gpus:
            device = free_gpus[0]
        device_str = f"cuda:{device}"

    device = torch.device(device_str)

    print(f"Using {device_str} for training")
    yoloutils.clear_vram()

    # Initiate model
    model.apply(yoloutils.weights_init_normal)

    if load_weights is not None:
        model.load_state_dict(torch.load(load_weights, map_location="cpu"))

    model.to(device)
    model.device = device

    class_names = utils.load_classes(opt["class_list"])

    # Get dataloader
    dataset = img_folder.train.to_dataset(multiscale=bool(opt["multiscale"]),)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt["batch_size"],
        shuffle=True,
        num_workers=opt["n_cpu"],
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # Limit logging rate of batch metrics
    logger = Logger(opt["log"], img_folder.prefix)
    log_freq = min(
        len(dataloader), opt["logs_per_epoch"] if "logs_per_epoch" in opt.keys() else 50
    )
    log_interval = int(len(dataloader) / log_freq)

    successive_stops = 0
    prev_strip_loss = float("inf")

    end_epoch = opt["start_epoch"] + opt["max_epochs"]

    last_epoch = opt["start_epoch"]

    for epoch in range(opt["start_epoch"], end_epoch):
        last_epoch = epoch
        model.train()
        start_time = time.time()

        ckpt_path = f"{opt['checkpoints']}/{img_folder.prefix}_ckpt_{epoch}.pth"

        if not os.path.exists(ckpt_path):
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                batches_done = len(dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device))

                loss, _ = model(imgs, targets)

                loss.backward()

                if batches_done % opt["gradient_accumulations"]:
                    # Accumulates gradient before each step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt["clip"])
                    optimizer.step()
                    optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                    epoch,
                    end_epoch,
                    batch_i,
                    len(dataloader),
                )

                metric_table = [
                    [
                        "Metrics",
                        *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))],
                    ]
                ]

                # Log metrics at each YOLO layer
                for metric in metrics:
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [
                        formats[metric] % yolo.metrics.get(metric, 0)
                        for yolo in model.yolo_layers
                    ]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.item())]

                    if batch_i % log_interval == 0:
                        logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(
                    seconds=epoch_batches_left
                    * (time.time() - start_time)
                    / (batch_i + 1)
                )
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

                model.seen += imgs.size(0)

            if epoch % opt["checkpoint_interval"] == 0:
                if hasattr(model, "module"):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(
                    state_dict,
                    f"{opt['checkpoints']}/{img_folder.prefix}_ckpt_{epoch}.pth",
                )

        else:
            model.load_state_dict(torch.load(ckpt_path))

        # Use UP criteria for early stop
        if bool(opt["early_stop"]) and (
            epoch == opt["start_epoch"] or epoch % opt["strip_len"] == 0
        ):
            print(f"\n---Evaluating validation set on epoch {epoch} for early stop---")

            valid_results = evaluate.get_results(
                model, img_folder.valid, opt, class_names, logger, epoch
            )

            if valid_results["val_loss"] > prev_strip_loss:
                successive_stops += 1
            else:
                successive_stops = 0
            print(f"Previous loss: {prev_strip_loss}")
            print(f"Current loss: {valid_results['val_loss']}")

            prev_strip_loss = valid_results["val_loss"]

            if successive_stops == opt["successions"]:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % opt["evaluation_interval"] == 0:
            print(f"\n---Evaluating test set on epoch {epoch}---")
            evaluate.get_results(
                model, img_folder.test, opt, class_names, logger, epoch
            )

    yoloutils.clear_vram()
    return last_epoch


def get_free_gpus(config, model=None):
    if model is None:
        model = models.get_train_model(config)

    # input_shape = (3, config["img_size"], config["img_size"])

    # TODO: Fix error where calling this results in error
    # stat_str, _ = summary_string(model, input_shape, config["batch_size"])
    # target_str = "Estimated Total Size (MB): "
    # mem_str = [s for s in stat_str.split("\n") if target_str in s][0]

    # memory_needed = float(mem_str.split(target_str)[1])

    # Rough estimate of model size, in bytes
    memory_needed = config["img_size"] ** 2 * config["batch_size"] * 3 * 4 * 160
    free_gpus = dict()
    for i in range(cuda.device_count()):
        bytes_free = cuda.get_device_properties(i).total_memory - cuda.memory_allocated(
            i
        )
        if bytes_free > memory_needed:
            free_gpus[i] = bytes_free
    free_gpus = dict(sorted(free_gpus.items(), key=lambda gpu: gpu[1], reverse=True))
    return list(free_gpus.keys())
