from __future__ import division

import os
import time
import datetime

import torch
from torch.autograd import Variable

from terminaltables import AsciiTable

import yolov3.evaluate as evaluate
from yolov3.models import Darknet

import retrain.utils as utils
from retrain.dataloader import ListDataset
from yolov3.logger import Logger
import yolov3.utils as yoloutils


def train(img_folder, opt, load_weights=None):
    """Trains a given image set, with an early stop.
    
    Precondition: img_folder has been split into train, test, and validation sets.
    """
    os.makedirs(opt["checkpoints"], exist_ok=True)
    os.makedirs(opt["output"], exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_str} for training")
    yoloutils.clear_vram()

    device = yoloutils.get_device()
    model_def = utils.parse_model_config(opt["model_config"])
    model = Darknet(model_def, opt["img_size"]).to(device)

    # Initiate model
    model.apply(yoloutils.weights_init_normal)

    if load_weights is not None:
        model.load_state_dict(torch.load(load_weights))

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

                loss, outputs = model(imgs, targets)

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
                    opt["max_epochs"],
                    batch_i,
                    len(dataloader),
                )

                metric_table = [
                    ["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]
                ]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
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
                    seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)
                )
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

                model.seen += imgs.size(0)

            if epoch % opt["checkpoint_interval"] == 0:
                torch.save(
                    model.state_dict(),
                    f"{opt['checkpoints']}/{img_folder.prefix}_ckpt_{epoch}.pth",
                )

        else:
            model.load_state_dict(torch.load(ckpt_path))

        # Use UP criteria for early stop
        if bool(opt["early_stop"]) and (
            epoch == opt["start_epoch"] + 1 or epoch % opt["strip_len"] == 0
        ):
            print("\n---Evaluating validation set for early stop---")

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
            print("\n---Evaluating test set...---")
            evaluate.get_results(
                model, img_folder.test, opt, class_names, logger, epoch
            )

    yoloutils.clear_vram()
    return last_epoch
