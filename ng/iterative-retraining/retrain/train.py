from __future__ import division

import os
import time
import datetime

import torch
from torch.autograd import Variable

from terminaltables import AsciiTable

import retrain.evaluate as evaluate
from retrain.models import Darknet

import retrain.utils as utils
from retrain.dataloader import ListDataset, ImageFolder
from retrain.logger import Logger


def train(folder, opt, model_def, load_weights=None):

    logger = Logger(opt["log"])
    os.makedirs(opt["checkpoints"], exist_ok=True)
    os.makedirs(opt["output"], exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device_str} for training")
    device = torch.device(device_str)

    model = Darknet(model_def).to(device)

    # Initiate model
    model.apply(utils.weights_init_normal)

    if load_weights is not None:
        model.load_state_dict(torch.load(load_weights))

    class_names = utils.load_classes(opt["class_list"])
    img_folder = ImageFolder(folder, len(class_names))

    test_prop = 1 - opt["train_init"] - opt["valid_init"]
    img_splits = img_folder.split_img_set(
        opt["train_init"], opt["valid_init"], test_prop
    )
    (train, valid, test) = img_splits

    # Get dataloader
    dataset = ListDataset(
        train.imgs,
        img_size=model_def["img_size"],
        compose=bool(opt["aug_compose"]),
        multiscale=bool(opt["multiscale"]),
    )

    dataset.augment()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model_def["batch"],
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
    log_freq = opt["logs_per_epoch"] if "logs_per_epoch" in opt.keys() else 50
    log_interval = int(len(dataloader) / log_freq)

    end_epoch = opt["start_epoch"] + opt["max_epochs"]

    for epoch in range(opt["start_epoch"], end_epoch):
        model.train()
        start_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device))

            loss, outputs = model(imgs, targets)

            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                epoch,
                opt.epochs,
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
                # tensorboard_log += [("stopping", stop_criteria)]
                if batch_i % log_interval == 0:
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            # log_str += f"\nStopping criteria (non-nan) {stop_criteria}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(
                seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)
            )
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            opt.iou_thres = 0.5
            opt.conf_thres = 0.5
            opt.nms_thres = 0.5
            evaluate.get_results(model, valid.imgs, opt, class_names, logger, epoch)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(
                model.state_dict(),
                f"{opt['checkpoints']}/{opt['prefix']}_ckpt_{epoch}.pth",
            )
            # if opt.early_stop and stop_criteria > 0:
            #     print(f"Stopping early, at epoch {epoch}")
            #     exit(0)