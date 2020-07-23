from __future__ import division

import os
import time
import datetime
import argparse
import math

import torch
from torch.autograd import Variable

from terminaltables import AsciiTable

import yolov3.utils.utils as utils
import yolov3.utils.parse_config as parse
import yolov3.evaluate as evaluate
from yolov3.models import Darknet
from yolov3.utils.logger import Logger
from yolov3.utils.datasets import ListDataset

import backpack as bp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of each image batch"
    )
    parser.add_argument(
        "--gradient_accumulations",
        type=int,
        default=2,
        help="number of gradient accums before step",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="config/yolov3.cfg",
        help="path to model definition file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/coco.data",
        help="path to data config file",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        help="if specified starts from checkpoint model",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="size of each image dimension"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="interval between saving model weights",
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=1,
        help="interval evaluations on validation set",
    )
    parser.add_argument(
        "--compute_map", default=False, help="if True computes mAP every tenth batch"
    )
    parser.add_argument(
        "--multiscale_training", default=True, help="allow for multi-scale training"
    )
    parser.add_argument(
        "--resume", default=-1, type=int, help="resume training from a specific epoch"
    )
    parser.add_argument(
        "--prefix", default="yolov3", type=str, help="prefix for checkpoint files"
    )
    parser.add_argument(
        "--clip",
        default=float("inf"),
        type=float,
        help="cutoff value for gradient clipping",
    )

    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {DEVICE_STR} for training")
    device = torch.device(DEVICE_STR)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse.parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = utils.load_classes(data_config["names"])

    # Initiate model
    model = bp.extend(Darknet(opt.model_def), debug=False).to(device)
    model.apply(utils.weights_init_normal)

    if opt.resume != -1 and opt.pretrained_weights is None:
        opt.pretrained_weights = f"checkpoints/{opt.prefix}_ckpt_{int(opt.resume)}.pth"

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    for param in model.parameters():
        param.requires_grad = True

    # Get dataloader
    dataset = ListDataset(train_path, augment=False, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
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

    # Modify this if needed; intended to reduce log size
    log_interval = int(len(dataloader) / 50)

    old_criteria = 0

    for epoch in range(opt.resume + 1, opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device), requires_grad=True)
            targets = Variable(targets.to(device), requires_grad=True)

            loss, outputs = model(imgs, targets)

            with bp.backpack(bp.extensions.Variance(), bp.extensions.BatchGrad()):
                loss.backward()

            vals = list()

            for name, param in model.named_parameters():
                try:
                    grad = param.grad_batch
                    var = param.variance

                    dim = 1
                    for i in grad.shape:
                        dim *= i

                    scale_factor = opt.batch_size / dim
                    val = 1 - scale_factor * torch.sum(
                        torch.div(torch.square(grad), var)
                    )

                    vals.append(val)
                    # print(name, dim, val)

                except AttributeError:
                    pass
                    # print(name)

            scale_factor = 1 / len(vals)
            stopping_criteria = scale_factor * sum(vals)

            non_nan_vals = [val for val in vals if not math.isnan(val)]
            scale_factor = 1 / len(non_nan_vals)
            stopping_criteria2 = scale_factor * sum(non_nan_vals)

            alpha = 0.5
            if old_criteria == 0:
                smoothed_criteria = stopping_criteria2
            else:
                smoothed_criteria = (
                    alpha * stopping_criteria2 + (1 - alpha) * old_criteria
                )

            old_criteria = smoothed_criteria

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
                if batch_i % log_interval == 0:
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            log_str += f"\nStopping criteria {stopping_criteria}"
            log_str += f"\nStopping criteria (non-nan) {smoothed_criteria}"

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
            evaluate.get_results(model, valid_path, opt, class_names, logger, epoch)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/{opt.prefix}_ckpt_{epoch}.pth")
            if smoothed_criteria > 0:
                print("Stopping early")
                exit(0)
