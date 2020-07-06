from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from terminaltables import AsciiTable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(
        tqdm.tqdm(dataloader, desc="Detecting objects")
    ):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(
                outputs, conf_thres=conf_thres, nms_thres=nms_thres
            )

        sample_metrics += get_batch_statistics(
            outputs, targets, iou_threshold=iou_thres
        )

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))
    ]
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels
    )

    return precision, recall, AP, f1, ap_class


def get_results(model, path, opt, class_names, logger=None, epoch=0):
    print("\n---- Evaluating Model ----")

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]

    if logger is not None:
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=8, help="size of each image batch"
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
        "--weights_path",
        type=str,
        default="weights/yolov3.weights",
        help="path to weights file",
    )
    parser.add_argument(
        "--class_path",
        type=str,
        default="data/coco.names",
        help="path to class label file",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.5,
        help="iou threshold required to qualify as detected",
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--nms_thres",
        type=float,
        default=0.5,
        help="iou thresshold for non-maximum suppression",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--img_size", type=int, default=416, help="size of each image dimension"
    )
    parser.add_argument(
        "--log_epoch", type=int, default=False, help="log results up to a certain epoch"
    )
    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not opt.log_epoch:
        model = get_eval_model(opt.model_def, opt.img_size, opt.weights_path)
        get_results(model, valid_path, opt, class_names)
    else:
        weights_path = opt.weights_path + "0.pth"
        model = get_eval_model(opt.model_def, opt.img_size, weights_path)
        logger = Logger("logs")

        for i in range(opt.log_epoch + 1):
            weights_path = opt.weights_path + str(i) + ".pth"
            model.load_state_dict(torch.load(weights_path, map_location=device))

            get_results(model, valid_path, opt, class_names, logger, i)
