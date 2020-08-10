from __future__ import division

import random
from PIL import Image
from tqdm import tqdm
import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from terminaltables import AsciiTable

from yolov3 import utils


def detect(input_imgs, conf_thres, model, nms_thres=0.5, nms=True):
    # Configure input
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = Variable(input_imgs.type(Tensor))

    with torch.no_grad():
        detections = model(input_imgs)
        detections = utils.non_max_suppression(detections, conf_thres, nms_thres)
    return detections


def save_images(imgs, img_detections, opt, best_label_only=False):
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            save_image(detections, path, opt, best_label_only)


def get_most_conf(detections):
    most_conf = None
    for d in detections:
        if most_conf is None or d[5] > most_conf[5]:
            most_conf = d
    return most_conf


def match_detections(model, img_folder, detections, config):
    """Match the labels for an image with its bounding box"""
    dataset = img_folder.to_dataset()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    device = utils.get_device()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    boxes = list()
    for (_, imgs, targets) in dataloader:

        imgs = Variable(imgs.to(device).type(Tensor), requires_grad=False)

        _, outputs = model(imgs, Variable(targets.to(device)))

        # Rescale target
        targets[:, 2:] = utils.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= config["img_size"]

        labels = targets[:, 1].tolist()

        # We wish to return a list of pairs (actual label, detection),
        # where either actual label or detection may be None
        overlaps = utils.get_batch_statistics(
            detections, targets, iou_threshold=config["iou_thres"]
        )[0]
        print(labels)

        detections = detections.squeeze(0)
        for i in range(len(overlaps[0])):
            for detection in detections:

                if detection[-1] == overlaps[2][i] and detection[-2] == overlaps[1][i]:
                    boxes.append((overlaps[0][i], detection))
                    break
        pairs = list()
        for label in labels:
            correct_box = None
            for i, (hit, detection) in enumerate(boxes):
                if hit and detection[-1] == label:
                    correct_box = detection
                    break
            del boxes[i]
            pairs.append((label, correct_box))
        for (hit, detection) in boxes:
            pairs.append((None, detection))
    print(pairs)
    return pairs


def save_image(detections, path, opt, classes, best_label_only=False):
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Create plot
    img = np.array(Image.open(path))
    fig_main = plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Rescale boxes to original image
    detections = utils.rescale_boxes(detections, opt["img_size"], img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)

    if best_label_only:
        detections = [get_most_conf(detections)]

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle(
            (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none",
        )
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s="\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()),
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = path.split("/")[-1].split(".")[0]
    plt.savefig(f"{opt['output']}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig_main)
    plt.close(fig)


def evaluate(
    model,
    img_list,
    iou_thres,
    conf_thres,
    nms_thres,
    img_size,
    batch_size,
    silent=False,
):
    # Get dataloader
    dataset = img_list.to_dataset(multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = utils.get_device()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    total_loss = 0.0
    for (_, imgs, targets) in tqdm(
        dataloader, desc="Detecting objects", disable=silent
    ):

        # Extract labels
        labels += targets[:, 1].tolist()

        imgs = Variable(imgs.to(device).type(Tensor), requires_grad=False)

        loss, outputs = model(imgs, Variable(targets.to(device)))
        total_loss += loss.item()

        # Rescale target
        targets[:, 2:] = utils.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        outputs = utils.non_max_suppression(
            outputs, conf_thres=conf_thres, nms_thres=nms_thres
        )

        sample_metrics += utils.get_batch_statistics(
            outputs, targets, iou_threshold=iou_thres
        )

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))
    ]
    precision, recall, AP, f1, ap_class = utils.ap_per_class(
        true_positives, pred_scores, pred_labels, labels
    )

    return precision, recall, AP, f1, ap_class, total_loss


def get_results(model, img_list, opt, class_names, logger=None, epoch=0, silent=False):

    precision, recall, AP, f1, ap_class, loss = evaluate(
        model,
        img_list=img_list,
        iou_thres=opt["iou_thres"],
        conf_thres=opt["conf_thres"],
        nms_thres=opt["nms_thres"],
        img_size=opt["img_size"],
        batch_size=8,
        silent=silent,
    )

    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
        ("val_loss", loss),
    ]

    if logger is not None:
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

    if not silent:
        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")

    return dict(evaluation_metrics)
