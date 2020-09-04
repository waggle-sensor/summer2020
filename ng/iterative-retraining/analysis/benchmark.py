import math

import os
import glob
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


from yolov3 import evaluate
from yolov3 import models
from yolov3 import utils as yoloutils
from retrain import utils
from retrain.dataloader import LabeledSet


def get_checkpoint(folder, prefix, epoch):
    """Retrieve the checkpoint file corresponding to the given prefix and epoch."""
    ckpts = glob.glob(f"{folder}/{prefix}*_ckpt_{epoch}.pth")

    if len(ckpts) == 0:
        return f"{folder}/init_ckpt_{epoch}.pth"

    return ckpts[0]


def benchmark(img_folder, prefix, epoch, config):
    return benchmark_avg(img_folder, prefix, epoch, epoch, 1, config)


def get_img_detections(checkpoints, prefix, config, loader, silent):
    detections_by_img = dict()
    model_def = yoloutils.parse_model_config(config["model_config"])
    model = models.get_eval_model(model_def, config["img_size"])
    yoloutils.clear_vram()

    for epoch in tqdm(checkpoints, "Benchmarking epochs", disable=silent):
        ckpt = get_checkpoint(config["checkpoints"], prefix, epoch)
        model.load_state_dict(torch.load(ckpt, map_location=model.device))

        for (img_paths, input_imgs) in loader:
            path = img_paths[0]
            if path not in detections_by_img.keys():
                detections_by_img[path] = None

            while True:
                try:
                    detections = evaluate.detect(
                        input_imgs, config["conf_thres"], model, config["nms_thres"]
                    )
                    break
                except RuntimeError:
                    # Cuda out of memory
                    model.to(yoloutils.get_device())
                    yoloutils.clear_vram()

            detections = [d for d in detections if d is not None]

            if len(detections) == 0:
                continue

            detections = torch.stack(detections)
            if detections_by_img[path] is None:
                detections_by_img[path] = detections
            else:
                detections_by_img[path] = torch.cat(
                    (detections_by_img[path], detections), 1
                )
    return detections_by_img


def make_results_df(config, img_folder, detections_by_img, total_epochs):
    metrics = [
        "file",
        "actual",
        "detected",
        "conf",
        "conf_std",
        "hit",
    ]

    results = pd.DataFrame(columns=metrics)
    classes = utils.load_classes(config["class_list"])

    for path, detections in detections_by_img.items():
        ground_truths = img_folder.get_classes(utils.get_label_path(path))
        detection_pairs = list()
        if detections is not None:
            region_detections, regions_std = yoloutils.group_average_bb(
                detections, total_epochs, config["iou_thres"]
            )

            # evaluate.save_image(region_detections, path, config, classes)
            if len(region_detections) == 1:
                detected_class = int(region_detections.numpy()[0][-1])
                if detected_class in ground_truths:
                    label = detected_class
                elif len(ground_truths) == 1:
                    label = ground_truths[0]
                else:
                    label = None
                detection_pairs = [(label, region_detections[0])]
            else:
                test_img = LabeledSet([path], len(classes))
                detection_pairs = evaluate.match_detections(
                    test_img, region_detections.unsqueeze(0), config
                )

        for (truth, box) in detection_pairs:
            if box is None:
                continue
            obj_conf, class_conf, pred_class = box.numpy()[4:]
            obj_std, class_std = regions_std[round(float(class_conf), 3)]

            row = {
                "file": path,
                "detected": classes[int(pred_class)],
                "actual": classes[int(truth)] if truth is not None else "",
                "conf": obj_conf * class_conf,
                "conf_std": math.sqrt(obj_std ** 2 + class_std ** 2),
            }
            row["hit"] = row["actual"] == row["detected"]

            results = results.append(row, ignore_index=True)

            if truth is not None:
                ground_truths.remove(int(truth))

        # Add rows for those missing detections
        for truth in ground_truths:
            row = {
                "file": path,
                "detected": "",
                "actual": classes[int(truth)],
                "conf": 0.0,
                "hit": False,
                "conf_std": 0.0,
            }

            results = results.append(row, ignore_index=True)
    return results


def benchmark_avg(img_folder, prefix, start, end, total_epochs, config, roll=False):
    loader = DataLoader(
        img_folder, batch_size=1, shuffle=False, num_workers=config["n_cpu"],
    )

    if roll:
        checkpoints_i = list(range(max(1, end - total_epochs + 1), end + 1))
    else:
        checkpoints_i = list(
            sorted(set(np.linspace(start, end, total_epochs, dtype=np.dtype(np.int16))))
        )

    single = total_epochs == 1
    if not single:
        print("Benchmarking on epochs", checkpoints_i)

    detections_by_img = get_img_detections(
        checkpoints_i, prefix, config, loader, single
    )

    results = make_results_df(config, img_folder, detections_by_img, total_epochs)
    results.sort_values(by="file", inplace=True)
    return results


def save_results(results, filename):
    output = open(filename, "w+")

    metrics = results.columns.tolist()
    results.to_csv(output, columns=metrics, index=False)
    output.close()


def series_benchmark_loss(img_folder, prefix, start, end, delta, config, filename=None):
    if filename is None:
        filename = f"{prefix}_loss_{start}_{end}.csv"

    out = open(f"{config['output']}/{filename}", "w+")
    out.write("epoch,loss,mAP,precision\n")

    model_def = yoloutils.parse_model_config(config["model_config"])
    model = models.get_eval_model(model_def, config["img_size"])

    for epoch in tqdm(range(start, end + 1, delta), "Benchmarking epochs"):
        ckpt = get_checkpoint(config["checkpoints"], prefix, epoch)
        model.load_state_dict(torch.load(ckpt, map_location=model.device))

        results = evaluate.get_results(model, img_folder, config, list(), silent=True)
        out.write(f"{epoch},{results['val_loss']},{results['val_mAP']}\n")
    out.close()


def simple_benchmark_avg(
    img_folder, prefix, start, end, total_epochs, config, roll=False
):
    """Deprecated version of benchmark averaging, meant for single object
    detection within an image. Used for a fair comparison baseline on old models
    """

    loader = DataLoader(
        img_folder, batch_size=1, shuffle=False, num_workers=config["n_cpu"],
    )

    results = pd.DataFrame(
        columns=["file", "confs", "actual", "detected", "conf", "hit"]
    )
    results.set_index("file")

    classes = utils.load_classes(config["class_list"])

    if roll:
        checkpoints_i = list(range(max(1, end - total_epochs + 1), end + 1))
    else:
        checkpoints_i = list(
            sorted(set(np.linspace(start, end, total_epochs, dtype=np.dtype(np.int16))))
        )

    single = total_epochs == 1

    if not single:
        print("Benchmarking on epochs", checkpoints_i)

    for n in tqdm(checkpoints_i, "Benchmarking epochs", disable=single):
        ckpt = get_checkpoint(config["checkpoints"], prefix, n)

        model_def = yoloutils.parse_model_config(config["model_config"])
        model = models.get_eval_model(model_def, config["img_size"], ckpt)

        for (img_paths, input_imgs) in loader:
            path = img_paths[0]
            if path not in results.file:
                actual_class = classes[
                    img_folder.get_classes(utils.get_label_path(path))[0]
                ]
                results.loc[path] = [path, dict(), actual_class, None, None, None]

            detections = evaluate.detect(input_imgs, config["conf_thres"], model)

            confs = results.loc[path]["confs"]

            for detection in detections:
                if detection is None:
                    continue
                (_, _, _, _, _, cls_conf, cls_pred) = detection.numpy()[0]

                if cls_pred not in confs.keys():
                    confs[cls_pred] = [cls_conf]

                else:
                    confs[cls_pred].append(cls_conf)

    for _, row in results.iterrows():
        best_class = None
        best_conf = float("-inf")

        for class_name, confs in row["confs"].items():
            avg_conf = sum(confs) / len(checkpoints_i)

            if avg_conf > best_conf:
                best_conf = avg_conf
                best_class = class_name

        if best_class is not None:
            row["detected"] = classes[int(best_class)]
            row["conf"] = best_conf
            row["hit"] = row["actual"] == row["detected"]
        else:
            row["detected"] = ""
            row["conf"] = 0.0
            row["hit"] = False

    return results


def get_test_sets(config, prefix):
    out_dir = config["output"]
    num_classes = len(utils.load_classes(config["class_list"]))
    epoch_splits = utils.get_epoch_splits(config, prefix)

    # Initial test set
    init_test_set = f"{out_dir}/init_test.txt"
    init_test_folder = LabeledSet(init_test_set, num_classes)

    # Only data from the (combined) iteration test sets (75% sampling + 25% seen data)
    iter_test_sets = [
        f"{out_dir}/{prefix}{i}_test.txt" for i in range(len(epoch_splits))
    ]
    iter_img_files = list()
    for file in iter_test_sets:
        iter_img_files += utils.get_lines(file)
    all_iter_sets = LabeledSet(iter_img_files, num_classes)

    # Test sets filtered for only sampled images
    sampled_imgs = [img for img in iter_img_files if config["sample_set"] in img]
    sample_test = LabeledSet(sampled_imgs, num_classes)

    # Data from all test sets
    all_test = LabeledSet(sampled_imgs, num_classes)
    all_test += init_test_folder

    test_sets = {
        "init": init_test_folder,
        "all_iter": all_iter_sets,
        "sample": sample_test,
        "all": all_test,
    }

    if prefix != "init":
        for i in range(len(epoch_splits)):
            test_sets[f"cur_iter{i + 1}"] = LabeledSet(iter_test_sets[i], num_classes)

    return test_sets


def series_benchmark(config, opt, prefix):
    # 1. Find the number of batches for the given prefix
    # 2. Find the starting/ending epochs of each split
    # 3. Benchmark that itertion's test set with the average method
    #    (Could plot this, but may not be meaningful due to differing test sets)
    # 4. Benchmark the overall test set with the same average method (and save results)
    #    4a. plot the overall test set performance as a function of epoch number
    # 5. (optional) serialize results of the overall test set as JSON for improved speed
    #    when using averages
    test_sets = get_test_sets(config, prefix)
    epoch_splits = utils.get_epoch_splits(config, prefix, True)

    if opt.batch_test is not None:
        test_sets["batch_test"], batches_removed = get_batch_test_set(
            config, opt.batch_test
        )
        epoch_splits = epoch_splits[:-batches_removed]

    # Begin benchmarking
    out_folder = f"{config['output']}/{prefix}-series"
    if opt.avg or opt.roll_avg:
        out_folder += "-roll-avg" if opt.roll_avg else "-avg"
    os.makedirs(out_folder, exist_ok=True)

    num_ckpts = get_num_ckpts(config, opt)
    for i, split in enumerate(epoch_splits):
        if i == 0:
            if "baseline" not in prefix or prefix != "init":
                # No need to benchmark initial series again
                continue
            else:
                start = 1
        else:
            start = epoch_splits[i - 1]

        for epoch in tqdm(range(start, split + 1, opt.delta)):
            for name, img_folder in test_sets.items():
                # Benchmark both iterations sets at the split mark
                if (epoch == start and "cur_iter" not in name) or (
                    "cur_iter" in name and name != f"cur_iter{i}"
                ):
                    continue

                out_name = f"{out_folder}/{name}_{epoch}.csv"
                if os.path.exists(out_name):
                    continue

                if opt.roll_avg:
                    result_df = benchmark_avg(
                        img_folder, prefix, 1, epoch, num_ckpts, config, roll=True
                    )
                elif opt.avg:
                    result_df = benchmark_avg(
                        img_folder, prefix, 1, epoch, num_ckpts, config
                    )
                else:
                    result_df = benchmark(img_folder, prefix, epoch, config)
                save_results(result_df, out_name)


def benchmark_next_batch(prefix, config, opt):
    """See initial training performance on batch splits."""
    out_dir = config["output"]
    num_classes = len(utils.load_classes(config["class_list"]))
    batch_sets = sorted(glob.glob(f"{out_dir}/sample*.txt"), key=utils.get_sample)

    epoch_splits = utils.get_epoch_splits(config, prefix, True)
    if prefix == "init":
        epoch_splits *= len(batch_sets)

    batch_folders = [LabeledSet(batch_set, num_classes) for batch_set in batch_sets]

    def get_filename(i, end_epoch):
        filename = f"{out_dir}/{prefix}{i}_benchmark_"
        filename += "roll_" if opt.roll_avg else "avg_"
        filename += f"1_{end_epoch}.csv"
        return filename

    benchmark_batch_splits(
        prefix, batch_folders, epoch_splits, get_filename, config, opt
    )


def get_batch_test_set(config, reserve_batches):
    out_dir = config["output"]
    num_classes = len(utils.load_classes(config["class_list"]))
    batch_sets = sorted(glob.glob(f"{out_dir}/sample*.txt"), key=utils.get_sample)

    test_imgs = list()
    batches_removed = 0
    for batch_set in reversed(batch_sets):
        imgs = utils.get_lines(batch_set)
        if len(imgs) < config["sampling_batch"] or reserve_batches != 0:
            test_imgs += imgs
            batches_removed += 1
            if not (len(imgs) < config["sampling_batch"]):
                reserve_batches -= 1
    return LabeledSet(test_imgs, num_classes), batches_removed


def benchmark_batch_test_set(prefix, config, opt, reserve_batches=0):
    """Benchmark against a test set created from a specified number of batch sets,
    using a rolling average of epochs."""
    test_folder, batches_removed = get_batch_test_set(config, reserve_batches)
    epoch_splits = utils.get_epoch_splits(config, prefix, True)

    if prefix != "init":
        epoch_splits = epoch_splits[:-batches_removed]

    def get_filename(i, end_epoch):
        filename = f"{config['output']}/{prefix}{i}_benchmark_"
        if opt.roll_avg is None:
            filename += "avg_"
        filename += f"test_{end_epoch}.csv"
        return filename

    batch_folders = [test_folder] * len(epoch_splits)
    benchmark_batch_splits(
        prefix, batch_folders, epoch_splits, get_filename, config, opt
    )


def get_num_ckpts(config, opt):
    if opt.roll_avg is not None:
        return opt.roll_avg
    if opt.avg == "" or opt.avg is False:
        return config["conf_check_num"]

    return opt.avg


def benchmark_batch_splits(
    prefix, batch_sets, epoch_splits, filename_func, config, opt
):
    num_ckpts = get_num_ckpts(config, opt)

    for i, (batch_set, end_epoch) in enumerate(zip(batch_sets, epoch_splits)):
        if len(batch_set) < config["sampling_batch"]:
            break

        filename = filename_func(i, end_epoch)

        if os.path.exists(filename):
            continue
        if opt.roll_avg is not None:
            results = benchmark_avg(
                batch_set, prefix, 1, end_epoch, num_ckpts, config, roll=opt.roll_avg
            )
        else:
            results = benchmark_avg(batch_set, prefix, 1, end_epoch, num_ckpts, config)

        save_results(results, filename)
