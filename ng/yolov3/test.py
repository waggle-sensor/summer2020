from __future__ import division

import argparse
import torch

import yolov3.utils.utils as utils
import yolov3.utils.parse_config as parse
import yolov3.evaluate as evaluate
from yolov3.utils.logger import Logger


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
        "--conf_thres", type=float, default=0.5, help="object confidence threshold"
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

    data_config = parse.parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = utils.load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not opt.log_epoch:
        model = evaluate.get_eval_model(opt.model_def, opt.img_size, opt.weights_path)
        evaluate.get_results(model, valid_path, opt, class_names)
    else:
        weights_path = opt.weights_path + "0.pth"
        model = evaluate.get_eval_model(opt.model_def, opt.img_size, weights_path)
        logger = Logger("logs")

        for i in range(opt.log_epoch + 1):
            weights_path = opt.weights_path + str(i) + ".pth"
            model.load_state_dict(torch.load(weights_path, map_location=device))

            evaluate.get_results(model, valid_path, opt, class_names, logger, i)
