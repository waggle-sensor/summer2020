import sys
import os
import glob
import cv2


def find_checkpoint(config, prefix, num):
    ckpt = f"{config['checkpoints']}/init_ckpt_{num}.pth"
    if not os.path.exists(ckpt):
        ckpt = glob.glob(f"{config['checkpoints']}/{prefix}*_ckpt_{num}.pth")[0]
    return ckpt


def get_label_path(img):
    return img[:-4].replace("images", "labels") + ".txt"


def parse_retrain_config(path):
    lines = [line for line in get_lines(path) if "=" in line]

    options = dict()
    for line in lines:
        key, value = [val.strip() for val in line.split("=")]

        try:
            options[key] = int(value)
        except ValueError:
            try:
                options[key] = float(value)
            except ValueError:
                options[key] = value
    return options


def parse_model_config(path):
    """Parse the yolov3 layer configuration file and returns module definitions."""
    lines = get_lines(path)
    module_defs = []
    for line in lines:
        if line.startswith("["):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


def get_lines(path):
    with open(path, "r") as file:
        lines = file.read().split("\n")
        return [line.strip() for line in lines if line and "#" not in line]


def load_classes(path):
    """Loads class labels at path."""
    with open(path, "r") as file:
        return file.read().split("\n")[:-1]


def save_stdout(filename, func, *pos_args, **var_args):
    old_stdout = sys.stdout
    sys.stdout = open(filename, "w+")
    func(*pos_args, **var_args)
    sys.stdout = old_stdout


def xyxy_to_darknet(img_path, x0, y0, x1, y1):
    
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    y1 = max(min(y1, h), 0)
    x1 = max(min(x1, w), 0)
    y0 = min(max(y0, 0), h)
    x0 = min(max(x0, 0), w)

    rect_h = y1 - y0
    rect_w = x1 - x0
    x_center = rect_w / 2 + x0
    y_center = rect_h / 2 + y0


    return x_center / w, y_center / h, rect_w / w, rect_h / h
