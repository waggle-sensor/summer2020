from __future__ import division, absolute_import

import torch
import torch.nn as nn
import numpy as np

from yolov3 import utils

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
else:
    from torch import FloatTensor


def get_modules_filters(module_def, output_filters, hyperparams):
    new_modules = list()
    filters = None
    if module_def["type"] == "convolutional":
        bn = int(module_def["batch_normalize"])
        filters = int(module_def["filters"])
        kernel_size = int(module_def["size"])
        pad = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            in_channels=output_filters[-1],
            out_channels=filters,
            kernel_size=kernel_size,
            stride=int(module_def["stride"]),
            padding=pad,
            bias=not bn,
        )
        new_modules.append(("conv", conv,))
        if bn:
            batch_norm = nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
            new_modules.append(("batch_norm", batch_norm))
        if module_def["activation"] == "leaky":
            new_modules.append(("leaky", nn.LeakyReLU(0.1)))

    elif module_def["type"] == "maxpool":
        kernel_size = int(module_def["size"])
        stride = int(module_def["stride"])
        if kernel_size == 2 and stride == 1:
            new_modules.append(("_debug_padding", nn.ZeroPad2d((0, 1, 0, 1))))
        maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2),
        )
        new_modules.append(("maxpool", maxpool))

    elif module_def["type"] == "upsample":
        upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
        new_modules.append(("upsample", upsample))

    elif module_def["type"] == "route":
        layers = [int(x) for x in module_def["layers"].split(",")]
        filters = sum([output_filters[1:][i] for i in layers])
        new_modules.append(("route", EmptyLayer()))

    elif module_def["type"] == "shortcut":
        filters = output_filters[1:][int(module_def["from"])]
        new_modules.append(("shortcut", EmptyLayer()))

    elif module_def["type"] == "yolo":
        anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
        # Extract anchors
        anchors = [int(x) for x in module_def["anchors"].split(",")]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        num_classes = int(module_def["classes"])
        img_size = int(hyperparams["height"])
        # Define detection layer
        yolo_layer = YOLOLayer(anchors, num_classes, img_size)

        new_modules.append(("yolo", yolo_layer))
    return new_modules, filters


def create_modules(hyperparams, module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        new_modules, filters = get_modules_filters(
            module_def, output_filters, hyperparams
        )
        for prefix, layer in new_modules:
            modules.add_module(f"{prefix}_{module_i}", layer)

        # Register module list and number of output filters
        module_list.append(modules)
        if filters is not None:
            output_filters.append(filters)

    return module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = (
            torch.arange(g)
            .repeat(g, 1)
            .view([1, 1, g, g])
            .type(FloatTensor)
            .cuda(self.device)
        )
        self.grid_y = (
            torch.arange(g)
            .repeat(g, 1)
            .t()
            .view([1, 1, g, g])
            .type(FloatTensor)
            .cuda(self.device)
        )
        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        ).cuda(self.device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(
                num_samples,
                self.num_anchors,
                self.num_classes + 5,
                grid_size,
                grid_size,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape).cuda(self.device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            (
                iou_scores,
                class_mask,
                obj_mask,
                noobj_mask,
                tx,
                ty,
                tw,
                th,
                tcls,
                tconf,
            ) = utils.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                device=self.device,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = (
                self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            )
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": utils.to_cpu(total_loss).item(),
                "x": utils.to_cpu(loss_x).item(),
                "y": utils.to_cpu(loss_y).item(),
                "w": utils.to_cpu(loss_w).item(),
                "h": utils.to_cpu(loss_h).item(),
                "conf": utils.to_cpu(loss_conf).item(),
                "cls": utils.to_cpu(loss_cls).item(),
                "cls_acc": utils.to_cpu(cls_acc).item(),
                "recall50": utils.to_cpu(recall50).item(),
                "recall75": utils.to_cpu(recall75).item(),
                "precision": utils.to_cpu(precision).item(),
                "conf_obj": utils.to_cpu(conf_obj).item(),
                "conf_noobj": utils.to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, model_def, img_size=416):
        super(Darknet, self).__init__()
        self.hyperparams = model_def[0]
        self.module_defs = model_def[1:]
        self.module_list = create_modules(self.hyperparams, self.module_defs)
        self.yolo_layers = [
            layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")
        ]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for layer in self.yolo_layers:
            layer.to(*args, **kwargs)
            layer.device = args[0]
        for module in self.module_list:
            module.to(*args, **kwargs)
            module.device = args[0]
        return self

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for module_def, module in zip(self.module_defs, self.module_list):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat(
                    [
                        layer_outputs[int(layer_i)]
                        for layer_i in module_def["layers"].split(",")
                    ],
                    1,
                )
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = utils.to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(
                f, dtype=np.int32, count=5
            )  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        conv_layer.bias
                    )
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for _, (module_def, module) in enumerate(
            zip(self.module_defs[:cutoff], self.module_list[:cutoff])
        ):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def get_eval_model(model_def, img_size, weights_path=None):
    device = utils.get_device()

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)
    model.device = device
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Set in evaluation mode

    return model


def get_train_model(config, device=None):
    model_def = utils.parse_model_config(config["model_config"])
    model = Darknet(model_def, config["img_size"])

    # Rough estimate of model size, in bytes
    memory_needed = utils.get_memory_needed(config)
    free_gpus = utils.get_free_gpus(memory_needed)

    if device is None or device not in free_gpus:
        device_str = f"cuda:{free_gpus[0]}" if len(free_gpus) != 0 else "cpu"
    else:
        device_str = f"cuda:{device}"

    device = torch.device(device_str)
    model.device = device

    return model.to(device)
