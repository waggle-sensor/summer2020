def parse_retrain_config(path):
    lines = [line for line in get_config_lines(path) if "=" in line]

    options = dict()
    for line in lines:
        key, value = [val.strip() for val in line.split("=")]

        try:
            options[key] = float(value)
        except ValueError:
            options[key] = value
    return options


def parse_model_config(path):
    """Parse the yolov3 layer configuration file and returns module definitions."""
    lines = get_config_lines(path)
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


def get_config_lines(path):
    file = open(path, "r")
    lines = file.read().split("\n")
    lines = [line.strip() for line in lines if line and "#" not in line]
    return lines


def load_classes(path):
    """Loads class labels at path."""
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names