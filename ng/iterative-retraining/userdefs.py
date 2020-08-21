def label_sample_set(img_path):
    """Sample function to label an image path with its ground truth with a list of labels.

    This function is customizable (e.g. including a GUI to annotate) depending on your needs.
    It should return a list of tuples, with each tuple representing a label with the values
    (class_label, bounding_box_x_center, bb_y_center, bb_width, bb_height)
    These coordinates should also be normalized according to the image's width and height.
    """
    path = img_path.replace("images", "classes")[:-4] + ".txt"
    if os.path.exists(path):
        labels = map(lambda x: map(float, x.split(" ")), open(path).read().split("\n"))
        for label in labels:
            label[0] = classes[int(label[0])]
        return labels
    return []


def get_sample_methods():
    return {
        "median-below-thresh": (sample.median_below_thresh_sample, {"thresh": 0.0}),
        "median-thresh": (sample.median_thresh_sample, {"thresh": 0.0}),
        "bin-quintile": (
            sample.bin_sample,
            {"stratify": False, "num_bins": 5, "curve": sample.const, "thresh": 0.0},
        ),
        "random": (sample.in_range_sample, {"min_val": 0.0, "max_val": 1.0}),
        "bin-normal": (
            sample.bin_sample,
            {
                "stratify": False,
                "num_bins": 5,
                "curve": sample.norm,
                "mean": 0.5,
                "std": 0.25,
            },
        ),
        "mid-below-thresh": (sample.in_range_sample, {"min_val": 0.0, "max_val": 0.5}),
        "iqr": (sample.iqr_sample, {"thresh": 0.0}),
        "normal": (sample.normal_sample, {"thresh": 0.0}),
        "mid-normal": (
            sample.normal_sample,
            {"thresh": 0.0, "avg": 0.5, "stdev": 0.25},
        ),
        "mid-thresh": (sample.in_range_sample, {"min_val": 0.5, "max_val": 1.0}),
    }
