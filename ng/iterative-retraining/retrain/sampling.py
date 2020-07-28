import random
import numpy as np
import statistics as stats
import os
import utils
import yolov3.utils.parse_config as parser
import scipy.stats
import matplotlib.pyplot as plt


def sort_list_dict(freq, desc=False):
    return {
        k: v
        for k, v in sorted(freq.items(), key=lambda item: len(item[1]), reverse=desc)
    }


def multi_argmax(arr):
    return [i for i, x in enumerate(arr) if x == np.max(arr)]


def iterative_stratification(images, proportions):
    remaining = dict()

    # Build the list of images per label that have not
    # been allocated to a subset yet
    for img, class_list in images.items():
        for c in class_list:
            if c not in remaining.keys():
                remaining[c] = set()
            remaining[c].add(img)

    desired = [dict() for _ in proportions]
    subsets = [list() for _ in proportions]

    # Compute the desired number of examples for each label,
    # for each subset
    for c, imgs in remaining.items():
        for i, weight in enumerate(proportions):
            desired[i][c] = round(len(imgs) * weight)

    while len(images.keys()) > 0:
        # Allocate the least frequent label (with at least
        # 1 example remaining) first
        remaining = sort_list_dict(remaining)
        least_freq_label = list(remaining.keys())[0]

        label_imgs = list(remaining[least_freq_label])
        random.shuffle(label_imgs)

        for img in label_imgs:
            # Allocate image to subset that needs the most of that label
            label_counts = [lab[least_freq_label] for lab in desired]
            subset_indexes = multi_argmax(label_counts)

            if len(subset_indexes) > 1:
                # Break ties by subset that needs the most overall examples
                all_label_counts = [sum(desired[i].values()) for i in subset_indexes]

                subset_indexes = [
                    subset_indexes[x] for x in multi_argmax(all_label_counts)
                ]
                if len(subset_indexes) > 1:
                    # Break further ties randomly
                    random.shuffle(subset_indexes)

            # Add image to the chosen subset and remove the image
            idx = subset_indexes[0]
            subset = subsets[idx]
            subset.append(img)

            for img_set in remaining.values():
                if img in img_set:
                    img_set.remove(img)

            # Decrease the desired number, based on all labels in that example
            for c in images[img]:
                desired[idx][c] -= 1

            images.pop(img)
        remaining.pop(least_freq_label)

    return subsets


def prob_sample(result, desired, prob_func, *prob_func_args):
    """Generate a list of files for sampling.
    result:     a ClassResult holding a list of images
    desired:    number of samples to extract per class
    prob_func:  a function that takes a confidence score as an input and
                outputs the probability of sampling the image with that confidence

    The function continues sampling until the desired number of samples is hit.
    Consequently, the probability function should be well-chosen to prevent long runtimes.
    """
    pool = result.get_all()
    random.shuffle(pool)
    chosen = list()

    while len(chosen) < desired:
        chosen_this_round = list()
        for row in pool:
            conf = row["conf"]
            choose = random.random() <= prob_func(conf, *prob_func_args)
            if choose:
                chosen_this_round.append(row)
        chosen += chosen_this_round
        for row in chosen_this_round:
            pool.remove(row)

    chosen = chosen[:desired]

    return chosen


def median_thresh_sample(result):
    confidences = result.get_confidences()

    median = stats.median(confidences)
    print(f"median: {median}")

    return prob_sample(result, in_range(result, median), const, median)


def iqr_sample(result, thresh=0.5):
    confidences = [conf for conf in result.get_confidences() if conf >= thresh]
    q1 = np.quantile(confidences, 0.25, interpolation="midpoint")
    q2 = np.quantile(confidences, 0.5, interpolation="midpoint")

    print(f"q1: {q1}, q2: {q2}")

    return prob_sample(result, in_range(result, q1, q2), const, q1, q2)


def normal_sample(result, p=0.4, thresh=0.5):
    """Sample all within one standard deviation of mean."""
    confidences = [conf for conf in result.get_confidences() if conf >= thresh]

    avg = stats.mean(confidences)
    stdev = stats.stdev(confidences)
    print(f"avg: {avg}, stdev: {stdev}")

    return prob_sample(
        result, round(p * in_range(result, avg - stdev, avg + stdev)), norm, avg, stdev,
    )


def const(conf, thresh=0.5, max_val=1.0):
    if max_val >= conf >= thresh:
        return 1.0
    return 0.0


def norm(conf, mean, std):
    return scipy.stats.norm(mean, std).pdf(conf)


# TODO generalize this
def create_labels(retrain_list, use_actual=False):
    classes = open("config/chars.names").read().split("\n")[:-1]
    for result in retrain_list:
        idx = (
            classes.index(result["actual"])
            if use_actual
            else classes.index(result["detected"])
        )

        label_path = result["file"].replace("images", "labels")[:-4] + ".txt"
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w+") as label:
            label.write(f"{idx} 0.5 0.5 1 1")


def in_range(result, min_val, max_val=1.0):
    """Get the number of elements in a ClassResult above a threshold."""
    return len([res for res in result.get_all() if max_val >= res["conf"] >= min_val])


def create_config(samples, sample_name, data_config):
    data_opts = parser.parse_data_config(data_config)

    new_valid = data_opts["valid"].replace(".txt", "-new.txt")
    if not os.path.exists(new_valid):
        utils.rewrite_test_list(data_opts["valid"], ORIG_DATA)

    config_path = OUTPUT + f"configs-retrain/{sample_name}/"
    os.makedirs(config_path, exist_ok=True)

    with open(config_path + "train.txt", "w+") as out:
        files = [result["file"] for result in samples]
        out.write("\n".join(files) + "\n")

    data_opts["train"] = config_path + "train-aug.txt"
    data_opts["valid"] = new_valid
    with open(config_path + data_config.split("/")[-1], "w+") as new_data_config:
        for k, v in data_opts.items():
            new_data_config.write(f"{k} = {v}\n")


def create_sample(results, name, max_samp, sample_func, *func_args):
    retrain_by_class = list()

    print(f"===== {name} ======")
    for result in results:
        if result.name == "All":
            continue
        retrain_by_class.append(sample_func(result, *func_args))

    retrain = list()
    for sample_list in retrain_by_class:
        if undersample:
            retrain += sample_list[:max_samp]
        else:
            retrain += sample_list

    create_labels(retrain, use_actual=True)

    return retrain


def sample_histogram(retrain, title):
    colors = ["lightgreen", "red"]

    hit = list()
    miss = list()

    for data in retrain:
        if data["hit"] == "True":
            hit.append(data["conf"])
        else:
            miss.append(data["conf"])
    hit_miss = [hit, miss]

    plt.figure()
    plt.hist(hit_miss, bins=10, color=colors, stacked=True)
    plt.xlabel("Confidence")
    plt.ylabel("Count of Chosen")
    if title == "iqr":
        title = "Quartile Range"
    plt.title(title + f" (n={len(retrain)})")
    plt.show()
