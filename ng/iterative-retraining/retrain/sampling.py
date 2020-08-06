import random
import os

import statistics as stats
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def sort_list_dict(freq, desc=False):
    return dict(sorted(freq.items(), key=lambda item: len(item[1]), reverse=desc))


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


def prob_sample(result, desired, prob_func, *func_args, **func_kwargs):
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
            choose = random.random() <= prob_func(conf, *func_args, **func_kwargs)
            if choose:
                chosen_this_round.append(row)
        chosen += chosen_this_round
        for row in chosen_this_round:
            pool.remove(row)

    chosen = chosen[:desired]

    return chosen


def median_thresh_sample(result, thresh=0.5):
    confidences = result.get_confidences(thresh)

    median = stats.median(confidences)
    print(f"median: {median}")

    return prob_sample(result, in_range(result, median), const, median)


def median_below_thresh_sample(result, thresh=0.5):
    confidences = result.get_confidences(thresh)

    median = stats.median(confidences)
    print(f"median: {median}")

    return prob_sample(result, in_range(result, median), const, median, below=True)


def iqr_sample(result, thresh=0.5):
    confidences = result.get_confidences(thresh)
    q1 = np.quantile(confidences, 0.25, interpolation="midpoint")
    q3 = np.quantile(confidences, 0.75, interpolation="midpoint")

    print(f"q1: {q1}, q3: {q3}")

    return prob_sample(result, in_range(result, q1, q3), const, q1, q3)


def normal_sample(result, p=0.75, thresh=0.5):
    """Sample all within one standard deviation of mean."""
    confidences = result.get_confidences(thresh)

    avg = stats.mean(confidences)
    stdev = stats.stdev(confidences)
    print(f"avg: {avg}, stdev: {stdev}")

    return prob_sample(
        result, round(p * in_range(result, avg - stdev, avg + stdev)), norm, avg, stdev,
    )


def const(conf, thresh=0.5, max_val=1.0, below=False):
    if not below:
        if max_val >= conf >= thresh:
            return 1.0
    else:
        if thresh >= conf:
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


def create_sample(results, name, max_samp, sample_func, **func_args):
    # The first part of this function simulates decisions made at the edge
    retrain_by_class = list()
    print(f"===== {name} ======")
    for result in results:
        if result.name == "All":
            continue
        retrain_by_class.append(sample_func(result, **func_args))

    retrain = list()

    # Evaluate the numbers that may be under the quota first
    # to distribute samples among all (inferred) classes
    retrain_by_class = sorted(retrain_by_class, key=len)
    for i, sample_list in enumerate(retrain_by_class):
        random.shuffle(sample_list)
        images_left = max_samp - len(retrain)
        images_per_class = round(images_left / (len(retrain_by_class) - i + 1))

        retrain += sample_list[: min(len(sample_list), images_left)]

    # At this point, images are "received" in the cloud
    # This process simulates manually labeling/verifying all inferences
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
