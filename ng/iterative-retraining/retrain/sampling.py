import random
import os

import statistics as stats
import numpy as np
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from retrain import utils


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

    desired = [dict() for _ in range(len(proportions))]
    subsets = [list() for _ in range(len(proportions))]

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


def create_sample(results, max_samp, sample_func, stratify=True, **func_args):
    # The first part of this function simulates decisions made at the edge
    retrain_by_class = list()

    if stratify:
        for result in results:
            if result.name == "All":
                continue
            sample = sample_func(result, **func_args)

            retrain_by_class.append(sample)
    else:
        retrain_by_class = [sample_func(results[-1], **func_args)]

    retrain = list()

    # Evaluate the numbers that may be under the quota first
    # to distribute samples among all (inferred) classes
    retrain_by_class = sorted(retrain_by_class, key=len)
    for i, sample_list in enumerate(retrain_by_class):
        # Remove duplicates due to multple labels per sample
        sample_list = [img for img in sample_list if img not in retrain]
        random.shuffle(sample_list)
        images_left = max_samp - len(retrain)
        images_per_class = round(images_left / (len(retrain_by_class) - i))

        # Enforce bandwidth limit
        retrain += sample_list[: min(len(sample_list), images_per_class)]

    return retrain


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
    chosen = set()

    while len(chosen) < desired:
        chosen_this_round = list()
        random.shuffle(pool)
        for row in pool:
            conf = row["conf"]
            choose = random.random() <= prob_func(conf, *func_args, **func_kwargs)
            if choose:
                chosen_this_round.append(row)
        for row in chosen_this_round:
            pool.remove(row)
            chosen.add(row["file"])

    chosen = list(chosen)[:desired]
    return chosen


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


def bin_sample(result, num_bins, curve, start=0.0, end=1.0, **func_kwargs):
    delta = (end - start) / num_bins
    bins = [
        in_range_sample(result, i * delta, (i + 1) * delta) for i in range(num_bins)
    ]

    total_area = integrate.quad(lambda x: curve(x, **func_kwargs), start, end)[0]

    chosen = list()
    for i, bin_imgs in enumerate(bins):
        bin_area = integrate.quad(
            lambda x: curve(x, **func_kwargs), i * delta, (i + 1) * delta
        )[0]
        bin_desired = round(bin_area / total_area * len(result))
        if bin_desired >= len(bin_imgs):
            chosen += bin_imgs
        else:
            random.shuffle(bin_imgs)
            chosen += bin_imgs[:bin_desired]
    return list(set(chosen))


def in_range_sample(result, min_val, max_val):
    return prob_sample(
        result,
        in_range(result, min_val, max_val),
        const,
        thresh=min_val,
        max_val=max_val,
    )


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


def normal_sample(result, avg=None, stdev=None, p=0.75, thresh=0.5):
    """Sample all within one standard deviation of mean."""
    confidences = result.get_confidences(thresh)

    if avg is None:
        avg = stats.mean(confidences)
    if stdev is None:
        stdev = stats.stdev(confidences)
    print(f"avg: {avg}, stdev: {stdev}")

    return prob_sample(
        result, round(p * in_range(result, avg - stdev, avg + stdev)), norm, avg, stdev,
    )


def in_range(result, min_val, max_val=1.0):
    """Get the number of elements in a ClassResult above a threshold."""
    return len([res for res in result.get_all() if max_val >= res["conf"] >= min_val])


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
    plt.title(title + f" (n={len(retrain)})")
    plt.show()
