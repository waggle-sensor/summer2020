import random
import numpy as np


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
