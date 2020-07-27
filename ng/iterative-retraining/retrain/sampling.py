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

    for img, class_list in images.items():
        for c in class_list:
            if c not in remaining.keys():
                remaining[c] = set()
            remaining[c].add(img)

    desired = [dict() for _ in proportions]
    subsets = [list() for _ in proportions]

    for c, imgs in remaining.items():
        for i, weight in enumerate(proportions):
            desired[i][c] = round(len(imgs) * weight)

    while len(images.keys()) > 0:
        remaining = sort_list_dict(remaining)
        least_freq_label = list(remaining.keys())[0]

        label_imgs = list(remaining[least_freq_label])
        random.shuffle(label_imgs)

        for img in label_imgs:
            get_label_count = lambda x: x[least_freq_label]
            label_counts = [get_label_count(lab) for lab in desired]
            subset_indexes = multi_argmax(label_counts)

            if len(subset_indexes) > 1:
                get_all_label_count = lambda i: sum(desired[i].values())
                all_label_counts = [sum(desired[i].values()) for i in subset_indexes]

                subset_indexes = [
                    subset_indexes[x] for x in multi_argmax(all_label_counts)
                ]
                if len(subset_indexes) > 1:
                    random.shuffle(subset_indexes)

            idx = subset_indexes[0]
            subset = subsets[idx]
            subset.append(img)

            for img_set in remaining.values():
                if img in img_set:
                    img_set.remove(img)

            for c in images[img]:
                desired[idx][c] -= 1

            images.pop(img)
        remaining.pop(least_freq_label)

    return subsets
