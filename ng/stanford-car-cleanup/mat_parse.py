import mat4py


def get_freq(class_arr):
    counts = dict()

    for c in class_arr:
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    return counts


def main():
    data = mat4py.loadmat("cars_annos.mat")
    freq = get_freq(data["annotations"]["class"])
    print(sorted([v for k, v in freq.items()]))


if __name__ == "__main__":
    main()
