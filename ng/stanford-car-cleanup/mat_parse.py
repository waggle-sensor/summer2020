import mat4py


def get_freq(class_arr, classes):
    counts = dict()

    for c in class_arr:
        c = classes[c - 1]
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    return counts


types = [
    "Sedan",
    "Hatchback",
    "SUV",
    "Coupe",
    "Van",
    "Convertible",
    "Wagon",
    "Minivan",
    "Cab",
]


def main():
    data = mat4py.loadmat("cars_annos.mat")

    classes = data["class_names"]
    freq = get_freq(data["annotations"]["class"], classes)

    type_freq = dict()
    counted = list()
    for car_type in types:
        type_freq[car_type] = 0
        for make, count in freq.items():
            if f" {car_type} " in make:
                type_freq[car_type] += count
                if make not in counted:
                    counted.append(make)

    print("\n".join(sorted(counted)))


if __name__ == "__main__":
    main()
