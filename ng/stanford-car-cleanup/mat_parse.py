import mat4py
import os
import cv2


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


def xyxy_to_darknet(img_path, x0, y0, x1, y1):

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    y1 = max(min(y1, h), 0)
    x1 = max(min(x1, w), 0)
    y0 = min(max(y0, 0), h)
    x0 = min(max(x0, 0), w)

    rect_h = y1 - y0
    rect_w = x1 - x0
    x_center = rect_w / 2 + x0
    y_center = rect_h / 2 + y0

    return x_center / w, y_center / h, rect_w / w, rect_h / h


def make_darknet_label(label_path, normalized_coords, class_num):
    coords = map(str, normalized_coords)
    with open(label_path, "w+") as out:
        out.write(f"{class_num} {' '.join(coords)}")


if __name__ == "__main__":
    data = mat4py.loadmat("cars_annos.mat")

    classes = data["class_names"]
    freq = get_freq(data["annotations"]["class"], classes)

    type_freq = dict()
    counted = list()
    make_map = dict()

    for car_type in types:
        type_freq[car_type] = 0
        for make, count in freq.items():
            if f" {car_type} " in make:
                type_freq[car_type] += count
                if make not in counted:
                    counted.append(make)
                    make_map[make] = car_type

    print("\n".join(map(str, make_map.items())))
    print(type_freq)
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)
    with open("output/cars.names", "w+") as out:
        out.write("\n".join(sorted(types)))

    annotations = map(
        dict,
        zip(*[[(k, v) for v in values] for k, values in data["annotations"].items()]),
    )
    for annot in annotations:
        image_path = annot["relative_im_path"].replace("car_ims", "data/images")
        label_path = image_path.replace("images", "labels")[:-4] + ".txt"
        if classes[annot["class"] - 1] not in counted:
            continue
        car_type_num = sorted(types).index(make_map[classes[annot["class"] - 1]])
        coords = xyxy_to_darknet(
            image_path,
            annot["bbox_x1"],
            annot["bbox_y1"],
            annot["bbox_x2"],
            annot["bbox_y2"],
        )
        make_darknet_label(label_path, coords, car_type_num)
