import os
import lxml.etree as et
from shutil import copyfile

"""
Generate Darknet labels for the VeRi car dataset, as located on LCRC.
Folder path: /lcrc/project/waggle/public_html/private/training_data/vehicle_tracking/VeRi

The contents of the above folder should be downloaded or symbolically linked to
the current directory, in a folder called "data".

Images will be moved into an images folder and a labels folder will be generated,
for both test and train sets.
"""


def parse_labels(label_xml):
    root = et.parse(label_xml).getroot()

    labels = list()

    for item in root.find("Items").findall("Item"):
        label = dict()
        label["path"] = item.get("imageName")
        label["class"] = int(item.get("typeID"))
        labels.append(label)

    return labels


if __name__ == "__main__":
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)

    original_classes = dict()
    for entry in open("data/list_type.txt").read().split("\n"):
        class_id, name = entry.split(" ")
        if name == "pickup":
            name = "cab"
        original_classes[int(class_id)] = name

    new_classes = open("car_types.names").read().split("\n")
    class_freq = {name: 0 for name in new_classes}
    img_count = 0
    for prefix in ("train", "test"):
        labels = parse_labels(f"data/{prefix}_label.xml")

        for label in labels:
            class_name = original_classes[label["class"]]
            if class_name not in new_classes:
                continue
            new_path = f"data/images/IMG{str(img_count).zfill(5)}.jpg"
            class_freq[class_name] += 1
            img_count += 1

            copyfile(f"data/image_{prefix}/{label['path']}", new_path)

            new_class_num = new_classes.index(class_name)
            with open(new_path.replace("images", "labels")[:-4] + ".txt", "w+") as out:
                out.write(f"{new_class_num} 0.5 0.5 1.0 1.0")

    print(class_freq)
