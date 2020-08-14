import numpy as np


# Malisiewicz et al.
def non_max_suppression(boxList, overlapThresh):
    # if no boxes, return empty list
    if len(boxList) == 0:
        return []

    # make sure boxes are all of type float
    if boxList.dtype.kind == "i":
        boxList = boxList.astype("float")

    result = []

    # split list of boxes into 4 lists of coordinates
    x1s = boxList[:, 0]
    y1s = boxList[:, 1]
    x2s = boxList[:, 2]
    y2s = boxList[:, 3]

    # compute list of box areas and sort by bottom-right y-coordinate
    area = (x2s - x1s + 1) * (y2s - y1s + 1)
    indexes = np.argsort(y2s)

    while len(indexes) > 0:
        # append last index from indexes list to result list
        last = len(indexes) - 1
        i = indexes[last]
        result.append(i)

        x1 = np.maximum(x1s[i], x1s[indexes[:last]])
        y1 = np.maximum(y1s[i], y1s[indexes[:last]])
        x2 = np.minimum(x2s[i], x2s[indexes[:last]])
        y2 = np.minimum(y2s[i], y2s[indexes[:last]])

        w = np.maximum(0, x2 - x1 + 1)
        h = np.maximum(0, y2 - y1 + 1)

        overlap = (w * h) / area[indexes[:last]]

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxList[result].astype("int")
