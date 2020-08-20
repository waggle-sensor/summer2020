from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        # initialize unique object ID
        self.nextObjectID = 0

        # initialize three ordered dictionaries:
        #   1. Key = objectID, Val = centroid coordinates
        #   2. Key = objectID, Val = number of consecutive frames this objectID has been marked "disappeared"
        #   3. Key = objectID, Val = coordinates of that objects bounding box (x, y, w, h)
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()

        # store the number of frames an object can be marked "disappeared" before it is deregistered
        self.maxDisappeared = maxDisappeared

    # registering a new object using next available ID to store its centroid
    def register(self, centroid, bbox):
        self.nextObjectID += 1
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.bbox[self.nextObjectID] = bbox

    # de-registering an object by deleting object ID from both dictionaries
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]

    # update state every frame
    def update(self, boxes):
        # if no current bounding boxes, de-register any object past limit and return early
        if len(boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.bbox

        # calculate centroid of each bounding box and organize in a numpy array
        inputCentroids = np.zeros((len(boxes), 2), dtype="int")
        inputRects = []
        for(i, (x, y, w, h)) in enumerate(boxes):
            cX = int(x + (w * 0.5))
            cY = int(y + (h * 0.5))
            inputCentroids[i] = (cX, cY)
            inputRects.append(boxes[i])

        # if currently not tracking any objects, register the centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])

        # otherwise, objects are being tracked so need to update centroids
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # find smallest value in each row, sort row indexes by minimum values
            rows = D.min(axis=1).argsort()

            # find smallest value in each column and sort based on ordered rows
            cols = D.argmin(axis=1)[rows]

            # keep track of rows and columns already examined
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # update centroid and disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # compute unexamined rows and columns
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that there are more object centroids than input centroids
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # in the event that there are more input centroids than object centroids
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        return self.bbox
