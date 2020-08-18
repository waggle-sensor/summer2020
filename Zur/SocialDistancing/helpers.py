# import packages
import cv2
import numpy as np


# function to transform and calculate bottom points of each bounding box
# input: list of boxes with object IDs, transformation matrix
# output: list of tuples (objectID, (x,y))
def transform_box_points(boundingboxes, matrix):
    bottom_points = []
    for box in boundingboxes:
        # calculate coordinate of bottom center point of each box
        bottom_center = np.array([[[int(box[1][0] + (box[1][2] * 0.5)), int(box[1][1] + box[1][3])]]], dtype="float32")

        # transform point coordinates using transformation matrix, append to list
        warped_pt = cv2.perspectiveTransform(bottom_center, matrix)[0][0]
        pnt = [int(warped_pt[0]), int(warped_pt[1])]
        bottom_points.append((box[0], pnt))

    return bottom_points


# function that that calculates distances between each pair of bottom_points and compares to minimum safe distance
# input: list of boxes with object IDs, list of bottom points with object IDs, minimum safe distance
# output: list of pairs of points tagged with safe boolean, list of pairs of boxes tagged with safe boolean
def violation_detection(boundingboxes, bpoints, safe_dist):
    distance_pairs = []
    box_pairs = []
    boxes = [box[1] for box in boundingboxes]
    bottom_points = [pnt[1] for pnt in bpoints]
    # loop through all combinations of pairs of points
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i < j:
                # calculate distance between the two points
                distance = np.sqrt((bottom_points[j][0] - bottom_points[i][0]) ** 2 +
                                   (bottom_points[j][1] - bottom_points[i][1]) ** 2)

                # compare to min safe distance, tag with appropriate safe boolean
                if distance < safe_dist:
                    safe = False
                    distance_pairs.append([bottom_points[i], bottom_points[j], safe])
                    box_pairs.append([boxes[i], boxes[j], safe])
                else:
                    safe = True
                    distance_pairs.append([bottom_points[i], bottom_points[j], safe])
                    box_pairs.append([boxes[i], boxes[j], safe])

    return distance_pairs, box_pairs


# function that checks each pair of distances and sorts individual distances based on safe boolean
# input: list of distance pairs tagged with safe boolean
# output: number of unsafe and safe points
def get_violation_count(distance_pairs):
    not_safe = []
    safe = []

    # if not safe, add points to red list
    for i in range(len(distance_pairs)):
        if not distance_pairs[i][2]:
            if (distance_pairs[i][0] not in not_safe) and (distance_pairs[i][0] not in safe):
                not_safe.append(distance_pairs[i][0])
            if (distance_pairs[i][1] not in not_safe) and (distance_pairs[i][1] not in safe):
                not_safe.append(distance_pairs[i][1])

    # if safe, add points to green list
    for i in range(len(distance_pairs)):
        if distance_pairs[i][2]:
            if (distance_pairs[i][0] not in not_safe) and (distance_pairs[i][0] not in safe):
                safe.append(distance_pairs[i][0])
            if (distance_pairs[i][1] not in not_safe) and (distance_pairs[i][1] not in safe):
                safe.append(distance_pairs[i][1])

    return len(not_safe), len(safe)


# function that draws red and green rectangles around each person depending on safe boolean
# input: current frame, list of bounding boxes, list of box pairs tagged with safe boolean
# output: current frame with rectangles
def street_output(frame, boundingboxes, box_pairs, violation_count, safe_count):
    box_list = [box[1] for box in boundingboxes]
    # default everyone with a green rectangle
    for i in range(len(box_list)):
        x, y, w, h = box_list[i][:]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i in range(len(box_pairs)):
        # if a pair of people aren't safe, draw red rectangles
        if not box_pairs[i][2]:
            x1, y1, w1, h1 = box_pairs[i][0]
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
            x2, y2, w2, h2 = box_pairs[i][1]
            frame = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

    text1 = "Pedestrians Detected: " + str(len(box_list))
    text2 = "Safe Pedestrians: " + str(safe_count)
    text3 = "Violator Pedestrians: " + str(violation_count)

    cv2.rectangle(frame, (7, 30), (225, 100), (255, 255, 255), -1)

    cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


# function that creates the bird's eye view output for current frame
# input: current frame, list of distance pairs, list of warped box points, bird's eye view window scale
# output: bird's eye view representation for current frame
def bird_output(frame, distance_pairs, scale_w, scale_h,):
    (H, W) = frame.shape[:2]

    # creating white window that is the size of the bird's eye view image
    blank_image = np.zeros((int(H * scale_h), int(W * scale_w), 3), np.uint8)
    blank_image[:] = (255, 255, 255)

    red = []
    green = []

    # if distance is less than minimum safe distance, append coordinate to red list
    for i in range(len(distance_pairs)):
        if not distance_pairs[i][2]:
            if distance_pairs[i][0] not in red and distance_pairs[i][0] not in green:
                red.append(distance_pairs[i][0])
            if distance_pairs[i][1] not in red and distance_pairs[i][1] not in green:
                red.append(distance_pairs[i][1])

            blank_image = cv2.line(blank_image,
                                   (int(distance_pairs[i][0][0] * scale_w), int(distance_pairs[i][0][1] * scale_h)),
                                   (int(distance_pairs[i][1][0] * scale_w), int(distance_pairs[i][1][1] * scale_h)),
                                   (0, 0, 255), 2)

    # if distance is greater than minimum safe distance, append coordinate to green list
    for i in range(len(distance_pairs)):
        if distance_pairs[i][2]:
            if distance_pairs[i][0] not in red and distance_pairs[i][0] not in green:
                green.append(distance_pairs[i][0])
            if distance_pairs[i][1] not in red and distance_pairs[i][1] not in green:
                green.append(distance_pairs[i][1])

    # plot green circles for each coordinate in green list
    for i in green:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 7, (0, 255, 0), -1)

    # plot red circles for each coordinate in red list
    for i in red:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 7, (0, 0, 255), -1)

    return blank_image
