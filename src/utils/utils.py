from copy import deepcopy
import sys
import os

import numpy as np

from geometry import BoundingBox


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from geometry import PointsArray, BoundingBox


def map_int_list(arr: list) -> list[int]:
    return list(map(int, arr))

def all_combinations_of_2_elements(number_of_elements: int) -> tuple[list, list]:
    return np.triu_indices(number_of_elements, k=1)

def distances_between_all_points(points: PointsArray) -> np.ndarray:
    i, j = all_combinations_of_2_elements(len(points))
    distances =  np.sum((points[i] - points[j]) ** 2, axis=1) # TODO refactor
    return np.array(distances)

def crop_out_of_frame_box(box: BoundingBox, frame_shape: tuple) -> None:
    box.top_left_pnt.x = min(max(0, box.top_left_pnt.x), frame_shape[1])
    box.top_left_pnt.y = min(max(0, box.top_left_pnt.y), frame_shape[0])
    box.bottom_right_pnt.x = min(max(0, box.bottom_right_pnt.x), frame_shape[1])
    box.bottom_right_pnt.y = min(max(0, box.bottom_right_pnt.y), frame_shape[0])

def round_box(bbox: BoundingBox)->BoundingBox:
    box = deepcopy(bbox)
    box.top_left_pnt.x = round(bbox.top_left_pnt.x)
    box.top_left_pnt.y = round(bbox.top_left_pnt.y)
    box.bottom_right_pnt.x = round(bbox.bottom_right_pnt.x)
    box.bottom_right_pnt.y = round(bbox.bottom_right_pnt.y)
    return box

def iou(box1: BoundingBox, box2: BoundingBox) -> float|None:
    if not(box1.validate_box_formation() and box2.validate_box_formation()):
        return None

    # determine the coordinates of the intersection rectangle
    x_left = max(box1.top_left_pnt.x, box2.top_left_pnt.x)
    y_top = max(box1.top_left_pnt.x, box2.top_left_pnt.y)
    x_right = min(box1.bottom_right_pnt.x, box2.bottom_right_pnt.x)
    y_bottom = min(box1.bottom_right_pnt.y, box2.bottom_right_pnt.y)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    box1_area = box1.width * box1.height
    box2_area = box2.width * box1.height

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


