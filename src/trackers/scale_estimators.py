import logging
import inspect

import numpy as np
import cv2

from geometry import BoundingBox, Point, PointsArray
from utils.utils import distances_between_all_points

_scale_logger = logging.getLogger(__name__)

def scale_by_all_distances(previous_points: PointsArray, current_points: PointsArray) -> float:
    previous_points_distances = distances_between_all_points(previous_points)
    current_points_distances = distances_between_all_points(current_points)
    ds = np.sqrt(np.median(current_points_distances / (previous_points_distances + 2**-23)))
    
    f_name = inspect.stack()[0][3]
    _scale_logger.info(f"{f_name}: {ds=}")
    return ds

def estimate_by_affine(previous_points: PointsArray, current_points: PointsArray)  -> float:
    f_name = inspect.stack()[0][3]
    
    matrix, _ = cv2.estimateAffinePartial2D(previous_points, current_points)
    _scale_logger.info(f"{f_name}: {matrix=}")
    
    s_x = np.sqrt(matrix[0,0]**2 + matrix[1,0]**2)
    s_y = np.sqrt(matrix[0,1]**2 + matrix[1,1]**2)
    s = (s_x+s_y) / 2
    _scale_logger.info(f"{f_name}: {s_x=:.3f}, {s_y=:.3f}, {s=:.3f}")
    print(f"{matrix=}")
    print(f"{f_name}: {s_x=:.3f}, {s_y=:.3f}, {s=:.3f}")
    return s
