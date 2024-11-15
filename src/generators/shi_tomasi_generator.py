from abc import ABC, abstractmethod
from functools import partial
import sys
import os

import numpy as np
import cv2

from geometry import BoundingBox, PointsArray, Point
from .pts_generator import PtsGenerator


class ShiTomasiGenerator(PtsGenerator):
    def __init__(self):
        # Initialize the FAST feature detector once during instantiation
        params = {
            "maxCorners": 100,
            "qualityLevel": 0.01,
            "minDistance": 2 
        }
        self.shi_tomasi_detector = partial(cv2.goodFeaturesToTrack, **params)

    def gen(self, bb: BoundingBox, image: np.ndarray) -> PointsArray:
        # Validate bounding box
        if not bb.validate_box_formation():
            raise ValueError("Invalid bounding box formation: top-left should be above and to the left of bottom-right")

        # Define the region of interest (ROI) using the bounding box
        x, y = bb.top_left_pnt.x, bb.top_left_pnt.y
        w, h = bb.width, bb.height
        x, y = int(x), int(y)
        w, h = round(w), round(h)
        roi = image[y:y+h, x:x+w]

        # Detect keypoints in the region of interest
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except cv2.error as err:
            print(f"ERROR: frame didn`t load properly. {err}")
            return PointsArray(np.array([]))
            
        keypoints = self.shi_tomasi_detector(image=gray_roi)
        points = [[point[0]+x, point[1]+y] for point in keypoints[::,0]]
        points = np.array(points, dtype=np.float32)
        points = PointsArray(points)
        return points