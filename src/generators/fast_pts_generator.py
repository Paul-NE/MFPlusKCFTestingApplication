from abc import ABC, abstractmethod
import sys
import os

import numpy as np
import cv2

from geometry import BoundingBox, PointsArray, Point
from .pts_generator import PtsGenerator


class FastPtsGenerator(PtsGenerator):
    def __init__(self):
        # Initialize the FAST feature detector once during instantiation
        self.fast_detector = cv2.FastFeatureDetector_create()

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
        keypoints = self.fast_detector.detect(roi, None)
        # Convert keypoints to Point objects and adjust coordinates relative to the full image
        points = [[kp.pt[0] + x, kp.pt[1] + y] for kp in keypoints]
        points = np.array(points, dtype=np.float32)
        return PointsArray(points)