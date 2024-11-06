import unittest
import logging
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import cv2

from geometry import BoundingBox, Point
from generators.fast_pts_generator import FastPtsGenerator  # Adjust with the actual module name

class TestFastPtsGenerator(unittest.TestCase):
    def setUp(self):
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        # Create a mock image with some features (white rectangle on black background)
        self.image = cv2.imread(f"{os.path.dirname(os.path.realpath(__file__))}/static/test.png")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Create a mock bounding box (top-left point at (10, 10) and bottom-right point at (50, 50))
        self.bb = BoundingBox(
            top_left_pnt=Point(x=10, y=10),
            bottom_right_pnt=Point(x=90, y=90)
        )
        
        # Create the FastPtsGenerator instance
        self.generator = FastPtsGenerator()

    def test_gen_valid_bounding_box(self):
        points = self.generator.gen(self.bb, self.image)

        # Log the points detected
        self._logger.debug("Detected points:")
        for point in points:
            self._logger.debug(f"Point: ({point.x}, {point.y})")
        
        # Verify that points is a list
        self.assertIsInstance(points, list)

        # Verify that each point is a Point object
        for point in points:
            self.assertIsInstance(point, Point)

        # Ensure that points are within the bounding box region
        for point in points:
            self.assertTrue(self.bb.top_left_pnt.x <= point.x <= self.bb.bottom_right_pnt.x)
            self.assertTrue(self.bb.top_left_pnt.y <= point.y <= self.bb.bottom_right_pnt.y)

    def test_empty_image(self):
        # Create a completely black image with no features
        empty_image = np.zeros((100, 100), dtype=np.uint8)

        points = self.generator.gen(self.bb, empty_image)

        # In an empty image, there should be no points detected
        self.assertEqual(len(points), 0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()