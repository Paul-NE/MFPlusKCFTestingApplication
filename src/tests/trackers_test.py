from pathlib import Path
import unittest
import logging
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np

from trackers.kcf.corellation_tracker import CorellationTracker
from geometry import Point, BoundingBox

class TestTrackers(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def test_corellation_tracker(self):
        tracker = CorellationTracker()
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((100, 100), dtype=np.uint8)
        side_len = 30
        accuracy = 0.95
        max_error = side_len * (1 - accuracy)
        top_left_coords = Point(x=20, y=20)
        dx = 11
        dy = 11
        
        bottom_right_coords = Point(
            x=top_left_coords.x+side_len, 
            y=top_left_coords.y+side_len)
        
        box = BoundingBox(top_left_coords, bottom_right_coords)
        box2 = BoundingBox(
            Point(
                x=box.top_left_pnt.x + dx, 
                y=box.top_left_pnt.y + dy),
            Point(
                x=box.bottom_right_pnt.x + dx, 
                y=box.bottom_right_pnt.y + dy)
        )
        
        img1[
            box.top_left_pnt.y:box.bottom_right_pnt.y+1,
            box.top_left_pnt.x:box.bottom_right_pnt.x+1
        ] = np.ones((box.height+1, box.width+1),np.uint8)* 255
        
        img2[
            box2.top_left_pnt.y:box2.bottom_right_pnt.y+1,
            box2.top_left_pnt.x:box2.bottom_right_pnt.x+1
        ] = np.ones((box.height+1, box.width+1),np.uint8)* 255
        
        tracker.init(img1, box)
        result = tracker. update(img2)
        
        self._logger.info(f"test_corellation_tracker: \n{box=}\n{box2=}\n{result=}")
        self.assertAlmostEqual(box2.center.x, result.center.x, delta=max_error)
        self.assertAlmostEqual(box2.center.y, result.center.y, delta=max_error)
    

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
    unittest.main()