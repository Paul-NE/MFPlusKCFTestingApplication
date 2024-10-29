from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np

from trackers.kcf.kcf_refactored import CorellationTracker
from geometry import Point, BoundingBox

def test_corellation():
    tracker = CorellationTracker()
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)
    side_len = 30
    top_left_coords = Point(x=10, y=10)
    bottom_right_coords = Point(
        x=top_left_coords.x+side_len, 
        y=top_left_coords.y+side_len)
    box = BoundingBox(top_left_coords, bottom_right_coords)
    img1[
        box.top_left_pnt.y:box.bottom_right_pnt.y,
        box.top_left_pnt.x:box.bottom_right_pnt.x
    ] = np.ones((box.height, box.width),np.uint8)* 255
    dx = 10
    dy = 10
    img2[
        box.top_left_pnt.y + dy:box.bottom_right_pnt.y + dy,
        box.top_left_pnt.x + dx:box.bottom_right_pnt.x + dx
    ] = np.ones((box.height, box.width),np.uint8)* 255
    
    tracker.init(img1, box)
    result = tracker. update(img2)
    print(result)
    

if __name__=="__main__":
    test_corellation()