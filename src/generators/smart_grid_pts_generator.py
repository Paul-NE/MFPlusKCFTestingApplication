import sys
import os

import numpy as np

from .pts_generator import PtsGenerator
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from geometry import BoundingBox, PointsArray

class SmartGridPtsGenerator(PtsGenerator):
    def __init__(self, side_points:int=10) -> None:
        self._side_points = side_points
    def gen(self, bb: BoundingBox) -> PointsArray:
        w_step = int(bb.width // self._side_points)
        h_step = int(bb.height // self._side_points)
        w_step = max(w_step, 1)
        h_step = max(h_step, 1)
        min_step = min(w_step, h_step)
        max_step = max(w_step, h_step)
        step = max_step
        px = list(range(int(bb.top_left_pnt.x + 0.5), 
                        int(bb.bottom_right_pnt.x + 0.5) + 1, 
                        w_step))
        py = list(range(int(bb.top_left_pnt.y + 0.5), 
                        int(bb.bottom_right_pnt.y + 0.5) + 1, 
                        h_step))
        px, py = np.meshgrid(px, py)
        px = px.flatten()
        py = py.flatten()
        p0 = np.array([px, py])
        p0 = p0.swapaxes(1,0)
        p0 = p0.astype(np.float32)
        p0 = PointsArray(p0)
        
        return p0
    