import sys
import os

import numpy as np

from .pts_generator import PtsGenerator
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from geometry import BoundingBox, PointsArray

class GridPtsGenerator(PtsGenerator):
    def __init__(self, step=5) -> None:
        self._step = step
    def gen(self, bb: BoundingBox, _: np.ndarray = None) -> PointsArray:
        px = list(range(int(bb.top_left_pnt.x + 0.5), 
                        int(bb.bottom_right_pnt.x + 0.5) + 1, 
                        self._step))
        py = list(range(int(bb.top_left_pnt.y + 0.5), 
                        int(bb.bottom_right_pnt.y + 0.5) + 1, 
                        self._step))
        px, py = np.meshgrid(px, py)
        px = px.flatten()
        py = py.flatten()
        p0 = np.array([px, py])
        p0 = p0.swapaxes(1,0)
        p0 = p0.astype(np.float32)
        p0 = PointsArray(p0)
        
        return p0
    