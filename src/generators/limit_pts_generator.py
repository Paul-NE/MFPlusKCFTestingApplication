import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np

from .pts_generator import PtsGenerator
from geometry import BoundingBox, PointsArray, Point


def _process_points(bb: BoundingBox, points:PointsArray, area_limitation:float=1.):
    if points is None or points.size == 0:
        return points
    x, y = points.x, points.y
    w, h = bb.width/2, bb.height/2
    w, h = w*area_limitation, h*area_limitation
    c_x, c_y = bb.center.x, bb.center.y
    sorter = np.abs(y - c_y) < (1 - (x-c_x)**2/w**2)**0.5 * h
    return points[sorter]

def limit_pts_generator(generator: PtsGenerator, area_limitation:float=1.):
    # Modify only the 'greet' method
    if hasattr(generator, "gen"):
        original_gen = getattr(generator, "gen")
        def limited_gen(bb: BoundingBox, image: np.ndarray):
            pts = original_gen(bb, image)
            return _process_points(bb, pts, area_limitation)
        setattr(generator, 'gen', limited_gen)
    
    return generator

