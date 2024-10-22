import sys
import os

import numpy as np

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)
from geometry import PointsArray


class ForwardBackwardPntFilter:
    def __init__(self, fb_max_dist: float) -> None:
        self._fb_max_dist = fb_max_dist
    
    def filter_good(self, previous_pnts: PointsArray, current_pnts: PointsArray, backward_pnts: PointsArray) -> tuple[PointsArray,PointsArray,PointsArray]:
        if previous_pnts.shape[0] == 0 or current_pnts.shape[0] == 0 or backward_pnts.shape[0] == 0:
            return previous_pnts, current_pnts, backward_pnts
        fb_dist = np.abs(previous_pnts - backward_pnts).max(axis=1)
        good = fb_dist < self._fb_max_dist
        return previous_pnts[good], current_pnts[good], backward_pnts[good]
    
    def filter_bad(self, previous_pnts: PointsArray, current_pnts: PointsArray, backward_pnts: PointsArray) -> tuple[PointsArray,PointsArray,PointsArray]:
        if previous_pnts.shape[0] == 0 or current_pnts.shape[0] == 0 or backward_pnts.shape[0] == 0:
            return previous_pnts, current_pnts, backward_pnts
        fb_dist = np.abs(previous_pnts - backward_pnts).max(axis=1)
        bad = fb_dist >= self._fb_max_dist
        return previous_pnts[bad], current_pnts[bad], backward_pnts[bad]
    # def update(self, previous: PointaArray, current: PointaArray, point: list[bool]) -> None:
    #     self._log.good_pnts.previous = 
    #     self._log.good_pnts.current = 
    #     self._log.bad_pnts.previous = 
    #     self._log.bad_pnts.current = 
