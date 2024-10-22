import sys
import os

import numpy as np

from .pts_generator import PtsGenerator
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from geometry import BoundingBox, PointsArray

class RandomPtsGenerator(PtsGenerator):
    def __init__(self, n_samples=100):
        self._n_samples = n_samples
    
    def gen(self, bb: BoundingBox) -> PointsArray:
        p0 = np.empty((self._n_samples, 2))
        p0[:, 0] = np.random.randint(bb[0], bb[2] + 1, self._n_samples)
        p0[:, 1] = np.random.randint(bb[1], bb[3] + 1, self._n_samples)
        p0 = p0.astype(np.float32)
        p0 = PointsArray(p0)
        return p0
