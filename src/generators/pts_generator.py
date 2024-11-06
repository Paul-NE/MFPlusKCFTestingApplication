from abc import ABC, abstractmethod
import sys
import os

import numpy as np


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from geometry import BoundingBox, Point


class PtsGenerator(ABC):
    @abstractmethod
    def gen(bb: BoundingBox, image: np.ndarray) -> list[Point]:
        pass
