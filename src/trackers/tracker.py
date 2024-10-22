from abc import ABC, abstractmethod
from geometry import BoundingBox

import numpy as np


class Tracker(ABC):
    @abstractmethod
    def init(self, image:np.ndarray, bounding_box: BoundingBox) -> None:
        pass
    @abstractmethod
    def update(self, image: np.ndarray) -> BoundingBox:
        pass
    @property
    @abstractmethod
    def inited(self) -> bool:
        pass

class AdjustableTracker(Tracker):
    @abstractmethod
    def adjust_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        pass