from abc import ABC, abstractmethod
from geometry import BoundingBox

import numpy as np


class Scaler(ABC):
    @abstractmethod
    def init(self, image:np.ndarray, bounding_box: BoundingBox) -> None:
        pass
    @abstractmethod
    def update(self, image: np.ndarray, bounding_box: BoundingBox) -> BoundingBox:
        pass
    @property
    @abstractmethod
    def inited(self) -> bool:
        pass