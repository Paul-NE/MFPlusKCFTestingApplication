from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: float
    y: float
    
    def __sub__(self, other):
        return Point(
            x = self.x - other.x,
            y = self.y - other.y
            )
    
    def to_list(self) -> list:
        return [self.x, self.y]


class PointsArray(np.ndarray):
    def __new__(cls, array: np.ndarray):
        return super().__new__(cls, shape=array.shape, dtype=array.dtype) 
    
    def __init__(self, array: np.ndarray):
        super().__init__()
        self[...] = array[...]
    
    def get_pnt(self, index: int):
        if self.shape[1] != 2:
            raise IndexError()
        assert self.shape[1] == 2, "Can't make 2D point of more then 2 coordinates"
        return Point(*self[index])
    @property
    def x(self):
        return self[:, 0]
    @property
    def y(self):
        return self[:, 1]


@dataclass
class PointPairs:
    previous: PointsArray
    current: PointsArray
