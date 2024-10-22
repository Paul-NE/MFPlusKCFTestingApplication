from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import cv2

from geometry import BoundingBox
from utils.utils import map_int_list


@dataclass
class Style:
    color: tuple = (0, 0, 255)
    thickness: int = 2


class Mark(ABC):
    """Abstract base class for image marking classes. 

    """
    @abstractmethod
    def __init__(self, data: Any) -> None: 
        super().__init__()
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def draw_self(self, image: np.ndarray) -> np.ndarray:
        pass


class Rectangle(Mark):
    """Rectangle class inherits Mark abstract class
    Rectangle mark for image

    Args:
        Mark (_type_): _description_
    """
    def __init__(self, data: BoundingBox):
        self._data = data
        self._style: Style = Style(
            color=(0, 0, 255),
            thickness=2
        )
    
    @property
    def name(self):
        return "rectangle"
    
    @property
    def style(self):
        return self._style
    
    def draw_self(self, image: np.ndarray):
        top_left_pnt = self._data.top_left_pnt.to_list()
        bottom_right_pnt = self._data.bottom_right_pnt.to_list()
        top_left_pnt = map_int_list(top_left_pnt)
        bottom_right_pnt = map_int_list(bottom_right_pnt)
        new_image = cv2.rectangle(
            image, 
            top_left_pnt, 
            bottom_right_pnt, 
            color=self._style.color, 
            thickness=self._style.thickness
            )
        return new_image


class ImageMarks:
    """Object to store and draw marks
    """
    def __init__(self):
        self._marks: list[Mark] = []
    
    @property
    def marks(self) -> list[Mark]:
        return self._marks.copy()
    
    def draw_all(self, image: np.ndarray):
        image_copy = np.copy(image)
        for mark in self._marks:
            image_copy = mark.draw_self(image_copy)
        return image_copy
    
    def add(self, mark: Mark):
        self._marks.append(mark)
    
    def clear(self):
        self._marks.clear()