# from .bounding_box import BoundingBox
from.points import Point

from dataclasses import dataclass
from typing import Self


@dataclass
class BoundingBox:
    top_left_pnt: Point
    bottom_right_pnt : Point
    
    @property
    def width(self):
        return self.bottom_right_pnt.x - self.top_left_pnt.x
    
    @property
    def height(self):
        return self.bottom_right_pnt.y - self.top_left_pnt.y
    
    @property
    def center(self):
        return Point(
            (self.bottom_right_pnt.x + self.top_left_pnt.x) / 2,
            (self.bottom_right_pnt.y + self.top_left_pnt.y) / 2
        )
    
    def validate_box_formation(self) -> bool:
        """Checks if top left and bottob right points are mixed up 

        Returns:
            bool: True/False
        """
        if self.bottom_right_pnt.x <= self.top_left_pnt.x:
            return False
        if self.bottom_right_pnt.y <= self.top_left_pnt.y:
            return False
        return True
    
    def __iter__(self):
        yield self.top_left_pnt.x
        yield self.top_left_pnt.y
        yield self.bottom_right_pnt.x
        yield self.bottom_right_pnt.y
    
    @staticmethod
    def generate_from_list(arr: list) -> Self:
        """Factory method to convert list to BoundingBox
        
        Args:
            arr (list): list of 4 numeric elements in folowing format:
            arr[0] - top left x
            arr[1] - top left y
            arr[2] - bottom right x
            arr[3] - bottom right y

        Returns:
            BoundingBox: _description_
        """
        assert len(arr) == 4, "Needs exectly 4 elements"
        return BoundingBox(
            Point(
                x = arr[0],
                y = arr[1]
            ),
            Point(
                x = arr[2],
                y = arr[3]
            )
        )
