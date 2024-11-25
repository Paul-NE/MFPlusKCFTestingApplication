from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import tstd

from geometry import BoundingBox


def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        bb1 = {'x1': bb1[0], 
            'x2': bb1[2], 
            'y1': bb1[1], 
            'y2': bb1[3]}
        bb2 = {'x1': bb2[0], 
            'x2': bb2[2], 
            'y1': bb2[1],
            'y2': bb2[3]}
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou


class Analytics:
    @dataclass
    class Metrics:
        iou:float
        iou_dispertion:float
        map30:float
        map40:float
        map50:float
        map75:float
        map80:float
        map90:float
        
    def __init__(self):
        self._box1: list[BoundingBox]=[]
        self._box2: list[BoundingBox]=[]
        self._ious: list[float]=[]
    
    def update(self, bbox1: BoundingBox, bbox2: BoundingBox):
        self._box1.append(bbox1)
        self._box2.append(bbox2)
        self._ious.append(get_iou(list(bbox1), list(bbox2)))
    
    def summary(self) -> Metrics:
        summary = {}
        np_iou = np.array(self._ious)
        summary["iou"] = np.average(np_iou)
        summary["iou_dispertion"] = tstd(np_iou)
        map_percent = [30, 40, 50, 75, 80, 90]
        for percent in map_percent:
            map = np_iou[np_iou > (percent/100)].shape[0] / np_iou.shape[0]
            summary[f"map{percent}"] = map
        return summary

class IOUs(list):
    def iou_append(self, bbox1: BoundingBox, bbox2: BoundingBox):
        self.append(self.get_iou(list(bbox1), list(bbox2)))
    
    @property
    def avarage(self):
        if len(self):
            return sum(self) / len(self)
        else:
            return 0
    
    @staticmethod
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        bb1 = {'x1': bb1[0], 
            'x2': bb1[2], 
            'y1': bb1[1], 
            'y2': bb1[3]}
        bb2 = {'x1': bb2[0], 
            'x2': bb2[2], 
            'y1': bb2[1],
            'y2': bb2[3]}
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
