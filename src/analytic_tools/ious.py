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
        iou:float = None
        iou_dispertion:float = None
        map30:float = None
        map40:float = None
        map50:float = None
        map75:float = None
        map80:float = None
        map90:float = None
        succes:int = 0
        all_samples:int = 0
        ds_mean_error:float = None
        width_1:float = None
        width_2:float = None
        
    def __init__(self):
        self._box1: list[BoundingBox]=[]
        self._box2: list[BoundingBox]=[]
        self._ious: list[float]=[]
        self._all_samples = 0
    
    def update(self, bbox1: BoundingBox, bbox2: BoundingBox):
        self._all_samples += 1
        self._box1.append(bbox1)
        self._box2.append(bbox2)
        if bbox1 is None or bbox2 is None:
            self._ious.append(None)
            return 
        self._ious.append(get_iou(list(bbox1), list(bbox2)))
    
    def _get_ds(self) -> float:
        boxes = [[box_1, box_2] for box_1, box_2 in zip(self._box1, self._box2) if (box_1 is not None and box_2 is not None)]
        boxes_np = np.array(boxes)
        if len(boxes_np.shape) < 2:
            return
        box_1, box_2 = boxes_np.swapaxes(1, 0)
        ds1 = \
            np.array([w.width for w in box_1[1:] if w is not None]) - \
            np.array([w.width for w in box_1[:-1] if w is not None])
        ds2 = \
            np.array([w.width for w in box_2[1:] if w is not None]) - \
            np.array([w.width for w in box_2[:-1] if w is not None])
        return np.abs(np.abs(ds1) - np.abs(ds2)).mean()
    
    def summary(self) -> Metrics|None:
        summary = {}
        summary["ds_mean_error"] = self._get_ds()
        summary["width_1"] = np.mean([w.width for w in self._box1 if w is not None])
        summary["width_2"] = np.mean([w.width for w in self._box2 if w is not None])
        summary["all_samples"] = self._all_samples
        if not self._ious or all(v is None for v in self._ious):
            return self.Metrics(**summary)
        np_iou = np.array([v for v in self._ious if v is not None])
        summary["iou"] = np.average(np_iou)
        summary["iou_dispertion"] = tstd(np_iou)
        map_percent = [30, 40, 50, 75, 80, 90]
        for percent in map_percent:
            map = np_iou[np_iou > (percent/100)].shape[0] / np_iou.shape[0]
            summary[f"map{percent}"] = map
        summary["succes"] = np_iou.shape[0]
        return self.Metrics(**summary)

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
