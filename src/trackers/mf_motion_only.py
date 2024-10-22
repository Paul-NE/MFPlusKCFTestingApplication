from dataclasses import dataclass

import numpy as np
import cv2

from generators.pts_generator import PtsGenerator
from geometry import BoundingBox, Point, PointsArray
from utils.utils import distances_between_all_points, crop_out_of_frame_box, round_box

from .forward_bachkward_flow import ForwardBachkwardFlow
from .forward_backward_pnt_filter import ForwardBackwardPntFilter
from .tracker import AdjustableTracker
from .errors import NotInited


class MFMotionOnly(AdjustableTracker):
    """Object tracker based on optical flow

    Args:
        Tracker (_type_): _description_
    """
    def __init__(
            self, 
            pts_gener:PtsGenerator, 
            fb_filter:ForwardBackwardPntFilter, 
            fb_flow_generator:ForwardBachkwardFlow
        ):
        
        self._prev_image: np.ndarray
        self._prev_box: BoundingBox
        self._inited: bool = False
        
        self._pts_gener: PtsGenerator = pts_gener
        self._fb_filter: ForwardBackwardPntFilter = fb_filter
        self._flow_generator: ForwardBachkwardFlow = fb_flow_generator
    
    @property
    def inited(self):
        return self._inited
    
    def form_new_box(self, previous_box:BoundingBox, dx:float, dy:float) -> BoundingBox:
        return BoundingBox(
            top_left_pnt = Point(
                x = float(previous_box.top_left_pnt.x + dx),
                y = float(previous_box.top_left_pnt.y + dy)
                ),
            bottom_right_pnt=Point(
                x = float(previous_box.bottom_right_pnt.x + dx),
                y = float(previous_box.bottom_right_pnt.y + dy)
                )
            )
    
    def estimate_displacement(self, previous_points: PointsArray, current_points: PointsArray) -> tuple[float, float]:
        dx = np.median(current_points.x - previous_points.x)
        dy = np.median(current_points.y - previous_points.y)
        return dx, dy
    
    def init(self, image:np.ndarray, box:BoundingBox):
        self._prev_image = image
        if isinstance(box, list) or isinstance(box, tuple):
            box = BoundingBox.generate_from_list(box)
        self._prev_box = box
        self._inited = not self._inited
    
    def filter(self, previous_pts:PointsArray, current_pts:PointsArray, backward_pts:PointsArray) -> tuple[PointsArray, PointsArray]:
        # check forward-backward error and min number of points
        p0_bad, p1_bad, p0r_bad = self._fb_filter.filter_bad(
            previous_pnts=previous_pts,
            current_pnts=current_pts,
            backward_pnts=backward_pts
            )
        filtered_current_pts, filtered_backward_pts, _ = self._fb_filter.filter_good(
            previous_pnts=previous_pts,
            current_pnts=current_pts,
            backward_pnts=backward_pts
        )
        return filtered_current_pts, filtered_backward_pts
    
    def update(self, image: np.ndarray):
        if not self._inited:
            raise NotInited(f"Must be inited first")
        
        # sample points inside the bounding box
        previous_pts = self._pts_gener.gen(self._prev_box)
        previous_pts, current_pts, backward_pts = self._flow_generator.get_flow(self._prev_image, current_image=image, previous_pts=previous_pts)
        p0, p1 = self.filter(previous_pts, current_pts, backward_pts)
        
        # can't work with les then 2 points
        # It looks for distance between each 2 points
        if len(p0) < 2:
            return None
        
        dx, dy = self.estimate_displacement(p0, p1)
        bb_curr = self.form_new_box(self._prev_box, dx, dy)
        
        crop_out_of_frame_box(bb_curr, image.shape)
        
        self._prev_image = image
        self._prev_box = bb_curr
        return bb_curr
    
    def adjust_bounding_box(self, bounding_box):
        self._prev_box = bounding_box
