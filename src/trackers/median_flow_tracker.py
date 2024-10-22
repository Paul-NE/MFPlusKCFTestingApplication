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


class MedianFlowTracker(AdjustableTracker):
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
        self._ds_factor = 0.95
        
        self._prev_image: np.ndarray
        self._prev_box: BoundingBox
        self._inited: bool = False
        
        self._pts_gener: PtsGenerator = pts_gener
        self._fb_filter: ForwardBackwardPntFilter = fb_filter
        self._flow_generator: ForwardBachkwardFlow = fb_flow_generator
    
    @property
    def inited(self):
        return self._inited
    
    def form_new_box(self, previous_box:BoundingBox, dx:float, dy:float, dx_scale:float, dy_scale:float) -> BoundingBox:
        return BoundingBox(
            top_left_pnt = Point(
                x = float(previous_box.top_left_pnt.x + dx - dx_scale),
                y = float(previous_box.top_left_pnt.y + dy - dy_scale)
                ),
            bottom_right_pnt=Point(
                x = float(previous_box.bottom_right_pnt.x + dx + dx_scale),
                y = float(previous_box.bottom_right_pnt.y + dy + dy_scale)
                )
            )
    
    def estimate_displacement(self, previous_points: PointsArray, current_points: PointsArray) -> tuple[float, float]:
        dx = np.median(current_points.x - previous_points.x)
        dy = np.median(current_points.y - previous_points.y)
        return dx, dy
    
    def estimate_scale(self, previous_points: PointsArray, current_points: PointsArray) -> float:
        previous_points_distances = distances_between_all_points(previous_points)
        current_points_distances = distances_between_all_points(current_points)
        ds = np.sqrt(np.median(current_points_distances / (previous_points_distances + 2**-23)))
        ds = (1.0 - self._ds_factor) + self._ds_factor * ds # TODO self._ds_factor
        return ds
    
    def init(self, image:np.ndarray, box:BoundingBox):
        self._prev_image = image
        if isinstance(box, list) or isinstance(box, tuple):
            box = BoundingBox.generate_from_list(box)
        self._prev_box = box
        self._inited = not self._inited
    
    def estimate_box_change(self, p0, p1):
        dx, dy = self.estimate_displacement(p0, p1)
        ds = self.estimate_scale(p0, p1)

        # update bounding box
        dx_scale = (ds - 1.0) * self._prev_box.width / 2
        dy_scale = (ds - 1.0) * self._prev_box.height / 2
        
        print(
            f"{self._prev_box=}\n",
            f"{ds=}\n",
            f"{dx=}\n", 
            f"{dy=}\n", 
            f"{dx_scale=}\n", 
            f"{dy_scale=}\n"
        )
        return dx, dy, dx_scale, dy_scale
    
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
        
        dx, dy, dx_scale, dy_scale = self.estimate_box_change(p0, p1)
        bb_curr = self.form_new_box(self._prev_box, dx, dy, dx_scale, dy_scale)
        
        crop_out_of_frame_box(bb_curr, image.shape)
        
        self._prev_image = image
        self._prev_box = bb_curr
        
        return round_box(bb_curr)
    
    def adjust_bounding_box(self, bounding_box):
        self._prev_box = bounding_box
