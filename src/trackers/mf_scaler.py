from dataclasses import dataclass
from typing import Callable
import logging

import numpy as np
import cv2

from marks.marks import ImageMarks, Vector
from video_processors.window_manager import Window
from generators.pts_generator import PtsGenerator
from geometry import BoundingBox, Point, PointsArray
from utils.utils import distances_between_all_points, crop_out_of_frame_box

from .forward_bachkward_flow import ForwardBachkwardFlow
from .forward_backward_pnt_filter import ForwardBackwardPntFilter
from .scaler import Scaler
from .errors import NotInited

def _estimate_scale(previous_points: PointsArray, current_points: PointsArray) -> float:
    previous_points_distances = distances_between_all_points(previous_points)
    current_points_distances = distances_between_all_points(current_points)
    ds = np.sqrt(np.median(current_points_distances / (previous_points_distances + 2**-23)))
    return ds

class MFScaler(Scaler):
    """Object tracker based on optical flow

    Args:
        Tracker (_type_): _description_
    """
    @dataclass
    class Options:
        debug_visualization: bool = False
        debug_key_points_window: str = "key_points"
    def __init__(
            self, 
            pts_gener:PtsGenerator, 
            fb_filter:ForwardBackwardPntFilter, 
            fb_flow_generator:ForwardBachkwardFlow,
            scale_estimator: Callable[[PointsArray, PointsArray], float],
            min_keypoints_number: int = 10,
            options:Options|None=None
        ):
        self._debug_windows:dict[str, Window] = {}
        self._key_point_marker = ImageMarks()
        self._prev_image: np.ndarray
        self._inited: bool = False
        
        self._pts_gener: PtsGenerator = pts_gener
        self._fb_filter: ForwardBackwardPntFilter = fb_filter
        self._flow_generator: ForwardBachkwardFlow = fb_flow_generator
        self._estimate_scale = scale_estimator
        self._min_keypoint = min_keypoints_number
        
        self._options = options if options is not None else self.Options()
        if self._options.debug_visualization:
            self._debug_windows[self._options.debug_key_points_window] = Window(self._options.debug_key_points_window)
        
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.info(f"Initialized with options: {self._options}")
    
    @property
    def inited(self):
        return self._inited
    
    def form_new_box(self, current_box: BoundingBox, dx_scale:float, dy_scale:float) -> BoundingBox:
        width, height = current_box.width, current_box.height
        current_center = current_box.center
        return BoundingBox(
            top_left_pnt = Point(
                x = float(current_center.x - width/2 - dx_scale),
                y = float(current_center.y - height/2 - dy_scale)
                ),
            bottom_right_pnt=Point(
                x = float(current_center.x + width/2 + dx_scale),
                y = float(current_center.y + height/2 + dy_scale)
                )
            )
    
    def init(self, image:np.ndarray, box:BoundingBox):
        self._prev_image = image
        if isinstance(box, list) or isinstance(box, tuple):
            box = BoundingBox.generate_from_list(box)
        self._inited = not self._inited
    
    def _estimate_box_change(self, p0, p1, current_box: BoundingBox):
        ds = self._estimate_scale(p0, p1)

        # update bounding box
        dx_scale = (ds - 1.0) * current_box.width / 2
        dy_scale = (ds - 1.0) * current_box.height / 2
        
        return dx_scale, dy_scale
    
    def _filter(self, previous_pts:PointsArray, current_pts:PointsArray, backward_pts:PointsArray) -> tuple[PointsArray, PointsArray]:
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
        
        self._logger.info(f"Pts forward-backward filter not passed: {len(p0_bad)}")
        self._logger.info(f"Pts forward-backward filter passed: {len(filtered_current_pts)}")
        
        if not self._options.debug_visualization:
            return filtered_current_pts, filtered_backward_pts
        
        self._write_debug_point_marks(filtered_current_pts, filtered_backward_pts, (0, 255, 0))
        self._write_debug_point_marks(p0_bad, p1_bad, (0, 0, 255))
        
        return filtered_current_pts, filtered_backward_pts
    
    def _write_debug_point_marks(self, filtered_backward_pts: PointsArray, filtered_current_pts:PointsArray, color:tuple[int, int, int]):
        for previous_point_x, previous_point_y, current_point_x, current_point_y in zip(
            filtered_backward_pts.x, 
            filtered_backward_pts.y,
            filtered_current_pts.x, 
            filtered_current_pts.y
        ):
            mark = Vector((
                Point(previous_point_x, previous_point_y),
                Point(current_point_x, current_point_y)
            ))
            mark.style.color = color
            self._key_point_marker.add(mark)
    
    def update(self, image: np.ndarray, current_box:BoundingBox):
        if not self._inited:
            raise NotInited(f"Must be inited first")
        
        # sample points inside the bounding box
        previous_pts = self._pts_gener.gen(current_box, image)
        try:
            previous_pts, current_pts, backward_pts = self._flow_generator.get_flow(self._prev_image, current_image=image, previous_pts=previous_pts)
        except cv2.error as e:
            self._logger.warning(f"Cv2 error code {e.code}. Could not generate flow")
            return None
        p0, p1 = self._filter(previous_pts, current_pts, backward_pts)
        
        if len(p0) < self._min_keypoint:
            self._logger("Not enought ponts! Returning none")
            return None
        
        dx_scale, dy_scale = self._estimate_box_change(p0, p1, current_box)
        bb_new = self.form_new_box(current_box, dx_scale, dy_scale)
        
        crop_out_of_frame_box(bb_new, image.shape)
        
        self._update_debug_windows(image)
        self._prev_image = image
        return bb_new
    
    def _update_debug_windows(self, image:np.ndarray):
        self._debug_windows[self._options.debug_key_points_window].frame = self._key_point_marker.draw_all(image)
        self._key_point_marker.clear()
    
    def get_debug_windows(self):
        return list(self._debug_windows.values())

