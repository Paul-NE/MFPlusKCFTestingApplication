import numpy as np

from generators.pts_generator import PtsGenerator
from geometry import BoundingBox, Point, PointsArray
from utils.utils import distances_between_all_points, crop_out_of_frame_box

from .forward_bachkward_flow import ForwardBachkwardFlow
from .forward_backward_pnt_filter import ForwardBackwardPntFilter
from .scaler import Scaler
from .errors import NotInited


class MFScaler(Scaler):
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
        self._inited: bool = False
        
        self._pts_gener: PtsGenerator = pts_gener
        self._fb_filter: ForwardBackwardPntFilter = fb_filter
        self._flow_generator: ForwardBachkwardFlow = fb_flow_generator
    
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
    
    def estimate_scale(self, previous_points: PointsArray, current_points: PointsArray) -> float:
        previous_points_distances = distances_between_all_points(previous_points)
        current_points_distances = distances_between_all_points(current_points)
        ds = np.sqrt(np.median(current_points_distances / (previous_points_distances + 2**-23)))
        return ds
    
    def init(self, image:np.ndarray, box:BoundingBox):
        self._prev_image = image
        if isinstance(box, list) or isinstance(box, tuple):
            box = BoundingBox.generate_from_list(box)
        self._inited = not self._inited
    
    def estimate_box_change(self, p0, p1, current_box: BoundingBox):
        ds = self.estimate_scale(p0, p1)

        # update bounding box
        dx_scale = (ds - 1.0) * current_box.width / 2
        dy_scale = (ds - 1.0) * current_box.height / 2
        
        return dx_scale, dy_scale
    
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
    
    def update(self, image: np.ndarray, current_box:BoundingBox):
        if not self._inited:
            raise NotInited(f"Must be inited first")
        
        # sample points inside the bounding box
        previous_pts = self._pts_gener.gen(current_box)
        previous_pts, current_pts, backward_pts = self._flow_generator.get_flow(self._prev_image, current_image=image, previous_pts=previous_pts)
        p0, p1 = self.filter(previous_pts, current_pts, backward_pts)
        
        # can't work with les then 2 points
        # It looks for distance between each 2 points
        if len(p0) < 2:
            return None
        
        dx_scale, dy_scale = self.estimate_box_change(p0, p1, current_box)
        bb_new = self.form_new_box(current_box, dx_scale, dy_scale)
        
        crop_out_of_frame_box(bb_new, image.shape)
        
        self._prev_image = image
        return bb_new

