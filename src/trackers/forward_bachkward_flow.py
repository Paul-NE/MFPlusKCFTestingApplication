from typing import Callable
import numpy as np

from geometry import PointsArray


class ForwardBachkwardFlow:
    def __init__(
            self, 
            pts_tracking_func:Callable[[np.ndarray, np.ndarray, PointsArray], tuple[np.ndarray, np.ndarray, np.ndarray]]
        ):
        self._pts_tracking_func = pts_tracking_func
    
    def _forward(
            self, 
            previous_image: np.ndarray, 
            current_image: np.ndarray, 
            previous_pts: PointsArray
        ):
        """Calculates optical flow forward

        Args:
            previous_image (np.ndarray): image from t-1
            current_image (np.ndarray): image from t
            previous_pts (PointsArray): points in t-1 moment

        Returns:
            _type_: _description_
        """
        current_points, state, _ = self._pts_tracking_func(previous_image, current_image, previous_pts)
        current_points = PointsArray(current_points)
        indx = np.where(state == 1)[0]
        previous_pts = previous_pts[indx, :]
        current_points = current_points[indx, :]
        return previous_pts, current_points
    
    def _backward(
            self, 
            previous_image: np.ndarray, 
            current_image: np.ndarray,
            previous_pts: PointsArray,
            current_pts: PointsArray
        ):
        backward_pts, _, _ = self._pts_tracking_func(current_image, previous_image, current_pts)
        if backward_pts is None:
            backward_pts= PointsArray(np.array([], dtype=np.float32))
        return previous_pts, current_pts, backward_pts
    
    def get_flow(
            self, 
            previous_image: np.ndarray, 
            current_image: np.ndarray, 
            previous_pts: PointsArray
        ):
        previous_pts, current_pts = self._forward(previous_image, current_image, previous_pts)
        previous_pts, current_pts, backward_pts = self._backward(previous_image, current_image, previous_pts, current_pts)
        return previous_pts, current_pts, backward_pts