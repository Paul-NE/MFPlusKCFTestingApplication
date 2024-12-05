from pathlib import Path
import logging
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from geometry import BoundingBox
from generators.smart_grid_pts_generator import SmartGridPtsGenerator
from trackers.forward_bachkward_flow import ForwardBachkwardFlow
from trackers.forward_backward_pnt_filter import ForwardBackwardPntFilter
from analytic_tools.annotation_json import AnnotationJson

import cv2
import numpy as np


class LKAndTransformation:
    def __init__(self, points_generator, fb_flow_generator, fb_filter, estimation2d_func):
        self.points_generator = points_generator
        self.fb_flow_generator = fb_flow_generator
        self.fb_filter = fb_filter
        self.estimation2d_func = estimation2d_func
    
    def get_pts(self, frame_t1, frame_t2, bbox) -> tuple[np.ndarray, np.ndarray]:
        start_points = self.points_generator.gen(BoundingBox.generate_from_list(bbox), frame_t1)
        previous_pts, current_pts, backward_pts = self.fb_flow_generator.get_flow(frame_t1, frame_t2, start_points)
        T , _= self.estimation2d_func(previous_pts, current_pts)
        T = np.vstack([T, [0, 0, 1]])
        return current_pts, [transformed_point(point, T) for point in current_pts]


class QuitKeyPressed(Exception):
    pass


@contextmanager
def cv_context_video(path:str):
    """Open and return a video, closes it regardless of the error

    Args:
        path (str): _description_

    Yields:
        _type_: _description_
    """
    capture = cv2.VideoCapture(path)
    try:
        yield capture
    finally:
        if capture is None:
            return
        capture.release()
        cv2.destroyAllWindows()

def get_point_processors():
    """Generates main key-point prosessors

    Returns:
        _type_: _description_
    """
    points_generator = SmartGridPtsGenerator(5)
    flow_generator = partial(
        cv2.calcOpticalFlowPyrLK,
        nextPts = None,
        winSize  = (11, 11),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1)
    )
    fb_flow_generator = ForwardBachkwardFlow(flow_generator)
    fb_filter = ForwardBackwardPntFilter(1)
    return points_generator, fb_flow_generator, fb_filter

def transformed_point(p, t):
    P = [p[0], p[1], 1]
    P_new = t@P
    return P_new

def dists(lk_pts, t_pts) -> list[float]:
    return [((lk_point[:-1]-t_point[:-1])**2).sum()**0.5 for lk_point, t_point in zip(lk_pts, t_pts)]

def color_by_dist(dists) -> list[tuple[int, int, int]]:
    pass

def visualization(frame_t2, lk_pts, t_pts):
    frame_coppy= frame_t2.copy()
    colors = color_by_dist(dists(lk_pts, t_pts))
    for lk_point, t_point in zip(lk_pts, t_pts):
        cv2.circle(frame_coppy, (int(lk_point[0]), int(lk_point[1])), 2, (0, 255, 0), -1)
        cv2.circle(frame_coppy, (int(t_point[0]), int(t_point[1])), 2, (0, 255, 255), -1)
    
    cv2.imshow("frame", frame_coppy)
    k = cv2.waitKey(0)
    if ord("q") == k:
        raise QuitKeyPressed

def run_capture(capture, annotation, point_maker):
    """Running a test

    Args:
        capture (_type_): _description_
        annotation (_type_): _description_
        point_maker (_type_): _description_
    """
    _, frame_t1 = capture.read()
    while frame_t1 is not None:
        bbox = annotation.get_current_box()
        _, frame_t2 = capture.read()
        if frame_t2 is None:
            break
        lk_pts, t_pts = point_maker.get_pts(frame_t1, frame_t2, bbox)
        visualization(frame_t2, lk_pts, t_pts)
        frame_t1 = frame_t2

def main():
    """Setting up a test
    """
    point_maker = LKAndTransformation(*get_point_processors(), cv2.estimateAffinePartial2D)

    dir_name = Path("/home/poul/temp/Vids/annotated/New/StreetVid_2")
    vid = dir_name / "test.webm"
    ann = dir_name / "test.json"
    
    with cv_context_video(vid) as capture:
        annotation = AnnotationJson(ann, capture)
        first = annotation.get_first_frame_index()
        capture.set(cv2.CAP_PROP_FRAME_COUNT, first)
        
        run_capture(capture, annotation, point_maker)


if __name__=="__main__":
    main()