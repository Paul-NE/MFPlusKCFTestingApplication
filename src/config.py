from functools import partial
import logging

from pathlib import Path
import cv2

from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.nullable_annotation import NullableAnnotation
from analytic_tools.ious import IOUs
from trackers.forward_backward_pnt_filter import ForwardBackwardPntFilter
from trackers.forward_bachkward_flow import ForwardBachkwardFlow
from trackers.median_flow_tracker import MedianFlowTracker
from generators.smart_grid_pts_generator import SmartGridPtsGenerator
from generators.grid_pts_generator import GridPtsGenerator
from generators.pts_generator import PtsGenerator
from video_processors.webm_video_writer import WebmVideoWriter
from video_processors.video_test import VideoTest
from video_processors.frame_process_skipable_scale_only import FrameProcessorSkipable

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_test_results_write_path = Path(r"/home/poul/test_results.webm")
_test_folder_path = Path(r"/home/poul/temp/Vids/StreetVid_4")
_annotation_name = Path(r"test.txt")
_video_name = Path(r"test.webm")

_max_forward_backward_error = 1
_points_density = 10
_flow_generator = partial(
    cv2.calcOpticalFlowPyrLK,
    nextPts = None,
    winSize  = (11, 11),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1)
)
