from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from video_processors.video_test import VideoTest
from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.nullable_annotation import NullableAnnotation
from analytic_tools.ious import IOUs
from trackers.forward_backward_pnt_filter import ForwardBackwardPntFilter
from trackers.forward_bachkward_flow import ForwardBachkwardFlow
from trackers.median_flow_tracker import MedianFlowTracker
from trackers.mf_scaler import MFScaler
from trackers.mf_motion_only import MFMotionOnly
from generators.smart_grid_pts_generator import SmartGridPtsGenerator
from generators.grid_pts_generator import GridPtsGenerator
from generators.pts_generator import PtsGenerator
from video_processors.webm_video_writer import WebmVideoWriter
from video_processors.video_test import VideoTest
from video_processors.process_universal import FrameProcessorUni
from trackers.kcf.kcf import *
from app_gui.utils import load_json


def main(test_folder_path: Path, config_optioins: dict):
    debug = KCFDebugParams(showFeatures=False, showAlphaf=False, showTmpl=False, saveTrackerParams=False)
    flags = KCFFlags(hog=False, fixed_window=True, multiscale=False, normalizeShift=False, smoothMotion=True)
    train_params = KCFTrainParams(tmplsz=64, lambdar=0.0001, padding=2.5, output_sigma_factor=0.125, sigma=0.2,
                                    interp_factor=0.001)
    hog = KCFHogParams(NUM_SECTOR=9, cell_size=4)
    params = KCFParams(flags=flags, hog=hog, debug=debug, train=train_params)
    tracker_kcf = KCFTrackerNormal(params)
    
    _annotation_path = test_folder_path / config._annotation_name
    video_path = test_folder_path / config._video_name
    video_test_options = VideoTest.Options()
    _frame_process_options = FrameProcessorUni.Options(
        skip_first_n_frames = config_optioins["frame_processor_options"]["skip_first_n_frames"], 
        wait_key_value = config_optioins["frame_processor_options"]["wait_key_value"], 
        manual_roi_selection = config_optioins["frame_processor_options"]["manual_roi_selection"], 
        start_paused_value = config_optioins["frame_processor_options"]["start_paused_value"]
        )

    # _annotation = NullableAnnotation(_annotation_path)
    _annotation = None
    _video_writer = WebmVideoWriter(config._test_results_write_path)
    _analytics_engine = IOUs()

    pts_gener = SmartGridPtsGenerator(config._points_density)
    fb_flow_generator = ForwardBachkwardFlow(config._flow_generator)
    fb_filter = ForwardBackwardPntFilter(config._max_forward_backward_error)

    tracker = MFMotionOnly(
        pts_gener,
        fb_filter,
        fb_flow_generator
    )
    scaler = MFScaler(
            pts_gener,
            fb_filter,
            fb_flow_generator
        )

    operation = FrameProcessorUni(
            tracker=tracker_kcf,
            scaler=scaler,
            annotation=_annotation, 
            options=_frame_process_options,
            video_writer=_video_writer
            # analytics_engine=_analytics_engine,
        )
    oper = operation
    
    VideoTest(
        video=video_path, 
        operation=oper,
        options=video_test_options
    ).run()
    
    print(_analytics_engine.avarage)


if __name__ == "__main__":
    params = load_json("./config_.json")
    main(Path(r"/home/poul/temp/Vids/StreetVid_4"), params)