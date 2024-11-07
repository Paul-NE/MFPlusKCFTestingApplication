from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.nullable_annotation import NullableAnnotation
from analytic_tools.ious import IOUs

from trackers.forward_backward_pnt_filter import ForwardBackwardPntFilter
from trackers.forward_bachkward_flow import ForwardBachkwardFlow
from trackers.median_flow_tracker import MedianFlowTracker
from trackers.mf_scaler import MFScaler
from trackers.mf_motion_only import MFMotionOnly
from trackers.kcf.corellation_tracker import CorellationTracker
from trackers.kcf.kcf import *
from trackers.scale_estimators import estimate_by_affine, scale_by_all_distances

from generators.smart_grid_pts_generator import SmartGridPtsGenerator
from generators.grid_pts_generator import GridPtsGenerator
from generators.pts_generator import PtsGenerator
from generators.fast_pts_generator import FastPtsGenerator

from video_processors.video_test import VideoTest
from video_processors.webm_video_writer import WebmVideoWriter
from video_processors.video_test import VideoTest
from video_processors.process_universal import FrameProcessorUni

from app_gui.utils import load_json


def init_median_flow_scaler(config_options:dict) -> MFScaler:
    if not config_options["components"]["scale"]:
        return
    
    # pts_gener = SmartGridPtsGenerator(config._points_density)
    pts_gener = FastPtsGenerator()
    fb_flow_generator = ForwardBachkwardFlow(config._flow_generator)
    fb_filter = ForwardBackwardPntFilter(config._max_forward_backward_error)
    scale_estimator = estimate_by_affine
    
    options =MFScaler.Options(debug_visualization=True) 
    scaler = MFScaler(
        pts_gener,
        fb_filter,
        fb_flow_generator,
        scale_estimator,
        options=options,
    )
    return scaler

def init_kcf(config_options: dict):
    debug = KCFDebugParams(
        showFeatures=False, 
        showAlphaf=False, 
        showTmpl=False, 
        saveTrackerParams=False
        )
    flags = KCFFlags(
        hog=False, 
        fixed_window=True, 
        multiscale=False, 
        normalizeShift=False, 
        smoothMotion=False
        )
    train_params = KCFTrainParams(
        tmplsz=64, 
        lambdar=0.0001, 
        padding=2.5, 
        output_sigma_factor=0.125, 
        sigma=0.2, 
        interp_factor=0.001
        )
    hog = KCFHogParams(NUM_SECTOR=9, cell_size=4)
    params = KCFParams(flags=flags, hog=hog, debug=debug, train=train_params)
    return KCFTrackerNormal(params)

def init_corellation(config_options: dict):
    configs = config_options["corellation_tracker"]
    options = CorellationTracker.Options(
        *configs["options"])
    math_params = CorellationTracker.MathParameters(*configs["math_parameters"])
    return CorellationTracker(
        math_parameters=math_params,
        options=options
        )

def main(test_folder_path: Path, config_options: dict):
    video_path = test_folder_path / config._video_name
    video_test_options = VideoTest.Options()
    frame_process_options = FrameProcessorUni.Options(
        **config_options["frame_processor_options"]
        )

    _annotation_path = test_folder_path / config._annotation_name
    # _annotation = NullableAnnotation(_annotation_path)
    _annotation = None
    
    video_writer = WebmVideoWriter(config._test_results_write_path) if config_options["components"]["write_video"] else None
    _analytics_engine = IOUs()
    
    scaler = init_median_flow_scaler(config_options)
    corellation_tracker = init_corellation(config_options)
    scaler_windows = scaler.get_debug_windows()
    tracker_windows = corellation_tracker.get_debug_windows()
    windows = scaler_windows + tracker_windows

    operation = FrameProcessorUni(
        tracker=corellation_tracker,
        scaler=scaler,
        annotation=_annotation, 
        options=frame_process_options,
        video_writer=video_writer,
        windows_to_show=windows
        # analytics_engine=_analytics_engine,
        )
    
    VideoTest(
        video=video_path, 
        operation=operation,
        options=video_test_options
    ).run()
    
    print(_analytics_engine.avarage)
    
    if video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    params = load_json("./config_.json")
    main(Path(r"/home/poul/temp/Vids/StreetVid_4"), params)