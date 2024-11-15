from pathlib import Path
import logging
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from analytic_tools.nullable_annotation import NullableAnnotation
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.ious import IOUs
from marks.marks import *
import config

from trackers.scale_estimators import estimate_by_affine, scale_by_all_distances
from trackers.forward_backward_pnt_filter import ForwardBackwardPntFilter
from trackers.forward_bachkward_flow import ForwardBachkwardFlow
from trackers.kcf.corellation_tracker import CorellationTracker
from trackers.median_flow_tracker import MedianFlowTracker
from trackers.mf_motion_only import MFMotionOnly
from trackers.mf_scaler import MFScaler
from trackers.kcf.kcf import *

from generators.smart_grid_pts_generator import SmartGridPtsGenerator
from generators.shi_tomasi_generator import ShiTomasiGenerator
from generators.fast_pts_generator import FastPtsGenerator
from generators.limit_pts_generator import limit_pts_generator

from video_processors.video_test import VideoTest
from video_processors.webm_video_writer import WebmVideoWriter
from video_processors.video_test import VideoTest
from video_processors.process_universal import FrameProcessorUni

from app_gui.utils import load_json


app_logger = logging.getLogger(__name__)

def init_annotation(annotation_path, config_options:dict) -> NullableAnnotation:
    if config_options["annotation"]["process"]:
        try: 
            return NullableAnnotation(annotation_path)
        except OSError as e:
            app_logger.warning(f"No annotation file: {annotation_path}")
            return None
        except Exception as e:
            app_logger.warning(f"Unable to read {annotation_path}")
            app_logger.error(f"Error: {e}")
            return None
    return None

def _init_scale_estimation(config_options:dict):
    functions = {
        "scale_by_all_distances": scale_by_all_distances,
        "estimate_by_affine": estimate_by_affine
    }
    options = config_options["median_flow_methods"]["scale_estimator"]
    selected = [k for k,v in options.items() if v is True]
    if len(selected) > 1:
        app_logger.warning(f"Selected multiple options for scale estimation: {selected}. Using first one")
    try:
        choise = selected[0]
    except IndexError:
        app_logger.error("Selected no option for scale estimation!")
        raise ValueError()
    return functions[choise]

def _init_points_generator(config_options:dict):
    functions = {
        "fast_pts_generator": FastPtsGenerator,
        "smart_grid_pts_generator": SmartGridPtsGenerator,
        "shi_tomasi_generator": ShiTomasiGenerator
    }
    limit = config_options["median_flow_methods"]["points_generator"]["limit_by_ellipse"]
    options = config_options["median_flow_methods"]["points_generator"]["algorithm"]
    selected = [k for k,v in options.items() if v is True]
    if len(selected) > 1:
        app_logger.warning(f"Selected multiple options for points generator: {selected}. Using first one")
    try:
        choise = selected[0]
    except IndexError as e:
        app_logger.error(f"Selected no option for points generator! Error: {e}")
        raise ValueError()
    
    if limit:
        return limit_pts_generator(functions[choise]())
    return functions[choise]()

def init_median_flow_scaler(config_options:dict) -> MFScaler:
    if not config_options["components"]["scale"]:
        return
    # pts_gener = SmartGridPtsGenerator(config._points_density)
    pts_gener = _init_points_generator(config_options)
    fb_flow_generator = ForwardBachkwardFlow(config._flow_generator)
    fb_filter = ForwardBackwardPntFilter(config._max_forward_backward_error)
    scale_estimator = _init_scale_estimation(config_options)
    
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
    video_test_options = VideoTest.Options(process_video_options=True)
    frame_process_options = FrameProcessorUni.Options(
        **config_options["frame_processor_options"]
        )

    annotation_path = test_folder_path / config._annotation_name
    annotation = init_annotation(annotation_path, config_options)
    
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
        annotation=annotation, 
        options=frame_process_options,
        video_writer=video_writer,
        windows_to_show=windows,
        analytics_engine=_analytics_engine
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
    
    # video_path = sys.argv[0]
    # main(video_path, params)
    # main(Path(r"/home/poul/temp/Vids/annotated/dirtroad_06"), params)
    # main(Path(r"/home/poul/temp/Vids/annotated/StreetVid_4"), params)
    # main(Path(r"/home/poul/temp/Vids/annotated/vid_15_1"), params)
    # main(Path(r"/home/poul/temp/Vids/annotated/rotate_street_vid_1"), params)
    # main(Path(r"/home/poul/temp/Vids/boat/"), params)
    
    
    # main(Path(r"/home/poul/temp/Vids/annotated/StreetVid_2"), params) #!!!!!!!!!!!!!!!
