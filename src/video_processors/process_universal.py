from typing import Iterator, Self
from dataclasses import dataclass

import numpy as np
import cv2

from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.ious import IOUs
from trackers.scaler import Scaler
from trackers.tracker import AdjustableTracker
from trackers.errors import NotInited
from utils.utils import round_box
# from trackers.median_flow_scale_only import MedianFlowScaleOnly
from geometry import BoundingBox

from .webm_video_writer import WebmVideoWriter
from .video_test import VideoTest


class AnnotationManager:
    def __init__(self, annotation:AnnotationLight):
        self._annotation = annotation
        self._annotation_iter = iter(self._annotation)
    
    def update(self, message: VideoTest.Message) -> BoundingBox:
        try: 
            annotation = next(self._annotation_iter)
            return BoundingBox.generate_from_list(annotation) if annotation is not None else None
        except StopIteration:
            while True:
                return None


class TrackerScaleManager:
    def __init__(self, tracker:AdjustableTracker, scaler: Scaler):
        self._tracker = tracker
        self._scaler = scaler
    
    def update(self, message: VideoTest.Message, annotation: BoundingBox|None=None) -> BoundingBox:
        try:
            tracker_result = self._tracker.update(message.image)
            scaler_result = self._scaler.update(message.image, tracker_result)
            if scaler_result is not None:
                self._tracker.adjust_bounding_box(scaler_result)
                return scaler_result, tracker_result
            return tracker_result, scaler_result
        except NotInited:
            if annotation is not None:
                self._tracker.init(message.image, annotation)
                self._scaler.init(message.image, annotation)
                return None, None
            return None, None


class AnalyticsManager:
    def __init__(self, analytics_engine:IOUs):
        self._analytics_engine = analytics_engine
    
    def update(self, tracker: BoundingBox|None, annotation: BoundingBox|None):
        if tracker is None or annotation is None:
            return
        self._analytics_engine.iou_append(tracker, annotation)


class MarkerManager:
    def __init__(self, marker: ImageMarks):
        self._marker = marker
    
    def add_box(self, box: BoundingBox, color: tuple[int, int, int]|None = None):
        if box is None:
            return
        box_rect = Rectangle(box)
        if color is not None:
            box_rect.style.color = color
        self._marker.add(box_rect)
    
    def mark_image(self, image: np.ndarray):
        image_marked = self._marker.draw_all(image)
        self._marker.clear()
        return image_marked


class FrameProcessorUni:
    """Callable object for processing video frame by frame
    """
    @dataclass
    class Options:
        wait_key_value: int = 0
        start_paused_value: bool = False
        skip_first_n_frames: int = 0
    
    def __init__(
            self, 
            tracker:AdjustableTracker, 
            scaler:Scaler, 
            annotation:AnnotationLight, 
            video_writer:WebmVideoWriter|None=None, 
            options:Options=None) -> None:
        self.marker = MarkerManager(ImageMarks())
        self.tracker = TrackerScaleManager(tracker, scaler)
        self.annotation = AnnotationManager(annotation)
        self.border = 70
        self._options = options if options is not None else self.Options()
        self._setup()
    
    def _setup(self):
        self._paused = self._options.start_paused_value
        self._keep_running = True
        self._skip_first_n_frames = self._options.skip_first_n_frames
    
    def _pause_unpause(self):
        self._paused = not self._paused
        return True
    
    def _key_process(self) -> bool:
        key = cv2.waitKey(self._options.wait_key_value)
        reactions = {
            ord(" "): self._pause_unpause,
            ord("q"): lambda: False,
            ord("Q"): lambda: False,
            27: lambda: False
        }
        if key in reactions and reactions[key] is not None:
            self._keep_running = reactions[key]()
    
    def _process_pause(self, window: str, image: np.ndarray):
        while self._paused:
            self._show(window, image)
            continue
    
    def _show(self, window: str, image: np.ndarray):
        cv2.imshow(window, image)
        self._key_process()
    
    def __call__(self, message: VideoTest.Message) -> bool:
        """One itteration of video process

        Args:
            message (VideoTest.Message): video data

        Returns:
            bool: don't stop processing
        """
        
        annotation_box = self.annotation.update(message)
        if self.border > 0:
            self.border-=1
            return self._keep_running
        scaler_box, tracker_box = self.tracker.update(message, annotation_box)
        self.marker.add_box(annotation_box, (255, 0, 0))
        self.marker.add_box(scaler_box, (0, 255, 0))
        self.marker.add_box(tracker_box)
        
        image_marked = self.marker.mark_image(message.image)
        self._show(message.cv_window, image_marked)
        
        if self._paused: 
            self._process_pause(message.cv_window, image_marked)
        
        return self._keep_running