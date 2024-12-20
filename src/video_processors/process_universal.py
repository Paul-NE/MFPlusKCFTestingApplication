from typing import Iterator, Self, Callable, Any
from dataclasses import dataclass

import numpy as np
import cv2

from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.annotation_json import AnnotationJson
from analytic_tools.ious import Analytics
from trackers.scaler import Scaler
from trackers.tracker import AdjustableTracker
from trackers.errors import NotInited
from utils.utils import round_box
# from trackers.median_flow_scale_only import MedianFlowScaleOnly
from geometry import BoundingBox, Point

from .webm_video_writer import WebmVideoWriter
from .video_test import VideoTest
from .window_manager import Window, WindowManager


class ManualROISelector:
    def __init__(self, window_name: str, callback:Callable[[BoundingBox], Any]):
        self._window_name = window_name
        self.callback = callback
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.onmouse)
        self.drag_start = None
        self.drag_rect = None

    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        
        if not self.drag_start:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN or \
        (flags & cv2.EVENT_FLAG_LBUTTON and event != cv2.EVENT_LBUTTONUP):
            xo, yo = self.drag_start
            x0, y0 = np.minimum([xo, yo], [x, y])
            x1, y1 = np.maximum([xo, yo], [x, y])
            self.drag_rect = None
            if x1-x0 > 0 and y1-y0 > 0:
                self.drag_rect = (x0, y0, x1, y1)
            return
        
        if not self.drag_rect:
            return
        
        rect = self.drag_rect
        self.drag_start = None
        self.drag_rect = None
        box = BoundingBox(
            top_left_pnt=Point(
                x = rect[0],
                y = rect[1]
            ),
            bottom_right_pnt=Point(
                x = rect[2],
                y = rect[3]
            )
        )
        self.callback(box)

    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    @property
    def dragging(self):
        return self.drag_rect is not None


class AnnotationStop(Exception):
    """No further annotation"""
    pass


class AnnotationManager:
    def __init__(self, annotation:AnnotationLight):
        self._annotation = annotation
        self._annotation_iter = iter(self._annotation)
        self._last_seen_frame = 0
    
    def _process_frame_skipping(self, message: VideoTest.Message):
        current_frame = message.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        if self._last_seen_frame < current_frame - 1:
            frames_to_skip_count = int(current_frame - 1 - self._last_seen_frame)
            for _ in range(frames_to_skip_count):
                next(self._annotation_iter)
        self._last_seen_frame = current_frame
    
    def update(self, message: VideoTest.Message) -> BoundingBox:
        try: 
            self._process_frame_skipping(message)
            annotation = next(self._annotation_iter)
            return BoundingBox.generate_from_list(annotation) if annotation is not None else None
        except StopIteration:
            while True:
                return None


class AnnotationJsonManager:
    @dataclass
    class Options:
        skip_to_first_annotated: bool = True
        
    def __init__(self, annotation:AnnotationJson, options:Options=None):
        self._annotation = annotation
        self._last_idex = annotation.get_last_frame_index()
        self._first_index = annotation.get_first_frame_index()
        self._options = options if options is not None else self.Options()
    
    def update(self, message: VideoTest.Message) -> BoundingBox:
        cur_frame = message.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        if cur_frame > self._last_idex:
            raise AnnotationStop
        if self._options.skip_to_first_annotated and cur_frame < self._first_index:
            message.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self._first_index - 1)
            return None
        annotation = self._annotation.get_current_box()
        return BoundingBox.generate_from_list(annotation) if annotation is not None else None


class TrackerScaleManager:
    def __init__(self, tracker:AdjustableTracker|None, scaler: Scaler|None):
        assert tracker is not None or scaler is not None
        self._tracker = tracker
        self._scaler = scaler
        self._width_height_buffer: list[float, float]|None = None
        self._previous_state: BoundingBox|None = None
        self._px_border = 5
    
    def _validate_annotation_box(self, message: VideoTest.Message, box: BoundingBox) -> bool:
        w_max = message.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) - 1
        h_max = message.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1
        if box.top_left_pnt.x < self._px_border:
            return False
        if box.top_left_pnt.y < self._px_border:
            return False
        if box.bottom_right_pnt.x > w_max - self._px_border:
            return False
        if box.bottom_right_pnt.y > h_max - self._px_border:
            return False
        return True
    
    def init(self, message: VideoTest.Message, ground_true: BoundingBox) -> BoundingBox:
        """Forced initialization, not recommended to use

        Args:
            message (VideoTest.Message): _description_
            ground_true (BoundingBox): _description_

        Returns:
            BoundingBox: _description_
        """
        assert ground_true is not None
        assert self._tracker is not None
        if not self._validate_annotation_box(message, ground_true):
            return None, None
        
        self._tracker.init(message.image, ground_true)
        if self._scaler is not None:
            self._scaler.init(message.image, ground_true)
        
        return None, None
    
    def _tracker_update(self, message: VideoTest.Message, annotation: BoundingBox|None=None) -> BoundingBox:
        try:
            tracker_result = self._tracker.update(message.image)
            if self._scaler is None:
                return tracker_result, None
            
            scaler_result = self._scaler.update(message.image, tracker_result)
            if scaler_result is not None:
                self._tracker.adjust_bounding_box(scaler_result)
                return scaler_result, tracker_result
            
            return tracker_result, scaler_result
        except NotInited:
            if annotation is None:
                return None, None
            self._tracker.init(message.image, annotation)
            
            if self._scaler is None:
                return None, None
            self._scaler.init(message.image, annotation)
            return None, None
    
    def _no_tracker_update(self, message: VideoTest.Message, annotation: BoundingBox|None=None) -> BoundingBox:
        try:
            if not annotation:
                return None, None
            if self._width_height_buffer is None:
                raise NotInited
            
            center = annotation.center
            annotation_old_scale = BoundingBox(
                top_left_pnt=Point(
                    x = center.x - self._width_height_buffer[0]/2,
                    y = center.y - self._width_height_buffer[1]/2,
                ),
                bottom_right_pnt=Point(
                    x = center.x + self._width_height_buffer[0]/2,
                    y = center.y + self._width_height_buffer[1]/2,
                )
            )
            scaler_result = self._scaler.update(message.image, annotation_old_scale)
            if scaler_result is None:
                return self._previous_state, None
                return None, None
            self._width_height_buffer = [scaler_result.width, scaler_result.height]
            self._previous_state = scaler_result
            return scaler_result, annotation_old_scale
            
        except NotInited:
            self._scaler.init(message.image, annotation)
            self._width_height_buffer = [annotation.width, annotation.height]
            return None, None
    
    def update(self, message: VideoTest.Message, annotation: BoundingBox|None=None) -> BoundingBox:
        if self._tracker:
            return self._tracker_update(message, annotation)
        return self._no_tracker_update(message, annotation)


class AnalyticsManager:
    def __init__(self, analytics_engine:Analytics):
        self._analytics_engine = analytics_engine
    def set_limiting_frames(self, min_frame, max_frame):
        invert_op = getattr(self._analytics_engine, "invert_op", None)
        if callable(invert_op):
            invert_op(self.path.parent_op)
    def update(self, tracker: BoundingBox|None, annotation: BoundingBox|None):
        self._analytics_engine.update(tracker, annotation)


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
        manual_roi_selection: bool = False
        skip_first_n_frames: int = 0
    
    def __init__(
            self, 
            tracker:AdjustableTracker|None=None, 
            scaler:Scaler|None = None, 
            annotation:AnnotationLight|None=None, 
            video_writer:WebmVideoWriter|None=None, 
            windows_to_show:list[Window]|None=None,
            analytics_engine: Analytics|None = None,
            options:Options=None
            ) -> None:
        
        self._window_manager = WindowManager()
        self._setip_window_manager(windows_to_show)
        self._main_window: Window = None
        
        self._marker = MarkerManager(ImageMarks())
        self._video_writer = video_writer
        self._analytics_engine = analytics_engine
        self._tracker = TrackerScaleManager(tracker, scaler)
        self._annotation = AnnotationManager(annotation) if annotation is not None else None
        
        self._options = options if options is not None else self.Options()
        self._setup()
    
    def set_annotation_json(self, annotation:AnnotationJson|None=None):
        self._annotation = AnnotationJsonManager(annotation)
    
    def _setip_window_manager(self, windows_to_show:list[Window]|None) -> None:
        if windows_to_show is None:
            return
        self._window_manager.add_windows(windows_to_show)
    
    def _setup(self):
        self._paused = self._options.start_paused_value
        self._keep_running = True
        self._skip_n_frames = self._options.skip_first_n_frames
        self._mouse_handler = None
        self._mouse_box = None
        self._setup_manual_run_command()
    
    def _setup_manual_run_command(self):
        manual_annotated_commands = {
            True: self._manual_roi_run,
            False: self._annotated_run
        }
        self._run = manual_annotated_commands[self._options.manual_roi_selection]
    
    def _set_tracker_roi(self, roi: BoundingBox):
        self._mouse_box = roi
    
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
        while self._paused and self._keep_running:
            self._show(window, image)
            continue
    
    def _show(self, window_name: str, image: np.ndarray):
        # cv2.imshow(window_name, image)
        if self._main_window is None:
            self._main_window = Window(window_name)
            self._window_manager.add_windows([self._main_window])
        self._main_window.frame = image
        self._window_manager.show_all()
        self._key_process()
    
    def _manual_roi_run(self, message: VideoTest.Message) -> bool:
        if self._mouse_handler is None:
            self._mouse_handler = ManualROISelector(message.window_name, self._set_tracker_roi)
        if self._mouse_box is not None:
            self._tracker.init(message, self._mouse_box)
            self._mouse_box = None
            return
        scaler_box, tracker_box = self._tracker.update(message, None)
        self._marker.add_box(scaler_box, (0, 255, 0))
    
    def _annotated_run(self, message: VideoTest.Message) -> bool:
        try:
            annotation_box = self._annotation.update(message) if self._annotation is not None else None
        except AnnotationStop:
            annotation_box = None
            self._keep_running = False
        scaler_box, tracker_box = self._tracker.update(message, annotation_box)
        
        self._analytics_engine.update(annotation_box, scaler_box)
        self._marker.add_box(annotation_box, (255, 0, 255))
        self._marker.add_box(scaler_box, (0, 255, 0))
        # self._marker.add_box(tracker_box)
    
    def __call__(self, message: VideoTest.Message) -> bool:
        """One itteration of video process

        Args:
            message (VideoTest.Message): video data

        Returns:
            bool: don't stop processing
        """
        if self._skip_n_frames > 0:
            self._skip_n_frames-=1
            return self._keep_running
        
        self._run(message)
        
        image_marked = self._marker.mark_image(message.image)
        if self._video_writer is not None:
            self._video_writer.write(image_marked)
        self._show(message.window_name, image_marked)
        
        if self._paused: 
            self._process_pause(message.window_name, image_marked)
        
        return self._keep_running
    