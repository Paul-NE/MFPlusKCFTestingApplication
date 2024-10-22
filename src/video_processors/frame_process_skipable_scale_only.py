from typing import Iterator, Self
from dataclasses import dataclass

import numpy as np
import cv2

from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.ious import IOUs
from trackers.scaler import Scaler
from utils.utils import round_box
# from trackers.median_flow_scale_only import MedianFlowScaleOnly
from geometry import BoundingBox

from .webm_video_writer import WebmVideoWriter
from .video_test import VideoTest


class FrameProcessorSkipable:
    """Callable object for processing video frame by frame
    """
    @dataclass
    class Options:
        wait_key_value: int = 0
        start_paused_value: bool = False
        skip_first_n_frames: int = 0
        
    @dataclass
    class PreprocessedData:
        annotation: BoundingBox
        message: VideoTest.Message
    
    def __init__(self, annotation:AnnotationLight, tracker:Scaler, analytics_engine:IOUs, video_writer: WebmVideoWriter, marker: ImageMarks, options: Options = None) -> None:
        self._tracker: Scaler = tracker
        self._annotatuon: AnnotationLight = annotation
        self._annotatuon_iterator: Iterator
        self._analytics_engine: IOUs = analytics_engine
        self._video_writer: WebmVideoWriter = video_writer
        self._marker: ImageMarks = marker
        
        self._options = options if options is not None else self.Options()
        self._setup()
    
    def _setup(self):
        self._annotatuon_iterator = iter(self._annotatuon)
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
            self._show_marked(window, image)
            continue
    
    
    def _preprocess(self, message: VideoTest.Message) -> PreprocessedData:
        """Adds new to Message get.

        Args:
            message (VideoTest.Message): _description_

        Returns:
            PreprocessedData: Object - storing the data
        """
        try: 
            annotation = next(self._annotatuon_iterator)
        except StopIteration:
            annotation = None
        if annotation is not None:
            annotation = BoundingBox.generate_from_list(annotation)
        
        preprocessed = self.PreprocessedData(
            annotation,
            message
        )
        return preprocessed
    
    def _show(self, window: str, image: np.ndarray):
        cv2.imshow(window, image)
        self._key_process()
    
    def _show_marked(self, window: str, image: np.ndarray):
        marked = self._marker.draw_all(image)
        self._show(window, marked)
    
    def _tracker_init_update(self, data: PreprocessedData):
        if not self._tracker.inited and data.annotation is not None:
            self._tracker.init(data.message.image, data.annotation)
            return data.annotation
        elif not self._tracker.inited and data.annotation is None:
            return None
        elif self._tracker.inited and data.annotation is not None:
            tracker_result = self._tracker.update(data.message.image, data.annotation)
            print(f"{tracker_result.width / tracker_result.height=}")
            tracker_result = round_box(tracker_result)
            if tracker_result is not None:
                tracker_rect = Rectangle(tracker_result)
                tracker_rect.style.color = (255, 0, 0)
                self._marker.add(tracker_rect)
            return tracker_result
        else:
            return None
    
    def _process_tracker(self, data: PreprocessedData)->BoundingBox:
        """Refactor needed
        Run single tracker instance

        Args:
            data (_type_): _description_

        Returns:
            BoundingBox: _description_
        """
        result = self._tracker_init_update(data)
        self._process_tracker_result(data, result)
    
    def _process_tracker_result(self, data: PreprocessedData, tracker_result):
        if tracker_result is not None and data.annotation is not None:
            self._analytics_engine.iou_append(list(data.annotation), list(tracker_result))
        else:
            print("Tracker returned none")
    
    def _process(self, data: PreprocessedData):
        """Main programm logic processing

        Args:
            data (PreprocessedData): Data after preprocessing
        """
        if self._skip_first_n_frames > 0:
            self._skip_first_n_frames -= 1
            return
        
        if data.annotation is not None:
            self._marker.add(Rectangle(data.annotation))
        
        self._process_tracker(data)
        self._video_writer.write(self._marker.draw_all(data.message.image))
        
        self._show_marked(data.message.cv_window, data.message.image)
        
        if self._paused: 
            self._process_pause(data.message.cv_window, data.message.image)
    
    def _postprocess(self):
        """Clearing data before next iteration and saving some log data.
        """
        self._marker.clear()
    
    def __call__(self, message: VideoTest.Message) -> bool:
        """One itteration of video process

        Args:
            message (VideoTest.Message): video data

        Returns:
            bool: don't stop processing
        """
        preprocessed = self._preprocess(message)
        
        self._process(preprocessed)
        
        self._postprocess()
        return self._keep_running