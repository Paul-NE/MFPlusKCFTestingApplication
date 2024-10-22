from typing import Iterator, Self
from dataclasses import dataclass

import numpy as np
import cv2

from marks.marks import *
from analytic_tools.annotation_light import AnnotationLight
from analytic_tools.ious import IOUs
from trackers.tracker import Tracker
from geometry import BoundingBox

from .webm_video_writer import WebmVideoWriter
from .video_test import VideoTest


class FrameProcessor:
    """Callable object for prosessing video frame by frame
    """
    @dataclass
    class Options:
        wait_key_value: int = 1
        start_paused_value: bool = False
        
    @dataclass
    class PreprosessedData:
        annotation: BoundingBox
        message: VideoTest.Message
    
    def __init__(self, annotation:AnnotationLight, tracker:Tracker, analytics_engine:IOUs, video_writer: WebmVideoWriter, marker: ImageMarks, options: Options = None) -> None:
        self._tracker: Tracker = tracker
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
    
    
    def _preprocess(self, message: VideoTest.Message) -> PreprosessedData:
        """Adds new to Message get.

        Args:
            message (VideoTest.Message): _description_

        Returns:
            PreprosessedData: Object - storing the data
        """
        annotation = next(self._annotatuon_iterator)
        if annotation is not None:
            annotation = BoundingBox.generate_from_list(annotation)
        
        preprosessed = self.PreprosessedData(
            annotation,
            message
        )
        return preprosessed
    
    def _show(self, window: str, image: np.ndarray):
        cv2.imshow(window, image)
        self._key_process()
    
    def _show_marked(self, window: str, image: np.ndarray):
        marked = self._marker.draw_all(image)
        self._show(window, marked)
    
    def _tracker_init_update(self, data):
        if not self._tracker.inited and data.annotation is not None:
            self._tracker.init(data.message.image, data.annotation)
            return data.annotation
        elif not self._tracker.inited and data.annotation is None:
            return None
        else:
            tracker_result = self._tracker.update(data.message.image)
            if tracker_result is not None:
                tracker_rect = Rectangle(tracker_result)
                tracker_rect.style.color = (255, 0, 0)
                self._marker.add(tracker_rect)
            return tracker_result
    
    def _process_tracker(self, data)->BoundingBox:
        """Refactor needed
        Run single tracker instance

        Args:
            data (_type_): _description_

        Returns:
            BoundingBox: _description_
        """
        result = self._tracker_init_update(data)
        self._process_tracker_result(data, result)
    
    def _process_tracker_result(self, data, tracker_result):
        if tracker_result is not None:
            self._analytics_engine.iou_append(list(data.annotation), list(tracker_result))
        else:
            print("Tracker returned none")
    
    def _process(self, data: PreprosessedData):
        """Main programm logic processing

        Args:
            data (PreprosessedData): Data after preprocessing
        """
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
        preprosessed = self._preprocess(message)
        
        self._process(preprosessed)
        
        self._postprocess()
        return self._keep_running