from dataclasses import dataclass
from typing import Callable
import logging

import numpy as np
import cv2

class VideoTest:
    @dataclass
    class Options:
        window: str = "Test"
    
    @dataclass
    class Message:
        video_path: str
        video_capture: cv2.VideoCapture
        window_name: str
        image: np.ndarray
        
    def __init__(self, video: cv2.VideoCapture, operation: Callable[[Message], bool], options: Options = None):
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._video_path = video 
        self._operation = operation
        self._options = options if options is not None else self.Options()
        
        self._video_capture: cv2.VideoCapture
        self._setup()
    
    def _setup(self):
        self._video_capture = cv2.VideoCapture(self._video_path)
        if not self._video_capture.isOpened():
            self._logger.info(f"Unable to read video named: {self._video_path}")
    
    @property
    def window(self):
        return self._options.window
    
    @window.setter
    def window(self, win):
        self._options.window = win
    
    def _form_message(self, image):
        return self.Message(
            video_path = self._video_path,
            video_capture = self._video_capture,
            window_name = self._options.window,
            image = image
        )
    
    def _main_iteration(self) -> bool:
        ret, image = self._video_capture.read()
        if ret:
            message = self._form_message(image)
            ret = self._operation(message)
        return ret
    
    def run(self):
        self._logger.info(f"Video '{self._video_path}' is running")
        while self._main_iteration():
            self._logger.info(f"Running frame {self._video_capture.get(cv2.CAP_PROP_POS_FRAMES)}")
        self._logger.info(f"Video '{self._video_path}' ended")
        cv2.destroyAllWindows()