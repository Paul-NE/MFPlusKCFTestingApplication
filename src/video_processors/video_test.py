from pathlib import Path
from dataclasses import dataclass
from typing import Callable
import logging
import json

import numpy as np
import cv2

class VideoParamsProcess:
    def __init__(self, video_path:Path|str, capture:cv2.VideoCapture):
        v_path = Path(video_path)
        self.capture = capture
        self.path = v_path.parent / "params.json"
        self.params = None
        try:
            with open(self.path, "r") as params_file:
                self.params = json.load(params_file)
            self.start: int|None = self.params["start"]
            self.end: int|None = self.params["end"]
        except FileNotFoundError as err:
            print("No video options!") # TODO
            self.start = None
            self.end = None
    
    def preprocess(self):
        if self.start is not None:
            print("start", self.start)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start-1)
    
    def on_run(self):
        if self.end is not None and self.capture.get(cv2.CAP_PROP_POS_FRAMES) >= self.end:
            print("END", self.end)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.capture.get(cv2.CAP_PROP_FRAME_COUNT))


class VideoTest:
    @dataclass
    class Options:
        window: str = "Test"
        process_video_options: bool = False
    
    @dataclass
    class Message:
        video_path: str
        video_capture: cv2.VideoCapture
        window_name: str
        image: np.ndarray
        
    def __init__(self, video: str, operation: Callable[[Message], bool], options: Options = None):
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
        
        self._video_options_processor = VideoParamsProcess(self._video_path, self._video_capture) if self._options.process_video_options else None
    
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
        if self._video_options_processor is not None:
            self._video_options_processor.on_run()
        
        ret, image = self._video_capture.read()
        if ret:
            message = self._form_message(image)
            ret = self._operation(message)
        return ret
    
    def run(self):
        self._video_options_processor.preprocess()
        self._logger.info(f"Video '{self._video_path}' is running")
        while self._main_iteration():
            self._logger.info(f"Running frame {self._video_capture.get(cv2.CAP_PROP_POS_FRAMES)}")
        self._logger.info(f"Video '{self._video_path}' ended")
        cv2.destroyAllWindows()

if __name__=="__main__":
    vid_path = "/home/poul/temp/Vids/annotated/dirtroad_06/test.webm"
    VideoParamsProcess(vid_path)