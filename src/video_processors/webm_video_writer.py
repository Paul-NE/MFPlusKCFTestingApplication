import numpy as np
import cv2


class WebmVideoWriter:
    def __init__(self, path:str, fps:int=20, fourcc:str='VP90'):
        self.path = path
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._fps = fps
        
        self._cv_writer = None
    
    def write(self, image:np.ndarray):
        if self._cv_writer is None:
            shape = image.shape[:2][::-1]
            self._cv_writer = cv2.VideoWriter(
                self.path, 
                self._fourcc, 
                self._fps, 
                shape
                )
        self._cv_writer.write(image)
    
    def release(self):
        if self._cv_writer is not None:
            self._cv_writer.release()
    
    def __del__(self):
        self.release()