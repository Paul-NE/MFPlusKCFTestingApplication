from dataclasses import dataclass

import numpy as np
import cv2


class Window:
    def __init__(self, name:str):
        self._name:str = name
        self._frame:np.ndarray|None = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def frame(self) -> np.ndarray:
        return self._frame
    
    @frame.setter
    def frame(self, image:np.ndarray) -> np.ndarray|None:
        self._frame = image
    
    def show(self):
        if self._frame == None:
            return
        cv2.imshow(self._name, self._frame)


class WindowManager:
    def __init__(self):
        self._windows: dict[str, Window] = {}
    
    def add_windows(self, windows:list[Window]):
        for window in windows:
            assert "window.name" not in self._windows
            self._windows[window.name] = window
    
    def show_all(self):
        for name, win in self._windows.items():
            if win.frame is None:
                continue
            cv2.imshow(name, win.frame)
    
    def free_all(self):
        names = list(self._windows.keys())
        for name in names:
            cv2.destroyWindow(name)
            del self._windows[name]


def _window_manager_manual_test():
    win_1 = Window(name="test_1")
    win_2 = Window(name="test_2")
    
    win_manager = WindowManager()
    win_manager.add_windows([win_1, win_2])
    
    win_1.frame = np.zeros((100, 100), dtype=np.uint8)
    win_2.frame = np.zeros((200, 200), dtype=np.uint8)
    
    win_manager.show_all()
    cv2.waitKey(0)
    
    win_1.frame = np.ones((100, 100), dtype=np.uint8) * 255
    win_2.frame = np.ones((200, 200), dtype=np.uint8) * 255
    
    win_manager.show_all()
    cv2.waitKey(0)
    
    win_manager.free_all()


if __name__=="__main__":
    _window_manager_manual_test()