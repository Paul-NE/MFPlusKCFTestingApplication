import sys
import os

from PySide6 import QtWidgets
from PySide6.QtCore import QThread

from app_gui.video_test_gui import VideoTestGUI
from app import main


if __name__ == "__main__":
    # Create the application object
    app = QtWidgets.QApplication(sys.argv)

    # Create and show the main window
    window = VideoTestGUI(main)
    window.show()

    code = app.exec()

    os._exit(code)
