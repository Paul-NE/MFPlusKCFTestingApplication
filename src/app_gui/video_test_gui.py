import os
import sys
import subprocess
from pathlib import Path
from typing import Callable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PySide6 import QtWidgets

from .utils import load_json
from .settings_window import SettingsWindow


class VideoTestGUI(QtWidgets.QMainWindow):
    def __init__(self, executable: Callable[[Path, dict], None]):
        super().__init__()
        self.setWindowTitle("Video test manager")
        self.resize(400, 600)
        
        self._video_test_settings_path = "config_.json"
        self._video_folder_path = None
        self._worker = executable
        # Initialize UI elements
        self.init_ui()

    def init_ui(self):
        """Initialize the main window's UI components."""
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Create the widgets
        self.create_label(layout)
        self.create_run_button(layout)
        self.create_set_video_folder_button(layout)
        self.create_settings_button(layout)

        # Set the layout for the central widget
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_run_button(self, layout):
        file_button = QtWidgets.QPushButton("Run", self)
        file_button.clicked.connect(self.run_video)
        layout.addWidget(file_button)

    def run_video(self):
        settings = load_json(self._video_test_settings_path)
        self._worker(self._video_folder_path, settings)
        # subprocess().call()
        

    def create_label(self, layout):
        """Create and add a label to the layout."""
        self.label = QtWidgets.QLabel("Video folder not selected", self)
        layout.addWidget(self.label)

    def create_set_video_folder_button(self, layout):
        """Create and add the file selection button to the layout."""
        file_button = QtWidgets.QPushButton("Select video Folder", self)
        file_button.clicked.connect(self.open_directory_dialog)
        layout.addWidget(file_button)

    def create_settings_button(self, layout):
        """Create and add the subwindow button to the layout."""
        subwindow_button = QtWidgets.QPushButton("Settings", self)
        subwindow_button.clicked.connect(self.open_subwindow)
        layout.addWidget(subwindow_button)

    def open_directory_dialog(self):
        """Open a directory dialog and update the label with the selected folder."""
        inputed_video_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Select video Folder", dir="/home/poul/temp/Vids")
        if inputed_video_folder_path:
            self._video_folder_path = Path(inputed_video_folder_path)
            self.label.setText(f"Video folder: {inputed_video_folder_path}")
        else:
            self.label.setText("Video folder not selected")

    def open_subwindow(self):
        """Open a modal subwindow."""
        sub_window = SettingsWindow(self._video_test_settings_path)
        sub_window.exec()

    def closeEvent(self, event):
        """Handle the window close event."""
        print("Window closed!")
        QtWidgets.QApplication.quit()  # Ensure application quits properly
        event.accept()  # Call the accept method to ensure the window closes