import json
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import QThread

from .utils import load_json


class SettingsWindow(QtWidgets.QDialog):
    def __init__(self, settings_file):
        super().__init__()
        self.setWindowTitle("Settings Window with Nested Objects")
        # self.setGeometry(100, 100, 400, 400)
        self.resize(400, 600)

        # Load settings from JSON file
        self._settings_file = settings_file
        self._settings = load_json(self._settings_file)
        
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)

        # Create input widgets dynamically based on JSON
        self.inputs = {}  # Store references to input widgets (for all levels)
        self.create_widgets(self._settings, self.layout, self.inputs)

        # Save button
        save_button = QtWidgets.QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(save_button)

    def create_dict_form(self, parent_layout, inputs_ref, key, value):
        # Handle nested objects: create a group box with its own layout
        group_box = QtWidgets.QGroupBox(key)
        group_layout = QtWidgets.QFormLayout()
        group_box.setLayout(group_layout)
        parent_layout.addWidget(group_box)

        # Recursively create widgets for inner objects
        inputs_ref[key] = {}  # Create a nested dictionary in inputs
        self.create_widgets(value, group_layout, inputs_ref[key])
    
    def create_bool_form(self, parent_layout, inputs_ref, key, value):
        # Create a checkbox for boolean values
        checkbox = QtWidgets.QCheckBox(key)
        checkbox.setChecked(value)
        parent_layout.addWidget(checkbox)
        inputs_ref[key] = checkbox  # Store widget reference
    
    def create_int_form(self, parent_layout, inputs_ref, key, value):
        # Create a spin box for integer values
        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(f"{key}:")
        spinbox = QtWidgets.QSpinBox()
        spinbox.setValue(value)
        spinbox.setMaximum(10000)  # Adjust max as needed
        hbox.addWidget(label)
        hbox.addWidget(spinbox)
        parent_layout.addRow(hbox)
        inputs_ref[key] = spinbox  # Store widget reference

    def create_str_form(self, parent_layout, inputs_ref, key, value):
        # Create a line edit for string values
        label = QtWidgets.QLabel(f"{key}:")
        line_edit = QtWidgets.QLineEdit()
        line_edit.setText(str(value))
        parent_layout.addRow(label, line_edit)
        inputs_ref[key] = line_edit  # Store widget reference

    def create_widgets(self, data, parent_layout, inputs_ref):
        """Dynamically create label/input pairs from the JSON data."""
        forms = {
            "dict": self.create_dict_form,
            "bool": self.create_bool_form,
            "int": self.create_int_form,
            "str": self.create_str_form,
            "NoneType": self.create_str_form,
            "list": self.create_str_form
        }
        for key, value in data.items():
            forms[type(value).__name__](parent_layout, inputs_ref, key, value)

    def save_settings(self):
        """Recursively gather the current state of inputs and save to JSON."""
        self.update_settings(self._settings, self.inputs)
        
        # Save updated settings to the JSON file
        with open(self._settings_file, "w") as file:
            json.dump(self._settings, file, indent=4)
        print(f"Settings saved to {self._settings_file}")

    def update_settings(self, settings, inputs):
        """Recursively update the settings dictionary based on widget values."""
        for key, widget in inputs.items():
            if isinstance(widget, dict):
                # Recursively update nested objects
                self.update_settings(settings[key], widget)
            elif isinstance(widget, QtWidgets.QCheckBox):
                settings[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QSpinBox):
                settings[key] = widget.value()
            elif isinstance(widget, QtWidgets.QLineEdit):
                settings[key] = widget.text()