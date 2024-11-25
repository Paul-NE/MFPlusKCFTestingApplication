from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton
import sys
import os


class NumberSelectorDialog(QDialog):
    def __init__(self, start, end):
        super().__init__()
        self.selected_number = None  # To store the selected number

        self.setWindowTitle("Select a Number")
        self.setGeometry(100, 100, 300, 200)

        # Create a layout
        layout = QVBoxLayout()

        # Generate buttons for the range
        for i in range(start, end):
            button = QPushButton(str(i))
            button.clicked.connect(self.handle_button_click)  # Connect button click
            layout.addWidget(button)

        self.setLayout(layout)

    def handle_button_click(self):
        # Get the number from the button text and store it
        button = self.sender()
        self.selected_number = int(button.text())
        self.accept()  # Close the dialog and return control


def get_number(start, end):
    """Function to get a number from a given range."""
    # Ensure an application instance is available
    app = QApplication.instance()
    if app is None:  # If no QApplication exists, create one
        app = QApplication(sys.argv)

    dialog = NumberSelectorDialog(start, end)
    if dialog.exec():  # Show the dialog modally
        return dialog.selected_number  # Return the selected number
    return None  # Return None if dialog was closed without selection


# Example usage
if __name__ == "__main__":
    number = get_number(1, 10)  # Call the function to get a number
    print(f"Selected number: {number}")
    print("Program continues running.")
    
