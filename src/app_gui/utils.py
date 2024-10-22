import json
from pathlib import Path


def load_json(settings_file_path: Path):
    """Load settings from the JSON file."""
    try:
        with open(settings_file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {settings_file_path} not found.")
        return {}