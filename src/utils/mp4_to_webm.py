from pathlib import Path
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import cv2

from video_processors.webm_video_writer import WebmVideoWriter
from video_processors.video_test import VideoTest


dir_path = Path(r"/home/poul/temp/Vids/annotated/dirtroad_06")
source_name = "test.mp4"
target_name = "test.webm"
writer = WebmVideoWriter(dir_path / target_name)

def processor(message: VideoTest.Message) -> bool:
    writer.write(message.image)
    return True

VideoTest(dir_path / source_name, processor).run()
writer.release()