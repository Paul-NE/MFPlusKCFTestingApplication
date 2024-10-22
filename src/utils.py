from utils.utils import map_int_list
from geometry import BoundingBox
from video_processors.video_test import VideoTest
# from analytic_tools.annotation_light import AnnotationLight

import cv2
import logging

# Configure logging globally (done in main script usually)
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Annotation(list):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    # def 

def annotation_from_file(path:str):
    return Annotation(annotation_file_process(path))

def annotation_file_process(path:str):
    with open(path) as annotation_file:
        next(annotation_file)
        current_frame_number = 0
        for line in annotation_file:
            bounding_box, frane_number = parse_line(line)
            for _ in range(frane_number - current_frame_number):
                yield None
            current_frame_number = frane_number + 1
            yield from_xywh_to_xyxy(bounding_box)

def parse_line(line:str) -> tuple[list, int]|None:
    if not line:
        return None
    splited_line:list[str] = line.strip().split(",")
    if not all([element.isdecimal() for element in splited_line]):
        return None
    int_splited_line = map_int_list(splited_line)
    bounding_box, frame_number = int_splited_line[:-1], int_splited_line[-1]
    return bounding_box, frame_number

def from_xywh_to_xyxy(box_xywh: list) -> list:
    assert len(box_xywh) == 4
    arr_xyxy = list(box_xywh)
    arr_xyxy[2] += arr_xyxy[0]
    arr_xyxy[3] += arr_xyxy[1]
    return arr_xyxy

def write_annoptation_light(path:str, annotation:Annotation):
    with open(path, "w") as file:
        for box in annotation:
            if box is None:
                file.write("None\n")
                continue
            str_box = list(map(str, box))
            str_box[1] += ";"
            file.write(" ".join(str_box) + "\n")


class TerstRun:
    def __init__(self, annotation:Annotation):
        self._annotation:Annotation = annotation
        self._iter_annotation = iter(self._annotation)
    
    def __call__(self, message: VideoTest.Message) -> bool:
        if message.image is None:
            return False
        image = message.image
        try:
            bounding_box = next(self._iter_annotation)
        except StopIteration:
            bounding_box = None
        if bounding_box is not None:
            cv2.rectangle(image, bounding_box[:2], bounding_box[2:], (0, 0, 255))
        cv2.imshow(message.cv_window, message.image)
        cv2.waitKey(1)
        return True


def run_test(annotation: Annotation):
    oper = TerstRun(annotation)
    test = VideoTest(r"/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/Annotated_Vidos/Videos/StreetVid_4.webm", oper)
    test.run()

if __name__ == "__main__":
    annotation = annotation_from_file(r"/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/Annotated_Vidos/Annotation/StreetVid_4.csv")
    run_test(annotation)
    write_annoptation_light("/home/poul/temp/Vids/test.txt", annotation)