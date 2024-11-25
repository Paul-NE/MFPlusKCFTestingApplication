import json
from pathlib import Path

import numpy as np
import cv2


class NoSuchObjectFound(IndexError):
    pass


class AnnotationJson:
    def __init__(self, annotation_path: str, capture: cv2.VideoCapture):
        self._objects: list[TrackedObject] = []
        self._selected_object: int = 0
        self._capture = capture
        with open(annotation_path, "r") as annotation_file:
            annotation_json = json.load(annotation_file)
            for box_obj in annotation_json["box"]:
                self._objects.append(TrackedObject(box_obj, capture))
    
    @property
    def objects(self):
        return len(self._objects)
    
    @property
    def selected_object(self) -> int:
        return self._selected_object
    
    @selected_object.setter
    def selected_object(self, option: int):
        if not (0 <= option < len(self._objects)):
            raise NoSuchObjectFound
        self._selected_object = option
    
    def get_all_current_boxes(self):
        return [annotation.get_current_annitation() for annotation in self._objects]
    
    def get_current_box(self) -> tuple[int, int, int, int]|None:
        tracked_object = self._objects[self._selected_object]
        return tracked_object.get_current_annitation()
    
    def get_last_frame_index(self) -> int:
        tracked_object = self._objects[self._selected_object]
        return tracked_object.get_last_frame()
    
    def get_first_frame_index(self) -> int:
        tracked_object = self._objects[self._selected_object]
        return tracked_object.get_first_frame()


class TrackedObject:
    def __init__(self, box_dict: dict, capture: cv2.VideoCapture):
        self._sequence = box_dict["sequence"]
        self._capture = capture
        self._base_annotation: dict[int: dict[str: any]] = {}
        self._extended_annotation: dict[int: tuple[int, int, int, int]] = {}
        self._capture_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._capture_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        assert self._capture_w!=0 and self._capture_h!=0
        self._parse_base_annotation()
        self._generate_extended_annotation()
    
    def _parse_base_annotation(self):
        for bbox in self._sequence:
            x, y = bbox["x"]/100, bbox["y"]/100
            w, h = bbox["width"]/100, bbox["height"]/100
            frame = bbox["frame"]
            
            x *= self._capture_w
            w *= self._capture_w
            y *= self._capture_h
            h *= self._capture_h
            self._base_annotation[frame] = {
                "box": (
                    x,
                    y,
                    x+w,
                    y+h
                ),
                "enabled": bbox["enabled"]
            }
    
    def _generate_extended_annotation(self):
        # keys = list(self._base_annotation.keys())
        self._linear_interpolation()
        # for curr_key, next_key in zip(keys[:-1], keys[1:]):
        #     if self._base_annotation[curr_key]["enabled"]:
        #         pass
                # self._linear_interpolation(curr_key, next_key)
    
    def _linear_interpolation(self):
        keys = list(self._base_annotation.keys())
        top_left_x = [val["box"][0] for val in self._base_annotation.values()]
        top_left_y = [val["box"][1] for val in self._base_annotation.values()]
        bottom_right_x = [val["box"][2] for val in self._base_annotation.values()]
        bottom_right_y = [val["box"][3] for val in self._base_annotation.values()]
        
        t = list(range(keys[0], keys[-1]+1))
        tl_x = np.round(np.interp(t, keys, top_left_x))
        tl_y = np.round(np.interp(t, keys, top_left_y))
        br_x = np.round(np.interp(t, keys, bottom_right_x))
        br_y = np.round(np.interp(t, keys, bottom_right_y))
        
        boxes = np.array([tl_x, tl_y, br_x, br_y], dtype=np.int32)
        boxes = boxes.swapaxes(1, 0)
        self._extended_annotation = dict(zip(t, boxes))
    
    @property
    def annotation(self):
        return self._extended_annotation
    
    def get_current_annitation(self) -> tuple[int, int, int, int]|None:
        frame = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        if frame in self._extended_annotation:
            return self._extended_annotation[frame]
        return None
    
    def get_last_frame(self) -> int:
        return sorted(self._base_annotation.keys())[-1]
    
    def get_first_frame(self) -> int:
        return sorted(self._base_annotation.keys())[0]

def split_annotations(json_mini: str):
    with open(json_mini, "r") as json_file:
        annotations = json.load(json_file)
        for annotation in annotations:
            name = Path(annotation["video"]).stem
            with open(f"/home/poul/{name}.json", "w") as video_annotation:
                json.dump(annotation, video_annotation)


if __name__=="__main__":
    vid = "rotate_street_vid_1.webm"
    vid_path = f"/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/Annotated_Vidos/Videos/{vid}"
    
    annot_name = "Rotate_Street_Vid_1.json"
    annot_path = f"/home/poul/temp/{annot_name}"
    
    cap = cv2.VideoCapture(vid_path)
    
    annot_obj = AnnotationJson(annot_path, cap)
    annot_obj.selected_object = 0
    while cap.isOpened():
        waitkey_value = 1
        ret, frame = cap.read()
        if not ret:
            break
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        bbox = annot_obj.get_current_box()
        if bbox is not None:
            bbox = list(map(round, bbox))
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 0, 255))
            waitkey_value = 1
        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(waitkey_value)
        if key==ord("q"):
            break
    cap.release()