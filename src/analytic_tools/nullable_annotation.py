import re

class NullableAnnotation(list):
    def __init__(self, path_to_annotation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.read(path_to_annotation)
    
    def read(self, path):
        with open(path, "r") as annotation_file:
            self.process_file(annotation_file)
    
    def process_file(self, annotation_file):
        for line in annotation_file:
            self.parce_annotation_line(line)
    
    def parce_annotation_line(self, annotation_line):
        pts = self.string_to_pts(annotation_line)
        self.values_parser(pts)   
    
    def values_parser(self, values):
        pts_to_add = []
        if values is None:
            self.append(None)
            return
        for pnt in values:
            for i in pnt:
                pts_to_add.append(i)
        self.append(pts_to_add)
    
    def string_to_pts(self, annotation_line):
        splited_line = self.break_line_into_pts(annotation_line)
        if splited_line is None:
            return None
        values = [list(map(int, value.split(" "))) for value in splited_line]
        return values
    
    def break_line_into_pts(self, annotation_line):
        annotation_line = annotation_line.strip()
        if annotation_line == "None":
            return None
        splited_line = annotation_line.split("; ")
        splited_line[-1] = re.sub(";", "", splited_line[-1])
        return splited_line
