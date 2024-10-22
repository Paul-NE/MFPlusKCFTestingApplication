import csv
from test_read import Annotation

last_frame = 0
annot_arr = []
with open("/home/poul/StreetVid_4.csv", "r") as annot:
    next(annot)
    for a in annot:
        x, y, w, h, n = list(map(int, a.split(',')))
        print(x, y, w, h, n)
        annot_arr.append([x, y, w, h, n])

with open("/home/poul/test.txt", "w") as write:
    for x, y, w, h, n in annot_arr:
        write.write(f"{x} {y}; {x+w} {y+h};\n")