import unittest
import logging
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import cv2

from generators.fast_pts_generator import FastPtsGenerator
from generators.smart_grid_pts_generator import SmartGridPtsGenerator
from generators.limit_pts_generator import limit_pts_generator
from geometry import BoundingBox, Point

if __name__=="__main__":
    gen = FastPtsGenerator()
    gen = SmartGridPtsGenerator()
    gen_limited = limit_pts_generator(gen, 0.5)
    img = cv2.imread("/home/poul/Изображения/ldEjCje_tCSmL31mvTL5bzhJ80Y8dmrUh2SemYsfeCcxRNHmLTFCnug8dbS-e0tUYWSQpfCeojNiq77r0H0KOdAj.png")
    box = BoundingBox(
        top_left_pnt=Point(0, 0),
        bottom_right_pnt=Point(800, 400)
    )
    res = gen_limited.gen(box, img)
    for point in res:
        cv2.circle(img, list(map(int, point)), 2, (0, 0, 255), -1)
        
    cv2.imshow("img", img)
    cv2.waitKey(0)