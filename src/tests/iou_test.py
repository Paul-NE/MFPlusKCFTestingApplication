import unittest
import logging
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np

from geometry import BoundingBox, Point
from analytic_tools.ious import Analytics

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        # Sample Points and Bounding Boxes for Testing
        self.point1 = Point(0, 0)
        self.point2 = Point(4, 4)
        self.point3 = Point(1, 1)
        self.point4 = Point(5, 5)
        
        self.box1 = BoundingBox(self.point1, self.point2)  # Valid box
        self.box2 = BoundingBox(self.point3, self.point4)  # Overlapping box
        self.box_invalid = BoundingBox(self.point2, self.point1)  # Invalid box
        
        self.analytics = Analytics()

    def test_bounding_box_properties(self):
        # Test width, height, and center
        self.assertEqual(self.box1.width, 4)
        self.assertEqual(self.box1.height, 4)
        self.assertEqual(self.box1.center, Point(2, 2))

    def test_bounding_box_validation(self):
        # Test valid and invalid bounding boxes
        self.assertTrue(self.box1.validate_box_formation())
        self.assertFalse(self.box_invalid.validate_box_formation())

    def test_bounding_box_generate_from_list(self):
        # Test bounding box creation from list
        box = BoundingBox.generate_from_list([0, 0, 4, 4])
        self.assertEqual(box.top_left_pnt, Point(0, 0))
        self.assertEqual(box.bottom_right_pnt, Point(4, 4))
        self.assertTrue(box.validate_box_formation())
    
    def test_analytics_update(self):
        # Test updating the analytics with bounding boxes
        self.analytics.update(self.box1, self.box2)
        self.assertEqual(len(self.analytics._box1), 1)
        self.assertEqual(len(self.analytics._box2), 1)
        self.assertGreaterEqual(self.analytics._ious[0], 0)  # IOU is valid

    def test_analytics_summary(self):
        # Test the summary statistics
        self.analytics.update(self.box1, self.box2)
        summary = self.analytics.summary()
        
        self.assertIn("iou", summary)
        self.assertIn("iou_dispertion", summary)
        self.assertIn("map30", summary)
        self.assertIn("map50", summary)
        self.assertTrue(0 <= summary["map30"] <= 1)  # mAP values are between 0 and 1
    
    def test_analytics_multiple_updates(self):
        # Test multiple updates and summary
        self.analytics.update(self.box1, self.box2)
        self.analytics.update(self.box2, self.box1)
        summary = self.analytics.summary()
        
        self.assertEqual(len(self.analytics._box1), 2)
        self.assertEqual(len(self.analytics._box2), 2)
        self.assertAlmostEqual(summary["iou"], np.mean(self.analytics._ious))

if __name__=="__main__":
    unittest.main()