import cv2

import numpy as np

from .Modules.Correlation.gaussianCorrelation import *
from .Modules.KCF_functions import *
from .Modules.fhog import *
from geometry import BoundingBox, Point
from trackers.errors import NotInited
from trackers.tracker import AdjustableTracker


class CorellationTracker(AdjustableTracker):
    def __init__(self):
        self._inited: bool = False
        self._roi_box: BoundingBox|None = None

        self.resize_algorithm = cv2.INTER_NEAREST

        self.template_size = 32
        self.lambdar = 0.0001
        self.padding = 2.5
        self.output_sigma_factor = 0.125
        self.interp_factor = 0.012
        self.sigma = 0.6
        
        self.square_tmpl = True
        self._scaleW = 1.
        self._scaleH = 1.
        
        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.  # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None
        self.hann = None
        
    
    def init(self, image:np.ndarray, bounding_box: BoundingBox) -> None:
        self._roi_box = bounding_box
        top_left_point = bounding_box.top_left_pnt
        bottom_right_point = bounding_box.bottom_right_pnt
        roi = [
            top_left_point.x,
            top_left_point.y,
            bottom_right_point.x,
            bottom_right_point.y]
        self._roi = list(map(float, roi))
        
        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self._get_features(image, 1)
        
        # TODO refactor here ========================
        self._correlation_method = gaussianCorrelation_c(
            hogfeatures=False, 
            size_patch=self.size_patch,
            resize_algorithm=self.resize_algorithm, 
            showFeatures=False, 
            sigma=self.sigma)
        # TODO refactor here ========================
        
        self._train(self._tmpl, 1.0)
        self._inited = True
    
    def update(self, image: np.ndarray) -> BoundingBox:
        if not self._inited:
            raise NotInited()
        
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[2] + 1
        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._tmpl, self._get_features(image, 0, 1.0))
        
        roi_shift_by_x = loc[0] * self._scaleW
        roi_shift_by_y = loc[1] * self._scaleH
        self._roi[0] = cx - self._roi[2] / 2.0 + roi_shift_by_x
        self._roi[1] = cy - self._roi[3] / 2.0 + roi_shift_by_y

        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self._get_features(image, 0, 1.0)
        self._train(x, self.interp_factor)
        
        return BoundingBox(
            top_left_pnt=Point(
                x=self._roi[0],
                y=self._roi[1]
            ),
            bottom_right_pnt=Point(
                x=self._roi[0] + self._roi[2],
                y=self._roi[1] + self._roi[3]
            )
        )
    
    @property
    def inited(self) -> bool:
        return self._inited
    
    def adjust_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        bounding_box_width = bounding_box.width
        bounding_box_height = bounding_box.height
        center = self._roi_box.center
        
        res = BoundingBox(
            top_left_pnt=Point(
                x=center.x - bounding_box_width/2,
                y=center.y - bounding_box_height/2
            ),
            bottom_right_pnt=Point(
                x=center.x + bounding_box_width/2,
                y=center.y + bounding_box_height/2
            )
        )
        return res
    
    def _sub_pixel_peak(self, left, center, right):
        divisor = 2 * center - right - left
        return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor

    def _create_hanning_mats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t
        
        self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def _create_gaussian_peak(self, sizey, sizex):
        syh, sxh = sizey // 2, sizex // 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh)**2, (x - sxh)**2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def _correlation(self, x1, x2):
        return self._correlation_method.calculate(x1, x2)

    def _get_features(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] // 2  # float
        cy = self._roi[1] + self._roi[3] // 2  # float
        if inithann:
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding
            self._scale = max(padded_w, padded_h) / float(self.template_size)

            self._scaleW = padded_w / float(self.template_size)
            self._scaleH = padded_h / float(self.template_size)
            # self._scaleW = self._scaleH = self._scale

            self._tmpl_sz[0] = int(padded_w / self._scale)
            self._tmpl_sz[1] = int(padded_h / self._scale)
            
            if self.square_tmpl:
                self._tmpl_sz[0] = self._tmpl_sz[1] = self.template_size
            
            # Make _tmpl_sz even
            self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
            self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scaleW * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scaleH * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
            z = cv2.resize(z, tuple(self._tmpl_sz), interpolation=self.resize_algorithm)
        
        features_map = z if z.ndim == 2 else cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        
        features_map = features_map.astype(np.float32) / 255.0 - 0.5
        self.size_patch = [z.shape[0], z.shape[1], 1]
        if inithann:
            self._create_hanning_mats()

        features_map = self.hann * features_map
        return features_map

    def detect(self, z, x):
        correlation = self._correlation(x, z)
        
        _, max_point_value, _, max_point_indexes = cv2.minMaxLoc(correlation)
        p = [float(max_point_indexes[0]), float(max_point_indexes[1])]
        
        if correlation.shape[1] - 1 > max_point_indexes[0] > 0:
            p[0] += self._sub_pixel_peak(
                correlation[
                    max_point_indexes[1], 
                    max_point_indexes[0] - 1], 
                max_point_value, 
                correlation[
                    max_point_indexes[1], 
                    max_point_indexes[0] + 1])
        if correlation.shape[0] - 1 > max_point_indexes[1] > 0:
            p[1] += self._sub_pixel_peak(
                correlation[
                    max_point_indexes[1] - 1, 
                    max_point_indexes[0]], 
                max_point_value, 
                correlation[
                    max_point_indexes[1] + 1, 
                    max_point_indexes[0]])

        p[0] -= correlation.shape[1] / 2.
        p[1] -= correlation.shape[0] / 2.

        return p, max_point_value

    def _train(self, x, train_interp_factor):
        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
