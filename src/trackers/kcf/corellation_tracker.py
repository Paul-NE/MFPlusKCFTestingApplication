import logging

import cv2
import numpy as np

from .Modules.Correlation.gaussianCorrelation import *
from .Modules.KCF_functions import *
from .Modules.fhog import *
from geometry import BoundingBox, Point
from trackers.errors import NotInited
from trackers.tracker import AdjustableTracker
from video_processors.window_manager import Window


class CorellationTracker(AdjustableTracker):
    @dataclass
    class Options:
        retrain_on_update: bool = True
        retrain_on_adjust: bool = True
        debug_visualization: bool = False
        debug_features_window_name: str = "Corellation_tmpl"
        debug_features_window_size: tuple[int, int] = (320, 320)
    
    @dataclass
    class MathParameters:
        template_size: int = 32
        padding: float = 2.5
        output_sigma_factor: float = 0.125
        interp_factor: float = 0.012
        sigma: float = 0.6

    def __init__(self, math_parameters: MathParameters|None=None, options: Options|None = None):
        self._inited: bool = False
        self._roi_box: BoundingBox|None = None

        self._resize_algorithm = cv2.INTER_NEAREST
        self._square_tmpl = True

        self._math_parameters = math_parameters if math_parameters is not None else self.MathParameters()
        self._options = options if options is not None else self.Options()
        
        self._scaleW = 1.
        self._scaleH = 1.
        self._scale = 1.
        
        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self._size_patch = [0, 0, 0]  # [int,int,int]
        self._tmpl: np.ndarray|None = None
        self._hann: np.ndarray|None = None
        self._image_buffer: np.ndarray|None = None
        
        self._debug_windows:dict[str, Window] = {}
        if self._options.debug_visualization:
            self._debug_windows[self._options.debug_features_window_name] = Window(self._options.debug_features_window_name)
            
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.info(f"Created with math parameters {self._math_parameters}")
    
    def init(self, image:np.ndarray, bounding_box: BoundingBox) -> None:
        self._roi_box = bounding_box
        top_left_point = bounding_box.top_left_pnt
        roi = [
            top_left_point.x,
            top_left_point.y,
            bounding_box.width,
            bounding_box.height]
        self._roi = list(map(float, roi))
        
        assert (self._roi_box.width > 0 and self._roi_box.height > 0)
        self._tmpl = self._get_features(image, 1)
        
        # TODO refactor here ========================
        self._correlation_method = gaussianCorrelation_c(
            hogfeatures=False, 
            size_patch=self._size_patch,
            resize_algorithm=self._resize_algorithm, 
            showFeatures=False, 
            sigma=self._math_parameters.sigma)
        # TODO refactor here ========================
        
        self._inited = True
        self._logger.info(f"Inited. Current roi: {self._roi_box}")
    
    def update(self, image: np.ndarray) -> BoundingBox:
        if not self._inited:
            raise NotInited()
        self._image_buffer = image
        
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
        
        if self._options.retrain_on_update:
            x = self._get_features(image, 0, 1.0)
            self._train(x, self._math_parameters.interp_factor)
        
        new_roi = BoundingBox(
            top_left_pnt=Point(
                x=self._roi[0],
                y=self._roi[1]
            ),
            bottom_right_pnt=Point(
                x=self._roi[0] + self._roi[2],
                y=self._roi[1] + self._roi[3]
            )
        )
        self._roi_box = new_roi
        self._logger.info(f"Updated. Current roi: {new_roi}")
        return new_roi
    
    @property
    def inited(self) -> bool:
        return self._inited
    
    def adjust_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        bounding_box_width = bounding_box.width
        bounding_box_height = bounding_box.height
        center = self._roi_box.center
        cx = self._roi[0] + self._roi[2] / 2
        cy = self._roi[1] + self._roi[3] / 2
        self._roi[0] = cx - bounding_box_width/2
        self._roi[1] = cy - bounding_box_height/2
        self._roi[2] = bounding_box_width
        self._roi[3] = bounding_box_height
        
        if self._options.retrain_on_adjust:
            x = self._get_features(self._image_buffer, 0, 1.0)
            self._train(x, self._math_parameters.interp_factor)
        
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
        hann2t, hann1t = np.ogrid[0:self._size_patch[0], 0:self._size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self._size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self._size_patch[0] - 1)))
        hann2d = hann2t * hann1t
        
        self._hann = hann2d
        self._hann = self._hann.astype(np.float32)

    def _create_gaussian_peak(self, sizey, sizex):
        syh, sxh = sizey // 2, sizex // 2
        output_sigma = np.sqrt(sizex * sizey) / self._math_parameters.padding * self._math_parameters.output_sigma_factor
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
            padded_w = self._roi[2] * self._math_parameters.padding
            padded_h = self._roi[3] * self._math_parameters.padding
            self._scale = max(padded_w, padded_h) / float(self._math_parameters.template_size)

            self._scaleW = padded_w / float(self._math_parameters.template_size)
            self._scaleH = padded_h / float(self._math_parameters.template_size)
            # self._scaleW = self._scaleH = self._scale

            self._tmpl_sz[0] = int(padded_w / self._scale)
            self._tmpl_sz[1] = int(padded_h / self._scale)
            
            if self._square_tmpl:
                self._tmpl_sz[0] = self._tmpl_sz[1] = self._math_parameters.template_size
            
            # Make _tmpl_sz even
            self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
            self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scaleW * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scaleH * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
            z = cv2.resize(z, tuple(self._tmpl_sz), interpolation=self._resize_algorithm)
        
        features_map = z if z.ndim == 2 else cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        
        features_map = features_map.astype(np.float32) / 255.0 - 0.5
        self._size_patch = [z.shape[0], z.shape[1], 1]
        if inithann:
            self._create_hanning_mats()

        features_map = self._hann * features_map
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
        
        if not self._options.debug_visualization:
            return
        tmpl_vis = cv2.resize(self._tmpl, self._options.debug_features_window_size, interpolation=cv2.INTER_NEAREST)
        tmpl_vis += np.abs(np.min(tmpl_vis))
        tmpl_vis /= np.max(tmpl_vis)
        tmpl_vis *= 255
        tmpl_vis = np.round(tmpl_vis)
        tmpl_vis = np.astype(tmpl_vis, np.uint8)
        self._debug_windows[self._options.debug_features_window_name].frame = tmpl_vis
    
    def get_debug_windows(self):
        return list(self._debug_windows.values())
