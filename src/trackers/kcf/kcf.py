from pathlib import Path
from enum import Enum
from time import time
import pickle
import shutil
import glob
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from .Modules.Correlation.gaussianCorrelation import *
from .Modules.KCF_functions import *
from .Modules.fhog import *
from geometry import BoundingBox, Point
from trackers.errors import NotInited

# class PID:
# 	def __init__(self, K_p, K_i, K_d, initial_error=0):
# 		self.K_p = K_p
# 		self.K_i = K_i
# 		self.K_d = K_d
# 		self.error_window = [initial_error]*3

# 	def update(self, error):
# 		self.error_window[:-1] = self.error_window[1:]
# 		self.error_window[-1] = error
# 		p = self.K_p * (self.error_window[-1] - self.error_window[-2])
# 		i = self.K_i * self.error_window[-1]
# 		d = self.K_d * (self.error_window[-1] - 2 * self.error_window[-2] + self.error_window[-3])
# 		pid_shift = p + i + d
# 		return pid_shift


# KCF tracker
@dataclass
class KCFFlags:
	hog: bool = True
	fixed_window: bool = True
	multiscale: bool = False
	rectFeatures: bool = False  # todo
	normalizeShift: bool = False
	smoothMotion: bool = False


@dataclass
class KCFHogParams:
	cell_size: int = 4
	NUM_SECTOR: int = 9


@dataclass
class KCFTrainParams:
	tmplsz: int = 64
	lambdar: float = 0.0001
	padding: float = 2.5
	output_sigma_factor: float = 0.125
	interp_factor: float = 0.012
	sigma: float = 0.6
	scale_step: float = 1.05
	scale_weight: float = 0.96
	resize_algorithm: int = cv2.INTER_NEAREST
	ceil_shift: float = 1
	delta_t: float = 0.5


@dataclass
class KCFDebugParams:
	saveDir: str = ""
	showedColorMap: bool = cv2.COLORMAP_RAINBOW
	saveTrackerParams: bool = False
	saveFeatures: bool = False  # todo
	showFeatures: bool = False
	showAlphaf: bool = False
	showTmpl: bool = False
	printTrackerParams: bool = False
	printMappData: bool = False


@dataclass
class KCFParams:
	flags: KCFFlags
	train: KCFTrainParams
	debug: KCFDebugParams
	hog: KCFHogParams


class KCFTracker:
	def __init__(self, params: KCFParams):
		flags = params.flags
		self.flags = params.flags
		train = params.train
		self.train_params = params.train
		self.hog_params = params.hog
		self.debug = params.debug

		self.inited = False
		self.min_roi_size = 32
		self.frame_shape = None

		self.roi_motion_velocity = [.0, .0]
		self._roi_smooth = None
		# if flags.smoothMotion:
		# 	K_p = 0.075; K_i = 0.1; K_d = 0.2; initial_error = 0
		# 	self.PIDs = [PID(K_p=K_p, K_i=K_i, K_d=K_d, initial_error=initial_error),
		# 	             PID(K_p=K_p, K_i=K_i, K_d=K_d, initial_error=initial_error)]

		self.resize_algorithm = train.resize_algorithm
		self.lambdar = train.lambdar  # regularization
		self.padding = train.padding  # extra area surrounding the target
		self.output_sigma_factor = train.output_sigma_factor  # bandwidth of gaussian target
		self.i = 0

		self.random_shift = [0., 0.]
		self.correl = True
		self.fixed_scale = False
		self.square_tmpl = False
		self._scaleW = 1.
		self._scaleH = 1.

		self._norm_shift = flags.normalizeShift

		if flags.hog:  # HOG feature
			# VOT
			self.interp_factor = train.interp_factor  # linear interpolation factor for adaptation. Original value 0.012
			self.sigma = train.sigma  # gaussian kernel bandwidth. original value 0.6
			# TPAMI   #interp_factor = 0.02   #sigma = 0.5
			self.cell_size = self.hog_params.cell_size  # HOG cell size, original value 4
			self.NUM_SECTOR = self.hog_params.NUM_SECTOR
			self._hogfeatures = True
		else:  # raw gray-scale image # aka CSK tracker
			self.interp_factor = train.interp_factor  # original value 0.075
			self.sigma = train.sigma  # original value 0.2
			self.cell_size = 1
			self._hogfeatures = False

		if flags.multiscale:
			self.template_size = train.tmplsz  # template size
			self.scale_step = train.scale_step  # scale step for multi-scale estimation
			self.scale_weight = train.scale_weight  # to downweight detection scores of other scales for added stability
		elif flags.fixed_window:
			self.template_size = train.tmplsz
			self.scale_step = 1
		else:
			self.template_size = 1
			self.scale_step = 1

		self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
		self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
		self.size_patch = [0, 0, 0]  # [int,int,int]
		self._scale = 1.  # float
		self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
		self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
		# numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
		self._tmpl = None
		# numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
		self.hann = None

	def subPixelPeak(self, left, center, right):
		divisor = 2 * center - right - left  # float
		# return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor
		return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor

	def createHanningMats(self):
		hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

		hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
		hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
		hann2d = hann2t * hann1t

		if self._hogfeatures:
			hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
			self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
		else:
			self.hann = hann2d
		self.hann = self.hann.astype(np.float32)

	def createGaussianPeak(self, sizey, sizex):
		syh, sxh = sizey // 2, sizex // 2
		output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
		mult = -0.5 / (output_sigma * output_sigma)
		y, x = np.ogrid[0:sizey, 0:sizex]
		y, x = (y - syh)**2, (x - sxh)**2
		res = np.exp(mult * (y + x))
		return fftd(res)

	def Correlation(self, x1, x2):
		return self.correlationMethod.calculate(x1, x2)

	def getFeatures(self, image, inithann, scale_adjust=1.0):
		extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
		cx = self._roi[0] + self._roi[2] // 2  # float
		cy = self._roi[1] + self._roi[3] // 2  # float
		if inithann:
			padded_w = self._roi[2] * self.padding
			padded_h = self._roi[3] * self.padding

			if self.template_size > 1:
				if padded_w >= padded_h:
					self._scale = padded_w / float(self.template_size)
				else:
					self._scale = padded_h / float(self.template_size)

				if self.fixed_scale:
					self._scaleW = padded_w / float(self.template_size)
					self._scaleH = padded_h / float(self.template_size)
				else:
					self._scaleW = self._scaleH = self._scale

				self._tmpl_sz[0] = int(padded_w / self._scale)
				self._tmpl_sz[1] = int(padded_h / self._scale)
				if self.square_tmpl:
					self._tmpl_sz[0] = self._tmpl_sz[1] = self.template_size
			else:
				self._tmpl_sz[0] = int(padded_w)
				self._tmpl_sz[1] = int(padded_h)
				self._scaleW = self._scaleH = self._scale = 1.

			if self._hogfeatures:
				self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (
							2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
				self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (
							2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
			else:
				self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
				self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

		extracted_roi[2] = int(scale_adjust * self._scaleW * self._tmpl_sz[0])
		extracted_roi[3] = int(scale_adjust * self._scaleH * self._tmpl_sz[1])
		extracted_roi[0] = int(cx - extracted_roi[2] / 2)
		extracted_roi[1] = int(cy - extracted_roi[3] / 2)

		z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
		if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
			z = cv2.resize(z, tuple(self._tmpl_sz), interpolation=self.resize_algorithm)

		if self._hogfeatures:
			mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
			mapp = fhog.getFeatureMaps(z, self.cell_size, mapp, NUM_SECTOR=self.NUM_SECTOR)
			mapp = fhog.normalizeAndTruncate(mapp, 0.2, NUM_SECTOR=self.NUM_SECTOR)
			mapp = fhog.PCAFeatureMaps(mapp, NUM_SECTOR=self.NUM_SECTOR)

			self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
			# (size_patch[2], size_patch[0]*size_patch[1])
			features_map = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T
		else:
			if z.ndim == 3 and z.shape[2] == 3:
				# z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
				features_map = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
			elif z.ndim == 2:
				features_map = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
			features_map = features_map.astype(np.float32) / 255.0 - 0.5
			self.size_patch = [z.shape[0], z.shape[1], 1]

		if inithann:
			self.createHanningMats()  # createHanningMats need size_patch

		features_map = self.hann * features_map
		return features_map

	def detect(self, z, x):
		k = self.Correlation(x, z)
		if not self.correl:
			res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))
		else:
			res = k
		
		self.res = res
		_, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int
		p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]

		if self._hogfeatures:
			if self.debug.showFeatures:
				debug_img = cv2.resize(res, (self.size_patch[1]*30, self.size_patch[0]*30), interpolation=cv2.INTER_NEAREST)
				debug_img = cv2.arrowedLine(debug_img, list(map(lambda x: x//2,debug_img.shape[::-1])), list(map(lambda x: x*30, pi)), (255, 255, 255), 1)
				cv2.imshow('HOG RES', debug_img.T)
			if self.debug.saveFeatures:
				cv2.imwrite(fr'/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/testing/trashcan_test/hog_features/result/{self.i}.jpg', cv2.resize(cv2.normalize(res, None) * 255, (self.size_patch[0] * 30, self.size_patch[1] * 30)))
		else:
			if self.debug.showFeatures:
				cv2.imshow('Normal RES', cv2.resize(cv2.normalize(res, None), (self.size_patch[0] * 10, self.size_patch[1] * 10)))
		if res.shape[1] - 1 > pi[0] > 0:
			p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
		if res.shape[0] - 1 > pi[1] > 0:
			p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

		p[0] -= res.shape[1] / 2.
		p[1] -= res.shape[0] / 2.
		if self._norm_shift and p[0] and p[1]:

			max_value = min(res.shape) / 2
			squared_dist = p[0]**2 + p[1]**2

			if squared_dist > max_value**2:
				p[0] *= max_value / squared_dist**0.5
				p[1] *= max_value / squared_dist**0.5

		return p, pv
	
	def compensateCameraMotion(self, T=(0, 0), S=(1, 1), R=None):
		Cx = self._roi[0] + self._roi[2]/2 + T[0]
		Cy = self._roi[1] + self._roi[3]/2 + T[1]
		
		self._roi[2] = max(self.min_roi_size, self._roi[2]*S[0])
		self._roi[3] = max(self.min_roi_size, self._roi[3]*S[1])
		self._roi[0] = Cx - self._roi[2] / 2
		self._roi[1] = Cy - self._roi[3] / 2

		self._roi[0] = 0 if self._roi[0] < 0 else self._roi[0]
		self._roi[1] = 0 if self._roi[1] < 0 else self._roi[1]
		self._roi[0] = (self.frame_shape[0]-self._roi[2]) if self._roi[0] > (self.frame_shape[0]-self._roi[2]) else self._roi[0]
		self._roi[1] = (self.frame_shape[1]-self._roi[3]) if self._roi[1] > (self.frame_shape[1]-self._roi[3]) else self._roi[1]


	def train(self, x, train_interp_factor):
		if not self.correl:
			k = self.Correlation(x, x) # !!!
			alphaf = complexDivision(self._prob, fftd(k) + self.lambdar) # !!!

		self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
		if not self.correl:
			self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf # !!!

		if self.debug.showTmpl:
			resize_factor = 5
			tmpl_img = cv2.normalize(self._tmpl, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			cv2.imshow('_tmpl', cv2.applyColorMap(cv2.resize(tmpl_img, np.array(tmpl_img.shape)[::-1] * resize_factor, interpolation=self.resize_algorithm), self.debug.showedColorMap))
		if self.debug.showAlphaf:
			alphaf_image_real = cv2.normalize(real(self._alphaf), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			alphaf_image_real = cv2.resize(alphaf_image_real, np.array(self._alphaf.shape)[:2][::-1] * resize_factor*2, interpolation=self.resize_algorithm)

			alphaf_image_real = cv2.applyColorMap(alphaf_image_real, self.debug.showedColorMap)
			cv2.imshow('_alphaf(real)', alphaf_image_real)

			alphaf_image_imag = cv2.normalize(imag(self._alphaf), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			alphaf_image_imag = cv2.resize(alphaf_image_imag, np.array(self._alphaf.shape)[:2][::-1] * resize_factor*2, interpolation=self.resize_algorithm)
			alphaf_image_imag = cv2.applyColorMap(alphaf_image_imag, self.debug.showedColorMap)
			cv2.imshow('_alphaf(imaginary)', alphaf_image_imag)

	def init(self, roi, image):
		self.frame_shape = image.shape[:2]
		self._roi = list(map(float, roi))
		
		# !!! Shifting
		self._roi[0] += self.random_shift[0]*self._roi[2]
		self._roi[1] += self.random_shift[1]*self._roi[3]

		# print(f"{self._roi[2:]=}")
		assert (roi[2] > 0 and roi[3] > 0)
		self._tmpl = self.getFeatures(image, 1)
		self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
		self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
		self.res = np.zeros(self._tmpl_sz)
		    # "hogfeatures": False,
            # "size_patch": None,
            # "resize_algorithm": None,
            # "showFeatures": False,
            # "sigma": None,
		self.correlationMethod = gaussianCorrelation(
			hogfeatures=self._hogfeatures, 
			size_patch=self.size_patch,
			resize_algorithm=self.resize_algorithm, 
			showFeatures=self.debug.showFeatures, 
			sigma=self.sigma)
		self.train(self._tmpl, 1.0)

		self.inited = True

	def update(self, image):
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

		loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

		if self.scale_step != 1:
			# Test at a smaller _scale
			new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
			# Test at a bigger _scale
			new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

			if self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
				loc = new_loc1
				peak_value = new_peak_value1
				self._scale /= self.scale_step
				self._roi[2] /= self.scale_step
				self._roi[3] /= self.scale_step
			elif self.scale_weight * new_peak_value2 > peak_value:
				loc = new_loc2
				peak_value = new_peak_value2
				self._scale *= self.scale_step
				self._roi[2] *= self.scale_step
				self._roi[3] *= self.scale_step

		if self.flags.smoothMotion:
			# self.roi_motion_velocity[0] = self.PIDs[0].update(loc[0] * self.cell_size * self._scale)
			# self.roi_motion_velocity[1] = self.PIDs[1].update(loc[1] * self.cell_size * self._scale)
			# self.roi_motion_velocity[0] = self.roi_motion_velocity[0]*(1-self.train_params.delta_t) + loc[0] * self.cell_size * self._scale * self.train_params.delta_t
			# self.roi_motion_velocity[1] = self.roi_motion_velocity[1]*(1-self.train_params.delta_t) + loc[1] * self.cell_size * self._scale * self.train_params.delta_t
			# self.roi_motion_velocity[0] += loc[0] * self.cell_size * self._scale * self.train_params.delta_t
			# self.roi_motion_velocity[1] += loc[1] * self.cell_size * self._scale * self.train_params.delta_t
			self.roi_motion_velocity[0] = loc[0] * self.cell_size * self._scaleW
			self.roi_motion_velocity[1] = loc[1] * self.cell_size * self._scaleH
		else:
			self.roi_motion_velocity[0] = loc[0] * self.cell_size * self._scaleW
			self.roi_motion_velocity[1] = loc[1] * self.cell_size * self._scaleH
		self._roi[0] = cx - self._roi[2] / 2.0 + self.roi_motion_velocity[0]
		self._roi[1] = cy - self._roi[3] / 2.0 + self.roi_motion_velocity[1]

		if self._roi[0] >= image.shape[1] - 1:
			self._roi[0] = image.shape[1] - 1
		if self._roi[1] >= image.shape[0] - 1:
			self._roi[1] = image.shape[0] - 1
		if self._roi[0] + self._roi[2] <= 0:
			self._roi[0] = -self._roi[2] + 2
		if self._roi[1] + self._roi[3] <= 0:
			self._roi[1] = -self._roi[3] + 2
		assert (self._roi[2] > 0 and self._roi[3] > 0)

		x = self.getFeatures(image, 0, 1.0)
		self.train(x, self.interp_factor)
		self.i += 1

		if self.flags.smoothMotion:
			if self._roi_smooth is None:
				self._roi_smooth = self._roi
				return self._roi, peak_value
			self._roi_smooth = [prev * (1 - self.train_params.delta_t) + current * self.train_params.delta_t for prev, current in zip(self._roi_smooth, self._roi)]
			return self._roi_smooth, peak_value
		return self._roi, peak_value

	def adjust_bounding_box(self, box:BoundingBox):
		self._roi[2] = box.width
		self._roi[3] = box.height
		return  BoundingBox(
			top_left_pnt = Point(
				x = self._roi[0],
				y = self._roi[1]
			),
			bottom_right_pnt = Point(
				x=self._roi[0]+self._roi[2],
				y=self._roi[1]+self._roi[3]
			)
		)


class KCFLogging(KCFTracker):
	instances_count = 0
	def __init__(self, params: KCFParams):
		KCFLogging.instances_count += 1
		self._id = KCFLogging.instances_count

		super().__init__(params)

		if self._id == 1 and self.debug.saveTrackerParams:
			root = r"/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/testing/last_kcfs"
			run_num = 0
			while os.path.isdir(root+rf"/run_{run_num}") and run_num < 100:
				run_num += 1

			current_root = root+rf"/run_{run_num}"
			os.mkdir(current_root)
			os.mkdir(current_root+r"/pkl")
			os.mkdir(current_root+r"/inited")
			KCFLogging._run_root = current_root
			self._run_root = KCFLogging._run_root

		self.time_for_last_update = -1.
		self.iterations = 0
		self.initial_roi = None
		self.rois = []
		self.feedbacks = []
		self.logs = []
		self.labels = []
		self._inited = False

		self.savedir = r"/media/poul/8A1A05931A057E07/Job_data/Datasets/Test_results/TestArcives/KCF_Correlation_Features/Correlation"
		self.save_features = False

		self.m = 20
		self.poly_degrease = 2
		self.n_frames_forward_predict = 2
		self.prediction_track = [None] * self.n_frames_forward_predict
		self.plot_inited = False
		self.plot_poly = False

		if self.debug.saveTrackerParams:
			with open(self._run_root + rf"/pkl/KCFParams_data_{self._id}.pkl", "wb") as params_pkl:
				pickle.dump(params, params_pkl, pickle.HIGHEST_PROTOCOL)

	def __del__(self):
		KCFLogging.instances_count -= 1

	def init(self, *args, comment: str = "", **kwargs):
		super().init(*args, **kwargs)

		if self.save_features:
			self.prepare_dirs()
			self.save_all_features(image=args[1], roi=args[0])

		self._inited = True
		self.initial_roi = list(self._roi)
		self.rois.append(list(self._roi))
		self.feedbacks.append(1.)
		if self.debug.saveTrackerParams:
			with open(self._run_root + rf"/inited/{self._id}.txt", "w") as roi_file:
				roi_file.write(f"id:{self._id}; action:created; {str(self.initial_roi)}, {comment}")
    
	def update(self, image):
		self.iterations += 1

		t = time()
		roi, peak = super().update(image)
		self.time_for_last_update = time() - t

		if self.save_features:
			self.save_all_features(image=image, roi=roi)

		self.rois.append(roi)
		self.feedbacks.append(peak)
		# roi[0] = self.initial_roi[0]
		# roi[1] = self.initial_roi[1]
		return roi
	
	def save_res(self):
		cv2.imwrite(fr'{self.savedir}/features/res/{self.iterations}.jpg', 
			  cv2.resize(cv2.normalize(self.res, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255, (self.size_patch[0] * 30, self.size_patch[1] * 30), interpolation=cv2.INTER_NEAREST))
		
	def save_tmpl(self):
		cv2.imwrite(fr'{self.savedir}/features/_tmpl/{self.iterations}.jpg', 
			  cv2.resize(cv2.normalize(self._tmpl, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255, (self.size_patch[0] * 30, self.size_patch[1] * 30), interpolation=cv2.INTER_NEAREST))
		
	def save_frame(self, frame, roi=None):
		roi = list(map(int, roi))		
		cv2.rectangle(frame, roi[:2], [roi[0]+roi[2], roi[1]+roi[3]], (0, 0, 0))
		cv2.imwrite(fr'{self.savedir}/video_tracker/{self.iterations}.jpg', frame)

	def save_unite(self, frame, roi=None):
		tmpl = cv2.resize(cv2.normalize(self._tmpl, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255, (frame.shape[1]//2, frame.shape[1]//2+frame.shape[1]%2), interpolation=cv2.INTER_NEAREST)
		res = cv2.resize(cv2.normalize(self.res, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255, (frame.shape[1]//2+frame.shape[1]%2, frame.shape[1]//2+frame.shape[1]%2), interpolation=cv2.INTER_NEAREST)

		roi = list(map(int, roi))
		cv2.rectangle(frame, roi[:2], [roi[0]+roi[2], roi[1]+roi[3]], (0, 0, 0))
		features = np.hstack((tmpl, res))
		unite = np.vstack((frame, features))

		text_box_h = 20
		text_box = np.ones((text_box_h ,frame.shape[1]), dtype=np.uint8)*255

		tmpl_min_max = [np.min(self._tmpl), np.max(self._tmpl)]
		tmpl_min_max = list(map(float, tmpl_min_max))
		tmpl_min_max[0], tmpl_min_max[1] = round(tmpl_min_max[0], 3), round(tmpl_min_max[1], 3)

		res_min_max = [np.min(self.res), np.max(self.res)]
		res_min_max = list(map(float, res_min_max))
		res_min_max[0], res_min_max[1] = round(res_min_max[0], 3), round(res_min_max[1], 3)

		text_tmpl = f'tmpl {tmpl_min_max}'
		text_res = f'res {res_min_max}'

		org = (0, text_box_h-5)
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.5
		color = (0, 0, 0)
		thickness = 1

		cv2.putText(text_box, text_tmpl + '    ' + text_res, org, font, fontScale, color, thickness, cv2.LINE_AA)
		unite = np.vstack((unite, text_box))
		cv2.imwrite(fr'{self.savedir}/unite/{self.iterations}.jpg', unite)
		
	def save_alphaf(self):
		pass
		
	def save_all_features(self, image=None, roi=None):
		self.save_res()
		self.save_tmpl()
		if image is not None:
			self.save_frame(frame=image, roi=roi)
			self.save_unite(frame=image, roi=roi)


	def prepare_dirs(self):
		shutil.rmtree(self.savedir)
		Path(fr"{self.savedir}/features/res/").mkdir(parents=True, exist_ok=True)
		Path(fr"{self.savedir}/features/_tmpl").mkdir(parents=True, exist_ok=True)
		Path(fr"{self.savedir}/video_tracker").mkdir(parents=True, exist_ok=True)
		Path(fr"{self.savedir}/unite").mkdir(parents=True, exist_ok=True)


	def predict(self):
		m = self.m  # number of frames to predict
		deg = self.poly_degrease  # degrease for polynomial regression
		n_frames_predict = self.n_frames_forward_predict

		if len(self.rois) >= m:
			points = np.array([[x+w/2, y+h/2] for x, y, w, h in self.rois[-m:]])
			points = points.T

			time_axis = np.arange(m)

			x_poly = np.polyfit(time_axis, points[0], deg)
			y_poly = np.polyfit(time_axis, points[1], deg)

			x = np.polyval(x_poly, m + n_frames_predict)
			y = np.polyval(y_poly, m + n_frames_predict)

			self.prediction_track[1:] = self.prediction_track[:-1]
			self.prediction_track[0] = [x, y]

			# plt.plot()
			if self.plot_poly:
				if not self.plot_inited:
					plt.style.use("seaborn-v0_8-darkgrid")
					plt.ion()

					fig, (axX, axY) = plt.subplots(1, 2)
					self.axX = axX
					self.axY = axY

					axX.set_ylim(0, 1000)
					axX.set_xlim(0, self.m +20)
					axY.set_ylim(0, 1000)
					axY.set_xlim(0, self.m +20)

					pointsX = np.transpose([[t, np.polyval(x_poly, t)] for t in range(self.m + 20)])
					pointsY = np.transpose([[t, np.polyval(y_poly, t)] for t in range(self.m + 20)])

					self.lineX, = axX.plot(*pointsX)
					self.lineY, = axY.plot(*pointsY)
					self.plot_inited = True
				else:
					plt.plot(x, y)

					self.lineX.set_ydata([np.polyval(x_poly, t) for t in range(self.m + 20)])
					self.lineY.set_ydata([np.polyval(y_poly, t) for t in range(self.m + 20)])

					plt.draw()
					plt.gcf().canvas.flush_events()

				return [x, y], [x_poly, y_poly]
		return None, None


class KCFTrackerNormal(KCFLogging):
	def __init__(self, params):
		super().__init__(params)
		self. _inited = False
	def init(self, image: np.ndarray, box: BoundingBox):
		roi = [box.top_left_pnt.x, box.top_left_pnt.y, box.width, box.height]
		# roi = [box.top_left_pnt.x, box.top_left_pnt.y, box.bottom_right_pnt.x, box.bottom_right_pnt.y]
		self._inited = True
		return super().init(roi, image)

	def update(self, image):
		if not self._inited:
			raise NotInited()
		res_roi = super().update(image)
		box = BoundingBox(
			top_left_pnt = Point(
				x=res_roi[0],
				y=res_roi[1]
			),
			bottom_right_pnt = Point(
				x=res_roi[0] + res_roi[2],
				y=res_roi[1] + res_roi[3]
			)
		)
		return box

	def adjust_bounding_box(self, box: BoundingBox):
		res = super().adjust_bounding_box(box)
		return res