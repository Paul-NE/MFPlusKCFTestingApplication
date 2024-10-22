from enum import Enum


class SmoothingClient:
	def update(self, roi):
		pass


class NoSmooth(SmoothingClient):
	def update(self, roi):
		return roi


class InterpolationSmooth(SmoothingClient):
	def __init__(self, next_value_priority):
		self.next_value_priority = next_value_priority
		self.previous_box = []

	def update(self, roi):
		if self.previous_box:
			self.previous_box = list(roi)
			return roi
		ret = [prev_value*(1 - self.next_value_priority) + current_value*self.next_value_priority
		        for prev_value, current_value in zip(self.previous_box, roi)]
		self.previous_box = list(roi)
		return ret


class PID(SmoothingClient):
	def __init__(self, K_p, K_i, K_d):
		self.K_p = K_p
		self.K_i = K_i
		self.K_d = K_d
		self.bboxes = []

	def update(self, roi):
		if not self.bboxes:
			self.bboxes = [list(roi) for _ in range(3)]
			return roi

		# self.error_window[:-1] = self.error_window[1:]
		# self.error_window[-1] = error
		# p = self.K_p * (self.error_window[-1] - self.error_window[-2])
		# i = self.K_i * self.error_window[-1]
		# d = self.K_d * (self.error_window[-1] - 2 * self.error_window[-2] + self.error_window[-3])
		# pid_shift = p + i + d
		# return pid_shift


class SmoothMode(Enum):
	NoSmooth = 1
	"""Default KCF, no change in inner/output ROI """
	SmoothOut = 2
	"""Smooth output BBox"""


class SmoothAlgorithm(Enum):
	Interpolation = 1
	PIDRegulator = 2


class SmoothFragment:
	def __init__(self, algorithm, mode, *args, **kwargs):
		self.mode = mode
		self.algorithm = algorithm

	def init(self):
		pass

	def update(self):
		pass

