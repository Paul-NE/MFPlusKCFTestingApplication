import numpy as np
import cv2

from ..KCF_functions import *

class gaussianCorrelation:
    def __init__(self, *args, **kwargs) -> None:
        params = {
            "hogfeatures": False,
            "size_patch": None,
            "resize_algorithm": None,
            "showFeatures": False,
            "sigma": None,
            "saveFeatures": False,
        }
        for key in params:
            if key in kwargs:
                params[key] = kwargs[key]

        self.hogfeatures = params["hogfeatures"]
        self.size_patch = params["size_patch"]
        self.resize_algorithm = params["resize_algorithm"]
        self.showFeatures = params["showFeatures"]
        self.sigma = params["sigma"]
        self.saveFeatures = params["saveFeatures"]

    def calculate(self, x1, x2):
        if self.hogfeatures:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                # caux = rearrange(caux)
                c += caux
            c = rearrange(c)

            if self.showFeatures:
                cv2.imshow(
                    'Hog cor',
                    cv2.resize(cv2.normalize(c, None), (self.size_patch[1] * 30, self.size_patch[0] * 30),
                        interpolation=self.resize_algorithm))
            if self.saveFeatures:
                cv2.imwrite(fr'/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/testing/trashcan_test/hog_features/gaussianCorrelation/{self.i}.jpg', cv2.resize(cv2.normalize(c, None) * 255, (self.size_patch[0] * 30, self.size_patch[1] * 30), interpolation=self.resize_algorithm))
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)  # 'conjB=' is necessary!
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)
            if self.showFeatures:
                cv2.imshow('Normal cor', cv2.resize(cv2.normalize(c, None), (self.size_patch[0] * 30, self.size_patch[1] * 30), interpolation=self.resize_algorithm))

        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif x1.ndim == 2 and x2.ndim == 2:
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d
    
class gaussianCorrelation_c:
    def __init__(self, *args, **kwargs) -> None:
        params = {
            "hogfeatures": False,
            "size_patch": None,
            "resize_algorithm": None,
            "showFeatures": False,
            "sigma": None,
            "saveFeatures": False,
        }
        for key in params:
            if key in kwargs:
                params[key] = kwargs[key]

        self.hogfeatures = params["hogfeatures"]
        self.size_patch = params["size_patch"]
        self.resize_algorithm = params["resize_algorithm"]
        self.showFeatures = params["showFeatures"]
        self.sigma = params["sigma"]
        self.saveFeatures = params["saveFeatures"]

    def calculate(self, x1, x2):
        if self.hogfeatures:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                # caux = rearrange(caux)
                c += caux
            c = rearrange(c)

            if self.showFeatures:
                cv2.imshow(
                    'Hog cor',
                    cv2.resize(cv2.normalize(c, None), (self.size_patch[1] * 30, self.size_patch[0] * 30),
                        interpolation=self.resize_algorithm))
            if self.saveFeatures:
                cv2.imwrite(fr'/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/testing/trashcan_test/hog_features/gaussianCorrelation/{self.i}.jpg', cv2.resize(cv2.normalize(c, None) * 255, (self.size_patch[0] * 30, self.size_patch[1] * 30), interpolation=self.resize_algorithm))
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)  # 'conjB=' is necessary!
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)
        #     if self.showFeatures:
        #         cv2.imshow('Normal cor', cv2.resize(cv2.normalize(c, None), (self.size_patch[0] * 30, self.size_patch[1] * 30), interpolation=self.resize_algorithm))

        # if x1.ndim == 3 and x2.ndim == 3:
        #     d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
        #                 self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        # elif x1.ndim == 2 and x2.ndim == 2:
        #     d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
        #                 self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        # d = d * (d >= 0)
        # d = np.exp(-d / (self.sigma * self.sigma))

        return c