import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: Optional[bool] = None, intensityproperties: Optional[Dict] = None,
                 target_dtype: torch.dtype = torch.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict) or intensityproperties is None
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: torch.Tensor, seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: torch.Tensor, seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        image = image.to(dtype=self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            mask = seg >= 0 if seg is not None else torch.ones_like(image, dtype=torch.bool)
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / max(std, 1e-8)
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / max(std, 1e-8)
        return image

class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: torch.Tensor, seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        image = image.to(dtype=self.target_dtype)
        image = torch.clamp(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image

class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)

class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype, copy=False)
        image /= 255.
        return image