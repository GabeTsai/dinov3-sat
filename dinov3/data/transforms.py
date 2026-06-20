# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import random
from typing import Sequence

import torch
from torchvision.transforms import v2

logger = logging.getLogger("dinov3")

DEFAULT_GAMMA_SPECKLE_LOOKS_CHOICES = [1, 2, 4, 8, 16]
DEFAULT_GAMMA_SPECKLE_LOOKS_PROBS = [0.10, 0.20, 0.35, 0.25, 0.10]
DEFAULT_STUDENT_GAMMA_SIGMA_PIX_RANGE = (0.03, 0.10)
DEFAULT_TEACHER_GAMMA_SIGMA_PIX_RANGE = (0.015, 0.05)


def make_interpolation_mode(mode_str: str) -> v2.InterpolationMode:
    return {mode.value: mode for mode in v2.InterpolationMode}[mode_str]


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


def apply_d4(x, k: int):
    if k == 0:   # e: identity
        return x
    elif k == 1: # r90
        return torch.rot90(x, 1, [-2, -1])
    elif k == 2: # r180
        return torch.rot90(x, 2, [-2, -1])
    elif k == 3: # r270
        return torch.rot90(x, 3, [-2, -1])
    elif k == 4: # v: vertical flip
        return torch.flip(x, [-2])
    elif k == 5: # h: horizontal flip
        return torch.flip(x, [-1])
    elif k == 6: # t: transpose (diagonal reflection)
        return x.transpose(-2, -1)
    else:        # hvt: anti-diagonal reflection = t(r180)
        return torch.rot90(x, 2, [-2, -1]).transpose(-2, -1)


class RandomD4:
    def __init__(self, p: float = 1.0):
        self.p = p
    
    def __call__(self, x):
        if self.p == 0 or random.random() > self.p:
            return x
        return apply_d4(x, random.randint(0, 7))

class AddLogGammaSpeckle:
    """
    Add log-Gamma speckle to an already log-domain SAR tensor.
    Expects a floating point tensor in image units, typically [0, 1] before
    normalization. `sigma_pix` directly controls the pixel
    standard deviation of the additive log-noise in those units.
    """

    def __init__(
        self,
        p: float = 0.5,
        sigma_pix_range: tuple[float, float] = (0.025, 0.10),
        looks_choices: Sequence[int] = DEFAULT_GAMMA_SPECKLE_LOOKS_CHOICES,
        looks_probs: Sequence[float] = DEFAULT_GAMMA_SPECKLE_LOOKS_PROBS,
    ):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        if len(sigma_pix_range) != 2 or sigma_pix_range[0] < 0 or sigma_pix_range[1] < sigma_pix_range[0]:
            raise ValueError("sigma_pix_range must be a non-negative (min, max) pair")
        if len(looks_choices) != len(looks_probs):
            raise ValueError("looks_choices and looks_probs must have the same length")
        if any(look <= 0 for look in looks_choices):
            raise ValueError("looks_choices must be positive")
        if any(prob < 0 for prob in looks_probs) or sum(looks_probs) <= 0:
            raise ValueError("looks_probs must be non-negative and have positive sum")

        self.p = float(p)
        self.sigma_pix_range = (float(sigma_pix_range[0]), float(sigma_pix_range[1]))
        self.looks_choices = tuple(float(look) for look in looks_choices)
        prob_sum = float(sum(looks_probs))
        self.looks_probs = tuple(float(prob) / prob_sum for prob in looks_probs)

    def __call__(self, x):
        if self.p == 0 or random.random() > self.p:
            return x
        if not torch.is_floating_point(x):
            raise TypeError("AddLogGammaSpeckle expects a floating point tensor")

        work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_float = x.to(dtype=work_dtype)
        device = x_float.device

        looks_choices = torch.tensor(self.looks_choices, dtype=work_dtype, device=device)
        looks_probs = torch.tensor(self.looks_probs, dtype=work_dtype, device=device)
        looks = looks_choices[torch.multinomial(looks_probs, num_samples=1).item()]

        sigma_min, sigma_max = self.sigma_pix_range
        sigma_pix = torch.empty((), dtype=work_dtype, device=device).uniform_(sigma_min, sigma_max)
        strength = sigma_pix / torch.sqrt(torch.special.polygamma(1, looks))

        gamma = torch.distributions.Gamma(concentration=looks, rate=looks)
        noise = gamma.sample(x_float.shape).to(device=device, dtype=work_dtype)
        log_noise = torch.log(noise.clamp_min(torch.finfo(work_dtype).tiny))
        log_noise = log_noise - log_noise.mean()

        return (x_float + strength * log_noise).clamp(0.0, 1.0).to(dtype=x.dtype)
        
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [v2.ToImage(), v2.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built classification train transform\n{transform}")
    return transform


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        logger.info("resizing image as a square")
        size = (resize_size, resize_size)
        transform = v2.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        logger.info("resizing based on large side")
        transform = v2.Resize(size=None, max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = v2.Resize(resize_size, interpolation=interpolation)
        return transform


# Derived from make_classification_eval_transform() with more control over resize and crop
def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [v2.ToImage()]
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(v2.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built eval transform\n{transform}")
    return transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )


def voc2007_classification_target_transform(label, n_categories=20):
    one_hot = torch.zeros(n_categories, dtype=int)
    for instance in label.instances:
        one_hot[instance.category_id] = True
    return one_hot


def imaterialist_classification_target_transform(label, n_categories=294):
    one_hot = torch.zeros(n_categories, dtype=int)
    one_hot[label.attributes] = True
    return one_hot


def get_target_transform(dataset_str):
    if "VOC2007" in dataset_str:
        return voc2007_classification_target_transform
    elif "IMaterialist" in dataset_str:
        return imaterialist_classification_target_transform
    return None
