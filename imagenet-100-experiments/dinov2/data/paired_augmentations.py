# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms
import torchvision.transforms.functional as F
import torch
from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
import random
class PairedRandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=transforms.InterpolationMode.BICUBIC):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        # Get random crop parameters once
        i, j, h, w = transforms.RandomResizedCrop.get_params(img=img1, scale=self.scale, ratio=self.ratio)
        
        # Apply same crop to both images
        img1_cropped = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        img2_cropped = F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)
        
        return img1_cropped, img2_cropped

class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        # Decide once whether to flip
        do_flip = torch.rand(1).item() < self.p
        if do_flip:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
        return img1, img2

import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

class PairedColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img1, img2):
        # Get the random ordering and parameters
        order, brightness_factor, contrast_factor, saturation_factor, hue_factor = ColorJitter.get_params(
            brightness=(max(0, 1 - self.brightness), 1 + self.brightness),
            contrast=(max(0, 1 - self.contrast), 1 + self.contrast),
            saturation=(max(0, 1 - self.saturation), 1 + self.saturation),
            hue=(-self.hue, self.hue)
        )
        
        # Define a function to apply jitter in a specific order
        def apply_jitter(img, order):
            for o in order:
                if o == 0:
                    img = F.adjust_brightness(img, brightness_factor)
                elif o == 1:
                    img = F.adjust_contrast(img, contrast_factor)
                elif o == 2:
                    img = F.adjust_saturation(img, saturation_factor)
                elif o == 3:
                    img = F.adjust_hue(img, hue_factor)
            return img
        
        # Apply the same jitter with the same order to both images
        img1 = apply_jitter(img1, order)
        img2 = apply_jitter(img2, order)
        
        return img1, img2
class PairedRandomApply:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img1, img2):
        # Decide once whether to apply the transform
        do_apply = torch.rand(1).item() < self.p
        
        if do_apply:
            img1, img2 = self.transform(img1, img2)
        
        return img1, img2

class PairedRandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img1, img2):
        do_grayscale = torch.rand(1).item() < self.p
        if do_grayscale:
            # Convert to grayscale (3-channel)
            img1 = F.rgb_to_grayscale(img1, num_output_channels=3)
            img2 = F.rgb_to_grayscale(img2, num_output_channels=3)
        return img1, img2


class PairedRandomSolarize:
    def __init__(self, threshold=128, p=0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, img1, img2):
        do_solarize = torch.rand(1).item() < self.p
        if do_solarize:
            img1 = F.solarize(img1, self.threshold)
            img2 = F.solarize(img2, self.threshold)
        return img1, img2
    

class PairedGaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0, kernel_size=9):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.kernel_size = kernel_size

    def __call__(self, img1, img2):
        # Decide whether to apply blur
        do_blur = random.random() < self.p
        if do_blur:
            # Sample sigma once
            sigma_val = random.uniform(self.radius_min, self.radius_max)

            # Apply the same blur to both images
            img1 = F.gaussian_blur(img1, [self.kernel_size, self.kernel_size], [sigma_val, sigma_val])
            img2 = F.gaussian_blur(img2, [self.kernel_size, self.kernel_size], [sigma_val, sigma_val])

        return img1, img2


logger = logging.getLogger("dinov2")


class PairedDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        # self.geometric_augmentation_global = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
        #         ),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #     ]
        # )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # global_transfo1_extra = GaussianBlur(p=1.0)

        # global_transfo2_extra = transforms.Compose(
        #     [
        #         GaussianBlur(p=0.1),
        #         transforms.RandomSolarize(threshold=128, p=0.2),
        #     ]
        # )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        # self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        # self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image_pair):
        image, image_d = image_pair
        #import sys
        #(image.size == image_d.size, file=sys.stderr)
        output = {}

        # Define paired global transformations:
        paired_crop = PairedRandomResizedCrop(
            size=self.global_crops_size,
            scale=self.global_crops_scale, 
            ratio=(3/4,4/3),  # default ratio range, same as RandomResizedCrop defaults
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        paired_flip = PairedRandomHorizontalFlip(p=0.5)

        # The original global color jitter:
        #   transforms.RandomApply([ColorJitter(..., ...)], p=0.8), 
        #   transforms.RandomGrayscale(p=0.2)
        # We'll recreate this using paired transforms:
        paired_color_jitter = PairedColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        paired_color_jitter_apply = PairedRandomApply(paired_color_jitter, p=0.8)
        paired_grayscale = PairedRandomGrayscale(p=0.2)

        # For global_transfo1: we had GaussianBlur(p=1.0) + normalize
        paired_blur_strong = PairedGaussianBlur(p=1.0, radius_min=0.1, radius_max=2.0, kernel_size=9)

        # For global_transfo2: we had GaussianBlur(p=0.1), RandomSolarize(p=0.2) + normalize
        paired_blur_weak = PairedGaussianBlur(p=0.1, radius_min=0.1, radius_max=2.0, kernel_size=9)
        paired_solarize = PairedRandomSolarize(threshold=128, p=0.2)

        def paired_global_transform_1(img1, img2):
            # Apply color jittering and grayscale
            img1, img2 = paired_color_jitter_apply(img1, img2)
            img1, img2 = paired_grayscale(img1, img2)
            #img1 = F.to_tensor(img1)
            #img2 = F.to_tensor(img2)
            # Apply strong blur
            img1, img2 = paired_blur_strong(img1, img2)
            # Normalize (already deterministic)
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
            return img1, img2

        def paired_global_transform_2(img1, img2):
            # Apply color jittering and grayscale
            img1, img2 = paired_color_jitter_apply(img1, img2)
            img1, img2 = paired_grayscale(img1, img2)
            #img1 = F.to_tensor(img1)
            #img2 = F.to_tensor(img2)
            # Apply weak blur and solarize
            img1, img2 = paired_blur_weak(img1, img2)
            img1, img2 = paired_solarize(img1, img2)
            # Normalize
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
            return img1, img2

        # Generate global crop #1
        # Use paired crop and flip but feed in (image, image) since we only have one image
        im1_cropped, im1_cropped_d = paired_crop(image, image_d)
        #print(im1_cropped.size, file=sys.stderr)
        im1_flipped, im1_flipped_d = paired_flip(im1_cropped, im1_cropped_d)
        #import sys
        #print(type(im1_flipped), file=sys.stderr)
        global_crop_1, global_crop_1_d = paired_global_transform_1(im1_flipped, im1_flipped_d)

        # Generate global crop #2
        im2_cropped, im2_cropped_d = paired_crop(image, image_d)
        #print(im2_cropped.size, file=sys.stderr)
        im2_flipped, im2_flipped_d = paired_flip(im2_cropped, im2_cropped_d)
        global_crop_2, global_crop_2_d = paired_global_transform_2(im2_flipped, im2_flipped_d)

        # Now we have our deterministic global crops
        output["global_crops"] = [global_crop_1, global_crop_2]
        #output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        output["global_crops_denoised"] = [global_crop_1_d, global_crop_2_d]
        #output["global_crops_teacher_denoised"] = [global_crop_1_d, global_crop_2_d]

        # Local crops remain unchanged:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
