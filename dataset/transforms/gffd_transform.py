import random
import cv2
import numpy as np
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations import crop
from albumentations import Compose, HorizontalFlip, OneOf, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, \
    GaussianBlur, HueSaturationValue, RandomBrightnessContrast, ColorJitter, ToGray, Cutout
from albumentations.pytorch.functional import img_to_tensor
import torch
import torchvision.transforms as transforms


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class DownUpResize(DualTransform):
    def __init__(self, max_side, always_apply=False, p=1):
        super(DownUpResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.inter_methods = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR]
        self.ratios = [2, 3, 4]

    def apply(self, img, **params):
        interpolation_down = random.choice(self.inter_methods)
        interpolation_up = random.choice(self.inter_methods)
        h, w = img.shape[:2]
        if h > self.max_side or w > self.max_side:
            r = random.choice(self.ratios)
            img = cv2.resize(img, (int(w) // r, int(h) // r), interpolation=interpolation_down)
            # img_up = cv2.resize(img, (self.max_side, self.max_side), interpolation=interpolation_down)
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_up,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")

class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"


# def create_train_transforms(size=300):
#     return Compose([
#         ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
#         GaussNoise(p=0.1),
#         GaussianBlur(blur_limit=3, p=0.05),
#         HorizontalFlip(),
#         OneOf([
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
#             IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
#         ], p=1),
#         PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
#         ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
#     ]
#     )

def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        HueSaturationValue(p=0.1),
        RandomBrightnessContrast(p=0.1),
        # ColorJitter(p=0.2),
        # ToGray(p=0.1),
        OneOf([
            DownUpResize(max_side=size),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.1),
    ]
    )

def create_train_transforms_s(size=300):
    return Compose([
        HorizontalFlip(),
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ]
    )

def create_dct_transforms(conf):
    train_normalize = []
    train_normalize_y = transforms.Normalize(mean=conf['normalize_dct_y']['mean'],
                                             std=conf['normalize_dct_y']['std'])
    train_normalize_cb = transforms.Normalize(mean=conf['normalize_dct_cb']['mean'],
                                              std=conf['normalize_dct_cb']['std'])
    train_normalize_cr = transforms.Normalize(mean=conf['normalize_dct_cr']['mean'],
                                              std=conf['normalize_dct_cr']['std'])
    train_normalize.append(train_normalize_y)
    train_normalize.append(train_normalize_cb)
    train_normalize.append(train_normalize_cr)
    return train_normalize

def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


def direct_val(imgs, size):
    # img ?????????RGB??????
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    transforms = create_val_transforms(size)
    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}
    imgs = [img_to_tensor(transforms(image=each)['image'], normalize).unsqueeze(0) for each in imgs]
    imgs = torch.cat(imgs)

    return imgs

if __name__ == '__main__':
    create_train_transforms(380)