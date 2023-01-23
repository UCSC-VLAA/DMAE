import copy
import random
import numbers
from collections.abc import Sequence
import warnings
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class custom_train_transform(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=3,
                 mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252], inplace=False):
        super().__init__(size)
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img, heatmap):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        heatmap = TF.resized_crop(heatmap, i, j, h, w, self.size, 0)
        rand_num = random.random()

        if rand_num > 0.5:
            img = TF.hflip(img)
            heatmap = TF.hflip(heatmap)

        img = TF.to_tensor(img)
        heatmap = TF.to_tensor(heatmap)
        img = TF.normalize(img, self.mean, self.std, self.inplace)
        return img, heatmap
