from torchvision.datasets.folder import *
import os
import torchvision.transforms as transforms
import numpy as np


class ImageNet_Loader_Custom(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            salient_map_transform_mode='identical',
    ):
        super(ImageNet_Loader_Custom, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                     transform=transform,
                                                     target_transform=target_transform,
                                                     is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.salient_map_transform_mode = salient_map_transform_mode


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        salient_map_path = path.replace('imagenet', 'imagenet_saliency_map').replace('.JPEG', '_saliency.png')
        if os.path.exists(salient_map_path):
            salient_map = self.loader(salient_map_path)
        else:
            salient_map = transforms.ToPILImage()((np.ones([224, 224, 3]) * 255).astype(np.uint8))

        assert self.transform is not None
        if self.salient_map_transform_mode == 'identical':
            sample, salient_map = self.transform(sample, salient_map)
        else:
            raise NotImplementedError

        # if self.transform is not None:
        #     sample = self.transform(sample)

        return [sample, salient_map], target


    def __len__(self) -> int:
        return len(self.samples)