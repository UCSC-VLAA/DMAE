import medpy.io
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms


class Node21(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, img_size=(256, 256), normalize_tanh=False):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size

        if phase == 'train':
            with open(os.path.join(data_dir, 'train_mae.txt')) as f:
                fnames = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_mae.txt')) as f:
                fnames = f.readlines()
        fnames = [fname.strip() for fname in fnames]

        metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        self.datalist = list()
        for i in range(len(metadata)):
            fname = metadata.loc[i, 'img_name']
            if fname in fnames:
                label = metadata.loc[i, 'label']
                self.datalist.append((os.path.join(data_dir, 'images', fname), label))

        # transforms
        if phase == 'train':
            self.transforms = [
                transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor()
            ]
        else:
            self.transforms = [transforms.ToTensor()]
        if normalize_tanh:
            self.transforms.append(transforms.Normalize((0.5,), (0.5,)))
        self.transforms = transforms.Compose(self.transforms)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image, _ = medpy.io.load(fpath)
        image = image.astype(np.float)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        image = image.transpose(1, 0)
        image = Image.fromarray(image).convert('RGB')
        image = self.transforms(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
