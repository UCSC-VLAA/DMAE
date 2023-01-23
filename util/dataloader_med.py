import copy
import os
import random
import numpy as np
import torchvision.transforms.functional
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
import medpy.io

NORMALIZATION_STATISTICS = {"self_learning_cubes_32": [[0.11303308354465243, 0.12595135887180803]],
                            "self_learning_cubes_64": [[0.11317437834743148, 0.12611378817031038]],
                            "lidc": [[0.23151727, 0.2168428080133056]],
                            "luna_fpr": [[0.18109835972793722, 0.1853707675313153]],
                            "lits_seg": [[0.46046468844492944, 0.17490586272419967]],
                            "pe": [[0.26125720740546626, 0.20363551346695796]]}


# -------------------------------------Data augmentation-------------------------------------
class Augmentation():
    def __init__(self, normalize):
        if normalize.lower() == "imagenet":
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif normalize.lower() == "chestx-ray":
            self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        elif normalize.lower() == "none":
            self.normalize = None
        else:
            print("mean and std for [{}] dataset do not exist!".format(normalize))
            exit(-1)

    def get_augmentation(self, augment_name, mode):
        try:
            aug = getattr(Augmentation, augment_name)
            return aug(self, mode)
        except:
            print("Augmentation [{}] does not exist!".format(augment_name))
            exit(-1)

    def basic(self, mode):
        transformList = []
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def _basic_crop(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
        else:
            transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_224(self, mode):
        transCrop = 224
        return self._basic_crop(transCrop, mode)

    def _basic_resize(self, size, mode="train"):
        transformList = []
        transformList.append(transforms.Resize(size))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_resize_224(self, mode):
        size = 224
        return self._basic_resize(size, mode)

    def _basic_crop_rot(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
            transformList.append(transforms.RandomRotation(7))
        else:
            transformList.append(transforms.CenterCrop(transCrop))

        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_rot_224(self, mode):
        transCrop = 224
        return self._basic_crop_rot(transCrop, mode)

    def _full(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full(transCrop, transResize, mode)

    def full_448(self, mode):
        transCrop = 448
        transResize = 512
        return self._full(transCrop, transResize, mode)

    def _full_colorjitter(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_colorjitter_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full_colorjitter(transCrop, transResize, mode)


from torch.utils.data import Dataset


# --------------------------------------------Downstream ChestX-ray14-------------------------------------------
class ChestX_ray14(Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None):
        self.img_list = []
        self.img_label = []

        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            return img, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            return [img, heatmap], label


class Covidx(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase

        self.classes = ['normal', 'positive', 'pneumonia', 'COVID-19']
        self.class2label = {c: i for i, c in enumerate(self.classes)}

        # collect training/testing files
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.datalist = list()
        for line in lines:
            patient_id, fname, label, source = line.split(' ')
            if phase in ('train', 'val'):
                self.datalist.append((os.path.join(data_dir, 'train', fname), label))
            else:
                self.datalist.append((os.path.join(data_dir, 'test', fname), label))

        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        label = self.class2label[label]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class Node21(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase

        if phase == 'train':
            with open(os.path.join(data_dir, 'train_mae.txt')) as f:
                fnames = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_mae.txt')) as f:
                fnames = f.readlines()
        fnames = [fname.strip() for fname in fnames]

        self.datalist = list()
        for line in fnames:
            fname, label = line.split(' ')
            self.datalist.append((os.path.join(data_dir, 'images', fname), int(label)))
        # metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        # self.datalist = list()
        # for i in range(len(metadata)):
        #     fname = metadata.loc[i, 'img_name']
        #     if fname in fnames:
        #         label = metadata.loc[i, 'label']
        #         self.datalist.append((os.path.join(data_dir, 'images', fname), label))

        # transforms
        self.transform = transform

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
        image = self.transform(image)
        label = torch.tensor([label], dtype=torch.float32)
        return image, label


class Chexpert(Dataset):
    def __init__(self, path_image, path_list, transform, num_class, reduct_ratio=1):
        self.path_list = path_list
        self.transform = transform
        self.path_image = path_image
        self.num_class = num_class

        self.img_list = []
        self.img_label = []

        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        with open(self.path_list, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image, lineItems[0])
                    imageLabel = lineItems[5:5 + 14]
                    for idx, _ in enumerate(imageLabel):
                        imageLabel[idx] = self.dict[0][imageLabel[idx]]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        label = torch.zeros((self.num_class), dtype=torch.float)

        for i in range(0, self.num_class):
            label[i] = self.img_label[idx][i]

        return img, label

    def __len__(self):
        return len(self.img_list)


'''
 NIH:(train:75312  test:25596)
 0:A 1:Cd 2:Ef 3:In 4:M 5:N 6:Pn 7:pnx 8:Co 9:Ed 10:Em 11:Fi 12:PT 13:H
 Chexpert:(train:223415 val:235)
 0:NF 1:EC 2:Cd 3:AO 4:LL 5:Ed 6:Co 7:Pn 8:A 9:Pnx 10:Ef 11:PO 12:Fr 13:SD
 combined:
 0: Airspace Opacity(AO)	1: Atelectasis(A)	2:Cardiomegaly(Cd)	3:Consolidation(Co)
 4:Edema(Ed)	5:Effusion(Ef)	6:Emphysema(Em)	7:Enlarged Card(EC)	8:Fibrosis(Fi)	
 9:Fracture(Fr)	10:Hernia(H)	11:Infiltration(In)	12:Lung lession(LL)	13:Mas(M)	
 14:Nodule(N)	15:No finding(NF)	16:Pleural thickening(PT)	17:Pleural other(PO)	18:Pneumonia(Pn)	
 19:Pneumothorax(Pnx)	20:Support Devices(SD)
'''


class combine(Dataset):
    def __init__(self, path_image_1, path_image_2, path_list_1, path_list_2, transform1, transform2, reduct_ratio=1):

        self.path_image_1 = path_image_1
        self.path_image_2 = path_image_2
        self.path_list_1 = path_list_1
        self.path_list_2 = path_list_2
        self.transform1 = transform1
        self.transform2 = transform2
        self.num_class = 21

        self.img_list = []
        self.img_label = []
        self.source = []
        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        self.dict_nih2combine = {0: 1, 1: 2, 2: 5, 3: 11, 4: 13, 5: 14, 6: 18, 7: 19, 8: 3, 9: 4, 10: 6, 11: 8, 12: 16,
                                 13: 10}
        self.dict_chex2combine = {0: 15, 1: 7, 2: 2, 3: 0, 4: 12, 5: 4, 6: 3, 7: 18, 8: 1, 9: 19, 10: 5, 11: 17, 12: 9,
                                  13: 20}

        with open(self.path_list_1, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.path_image_1, lineItems[0])
                    imageLabel = lineItems[1:14 + 1]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for i in range(14):
                        tmp_label[self.dict_nih2combine[i]] = float(imageLabel[i])
                    self.img_label.append(tmp_label)
                    self.source.append(0)

        # random.seed(1)
        # self.reduct_ratio = reduct_ratio
        # self.img_list = np.array(self.img_list)
        # self.img_label = np.array(self.img_label)
        # self.source=np.array(self.source)
        # index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        # self.img_list = self.img_list[index]
        # self.img_label = self.img_label[index]
        # self.source = self.source[index]
        # self.img_list = self.img_list.tolist()
        # self.img_label = self.img_label.tolist()
        # self.source=self.source.tolist()
        # index=sample(range(166739), len(self.img_list))
        cnt = -1

        with open(self.path_list_2, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                cnt += 1
                if line:  # and cnt in index:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image_2, lineItems[0])
                    imageLabel = lineItems[5:5 + 14]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for idx, _ in enumerate(imageLabel):
                        # if idx not in [5,8,2,6,10]:
                        #     continue
                        # if idx in [5,8]:
                        #     imageLabel[idx]=self.dict[0][imageLabel[idx]]
                        # elif idx in [2,6,10]:
                        #     imageLabel[idx]=self.dict[1][imageLabel[idx]]
                        # labels.append(float(imageLabel[idx]))
                        tmp_label[self.dict_chex2combine[idx]] = self.dict[0][imageLabel[idx]]
                    self.img_label.append(tmp_label)
                    self.source.append(1)
        self.img_label = torch.tensor(self.img_label)
        self.source = torch.tensor(self.source)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform1 is not None:
            img = self.transform1(img)
        # label = torch.zeros((self.num_class),dtype=torch.float)
        #
        # for i in range(0, self.num_class):
        #     label[i] = self.img_label[idx][i]

        return img, self.img_label[idx], self.source[idx]

    def __len__(self):
        return len(self.img_list)


class combine_semi(Dataset):
    def __init__(self, path_image_1, path_image_2, path_list_1, path_list_2, transform1, transform2, transform_semi,
                 reduct_ratio=1):

        self.path_image_1 = path_image_1
        self.path_image_2 = path_image_2
        self.path_list_1 = path_list_1
        self.path_list_2 = path_list_2
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform_semi = transform_semi
        self.num_class = 21

        self.img_list = []
        self.img_label = []
        self.source = []
        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        self.dict_nih2combine = {0: 1, 1: 2, 2: 5, 3: 11, 4: 13, 5: 14, 6: 18, 7: 19, 8: 3, 9: 4, 10: 6, 11: 8, 12: 16,
                                 13: 10}
        self.dict_chex2combine = {0: 15, 1: 7, 2: 2, 3: 0, 4: 12, 5: 4, 6: 3, 7: 18, 8: 1, 9: 19, 10: 5, 11: 17, 12: 9,
                                  13: 20}

        with open(self.path_list_1, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.path_image_1, lineItems[0])
                    imageLabel = lineItems[1:14 + 1]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for i in range(14):
                        # if i not in [0,9,2,8,7]:
                        #     continue
                        tmp_label[self.dict_nih2combine[i]] = float(imageLabel[i])
                    self.img_label.append(tmp_label)
                    self.source.append(0)

        # random.seed(1)
        # self.reduct_ratio = reduct_ratio
        # self.img_list = np.array(self.img_list)
        # self.img_label = np.array(self.img_label)
        # self.source=np.array(self.source)
        # index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        # self.img_list = self.img_list[index]
        # self.img_label = self.img_label[index]
        # self.source = self.source[index]
        # self.img_list = self.img_list.tolist()
        # self.img_label = self.img_label.tolist()
        # self.source=self.source.tolist()
        # index=sample(range(166739), len(self.img_list))
        cnt = -1

        with open(self.path_list_2, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                cnt += 1
                if line:  # and cnt in index:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image_2, lineItems[0])
                    imageLabel = lineItems[5:5 + 14]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for idx, _ in enumerate(imageLabel):
                        # if idx not in [2, 7, 8, 5, 10]:
                        #     continue
                        # if idx in [5,8]:
                        #     imageLabel[idx]=self.dict[0][imageLabel[idx]]
                        # elif idx in [2,6,10]:
                        #     imageLabel[idx]=self.dict[1][imageLabel[idx]]
                        # labels.append(float(imageLabel[idx]))
                        tmp_label[self.dict_chex2combine[idx]] = self.dict[0][imageLabel[idx]]
                    self.img_label.append(tmp_label)
                    self.source.append(1)
        self.img_label = torch.tensor(self.img_label)
        self.source = torch.tensor(self.source)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')
        img2 = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform1 is not None:
            img = self.transform1(img)
        # label = torch.zeros((self.num_class),dtype=torch.float)
        #
        # for i in range(0, self.num_class):
        #     label[i] = self.img_label[idx][i]

        return img, self.img_label[idx], self.source[idx], self.transform_semi(img2)

    def __len__(self):
        return len(self.img_list)


class FELIX(Dataset):
    def __init__(self, data_dir, file, augment, whitening=True):
        self.imgs = []
        self.labels = []
        self.normal_slice_indexes = []
        debug_length = 0
        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    data_path = os.path.join(data_dir, line)[:-1]
                    # print(data_path)
                    data = np.load(data_path).transpose([1, 0, 2, 3])
                    img = data[..., 0]
                    img = np.clip(img, -125, 275)
                    label = data[..., 1]
                    normal_slice_indexes = np.where(np.all(label <= 20, axis=(0, 1)))[0]
                    mean = 20.77
                    std = 102.79

                    img = (img - mean) / std
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)
                    self.imgs.append(img)
                    # self.labels.append(label)
                    self.normal_slice_indexes.append(normal_slice_indexes)
                    debug_length += 1

        self.whitening = whitening
        self.augment = augment

        self.data_len = len(self.imgs)

    def get_mid_slice_index(self, normal_indexes):
        index = None
        while index is None:
            index = np.random.choice(normal_indexes)
            if (index - 1) not in normal_indexes or (index + 1) not in normal_indexes:
                index = None
        return index

    def __len__(self):
        return 131072

    def __getitem__(self, index):
        index = index % self.data_len
        img = self.imgs[index]
        # label = self.labels[index]
        normal_slice_indexes = self.normal_slice_indexes[index]
        z = self.get_mid_slice_index(normal_slice_indexes)
        img = img[:, :, z - 1:z + 2]
        img = self.augment(img)
        # label = torch.tensor(label, dtype=torch.float)
        label = torch.zeros(0)
        return img, label
