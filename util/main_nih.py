import os
import sys
import math
import numpy as np
from tqdm import tqdm
from optparse import OptionParser
# from shutil import copyfile
from sklearn.metrics._ranking import roc_auc_score
from utility import set_GPU
from model import densenet121
from dataloader_med import Augmentation, ChestX_ray14
import torch.nn.functional as F

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=0, type="string")
# network architecture
parser.add_option("--model", dest="model_name", help="DenseNet121", default="DenseNet121", type="string")
parser.add_option("--init", dest="init", help="Random | ImageNet | IntelligentAgent",
                  default="Random", type="string")
parser.add_option("--dense_unit", dest="dense_unit", help="number of dense units before last layer",
                  default=None, type="int")
parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                  default=14, type="int")
# data loader
parser.add_option("--data_set", dest="data_set", help="ChestX-ray14", default="ChestX-ray14", type="string")
parser.add_option("--normalization", dest="normalization", help="how to normalize data", default="default",
                  type="string")
parser.add_option("--augment", dest="augment", help="full", default="full", type="string")
parser.add_option("--num_meta", dest="num_meta", help="num of meta data", default=None, type="int")
parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
parser.add_option("--train_list", dest="train_list", help="file for training list",
                  default=None, type="string")
parser.add_option("--val_list", dest="val_list", help="file for validating list",
                  default=None, type="string")
parser.add_option("--test_list", dest="test_list", help="file for test list",
                  default=None, type="string")
# training detalis
parser.add_option("--mode", dest="mode", help="train | test | valid", default="train", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=1000, type="int")
parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam", type="string")
parser.add_option("--lr", dest="lr", help="learning rate", default=2e-4, type="float")
parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default=None, type="string")
parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
# pretrained weights
parser.add_option("--proxy_weights", dest="proxy_weights", help="Pretrained model path", default=None, type="string")

(options, args) = parser.parse_args()

assert options.init in ['Random',
                        'ImageNet',
                        'IntelligentAgent'
                        ]

num_gpu = set_GPU(options.GPU)

model_path = "./Models/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
output_path = "./Outputs/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

exp_name = "DenseNet121_ImageNet"


class setup_config():
    def __init__(self,
                 model_name='kmt',
                 init='ImageNet',
                 dense_unit=None,
                 num_class=14,
                 data_set=None,
                 normalization="default",
                 augment='full',
                 num_meta=None,
                 train_list='dataset/train_official.txt',
                 val_list='dataset/val_official.txt',
                 test_list='dataset/test_official.txt',
                 img_size=224,
                 img_depth=3,
                 batch_size=32,
                 num_epoch=1000,
                 optimizer=None,
                 lr=0.001,
                 lr_Scheduler=None,
                 patience=10,
                 proxy_weights=None,
                 ):
        self.model_name = model_name
        self.dense_unit = dense_unit
        self.num_class = num_class
        self.init = init
        self.exp_name = self.model_name + "_" + self.init
        self.data_set = data_set
        self.augment = augment
        self.img_size = img_size
        self.img_depth = img_depth
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.lr = lr
        self.lr_Scheduler = lr_Scheduler
        self.patience = patience

        self.data_dir = '/home/zongwei/mintong/data/images'
        self.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                           'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                           'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        self.num_meta = num_meta
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list

        self.activate = "sigmoid"
        if self.init == "IntelligentAgent":
            self.proxy_weights = proxy_weights

        if normalization == "default":
            if self.init.lower() == "random":
                self.normalization = "chestx-ray"
            elif self.init.lower() == "imagenet":
                self.normalization = "imagenet"
            elif self.init.lower() == "intelligentagent":
                self.normalization = "none"
        else:
            self.normalization = normalization

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config()

logs_path = os.path.join(model_path, exp_name)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

augment = Augmentation(normalize=config.normalization).get_augmentation(
    "{}_{}".format(config.augment, config.img_size), "train")

train_data = ChestX_ray14(config.data_dir, config.train_list,
                          augment=augment,
                          num_class=config.num_class, num_meta=config.num_meta,
                          batch_size=config.batch_size, img_depth=config.img_depth)

valid_data = ChestX_ray14(config.data_dir, config.val_list,
                          augment=augment,
                          num_class=config.num_class, num_meta=config.num_meta,
                          batch_size=config.batch_size, img_depth=config.img_depth)

output_file = os.path.join(output_path, exp_name + "_test.txt")
augment_test = Augmentation(normalize=config.normalization).get_augmentation(
    "{}_{}".format(config.augment, config.img_size), "valid")

test_data = ChestX_ray14(config.data_dir, config.test_list,
                         augment=augment_test,
                         num_class=config.num_class, num_meta=config.num_meta,
                         batch_size=config.batch_size, img_depth=config.img_depth)


def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    return outAUROC


def evaluate(model, dataloader):
    predict = []
    target = []
    model.eval()
    loss = torch.tensor(0.).cuda()

    tsne_fea = []
    tsne_label = []

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs, fea = model(inputs)

            tsne_fea.append(fea)
            labels_tsne = []
            for idx in range(len(target)):
                if target[idx][5] == 1:
                    labels_tsne.append(1)
                else:
                    labels_tsne.append(0)
            tsne_label.append(labels_tsne)

            predict.append(outputs)
            target.append(labels)
            loss += F.binary_cross_entropy(outputs, labels)

    tsne_fea = torch.stack(tsne_fea, dim=0)
    tsne_label = torch.stack(tsne_label, dim=0)

    # np.save('tsne_fea_all', tsne_fea.cpu().numpy())
    # np.save('tsne_lab_all', tsne_label.cpu().numpy())

    predict = torch.cat(predict, dim=0).cpu().numpy()
    target = torch.cat(target, dim=0).cpu().numpy()
    auc = computeAUROC(target, predict, 14)
    print(f'\n evalute: {auc}  avg_auc: {np.average(auc)} \n')
    loss = loss.item()
    print(f'evaluate loss: {loss}')
    model.train()

    return auc, loss


import torch.optim

model = densenet121().cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=0, eps=1e-4,
                                                          verbose=1)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
dataloader_eval = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)

best_loss = 100000
es = 0
best_test_auc = -1
best_epoch = -1
best = None

# state=torch.load('best.pt')
# model.load_state_dict(state)
# auc, test_loss=evaluate(model,dataloader_test)


for i in range(64):
    print(f'epoch: {i}/64')
    for img, label in tqdm(dataloader):
        img, label = img.cuda(), label.cuda()
        output = model(img)
        loss = F.binary_cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    auc, val_loss = evaluate(model, dataloader_eval)

    # early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        es = 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        es += 1
        print("Counter {} of 5".format(es))
        if es > 10:
            print("Early stopping now!")
            break

    # reduce lr on the plateau
    lr_scheduler.step(val_loss)

    if i % 1 == 0:
        print('test now: ')
        auc, test_loss = evaluate(model, dataloader_test)
        if np.average(auc) > best_test_auc:
            best_test_auc = np.average(auc)
            best = auc
            best_epoch = i
        print('best until now:')
        print(f'avg: {best_test_auc}  {best} best_epoch:{best_epoch}')
