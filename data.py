import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, feat_path, lbl_path, transform=None, target_transform = None):
        self.feat_path = feat_path
        self.lbl_path = lbl_path
        self.transform = transform
        self.target_transform = target_transform
        self.feature = None
        self.label = None
        with h5py.File(self.feat_path, 'r') as file:
            self.dataset_len = len(file['x'])
            
    def __getitem__(self, index):
        if self.feature is None:
            self.feature = np.array(h5py.File(self.feat_path, 'r')["x"])
            self.label = np.array(h5py.File(self.lbl_path, 'r')["y"])
        if self.transform:
            self.feature[index] = self.transform(self.feature[index])
        if self.target_transform:
            self.label[index] = self.target_transform(self.label[index])   
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.dataset_len





class H5Dataset_cbnTrain(torch.utils.data.Dataset):
    def __init__(self, feat_path, lbl_path, transform=None, target_transform = None):
        self.feat_path = feat_path
        self.lbl_path = lbl_path
        self.transform = transform
        self.target_transform = target_transform
        self.feature = None
        self.label = None
        with h5py.File(self.feat_path, 'r+') as file:
            self.dataset_len = len(file['x'])
            
    def __getitem__(self, index):
        if self.feature is None:
            self.feature = h5py.File(self.feat_path, 'r+')["x"]
            self.label = h5py.File(self.lbl_path, 'r+')["y"]
        if self.transform:
            self.feature[index] = self.transform(self.feature[index])
        if self.target_transform:
            self.label[index] = self.target_transform(self.label[index])   
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.dataset_len

