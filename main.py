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

from data import *
from config import *
from models import *
from train import *


if __name__ == '__main__':
    
    xy_train = H5Dataset(feat_path = train_feat_path, 
                    lbl_path = train_label_path, 
                    transform = transform, 
                    target_transform = target_transform)

    xy_val = H5Dataset(feat_path = val_feat_path, 
                    lbl_path = val_label_path, 
                    transform = transform, 
                    target_transform = target_transform)

    train_dataloader = DataLoader(xy_train, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(xy_val, batch_size=32, shuffle=False)

    dataloaders = {'train' : train_dataloader, 'val' : val_dataloader}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft, train_logs = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
    
    

    