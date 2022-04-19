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
from models


if __name__ == '__main__':
    
    xy_train = H5Dataset(feat_path = "/content/camelyonpatch_level_2_split_train_x.h5", 
                    lbl_path = "/content/camelyonpatch_level_2_split_train_y.h5", 
                    transform = transform, 
                    target_transform = target_transform)

    xy_val = H5Dataset(feat_path = "/content/camelyonpatch_level_2_split_valid_x.h5", 
                    lbl_path = "/content/camelyonpatch_level_2_split_valid_y.h5", 
                    transform = transform, 
                    target_transform = target_transform)


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)