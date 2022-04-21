import copy
import argparse
import os
import time
import pickle
from PIL import Image
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

import data
import configs
from data import H5Dataset
from models.resnet_BN import resnet18
from trainer import train_model

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', required=True, help="Experiment name")
#parser.add_argument('-save_path', default='logs/', help="Path to save trained models")
parser.add_argument('-lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-batch_size', default=2, type=int, help="Batch size")
parser.add_argument('-seed', default=0, type=int, help="Seed for weight init")
parser.add_argument('-epochs', default=25, type=int, help="Number of epochs")
parser.add_argument('-save_freq', default=5, type=int, help="Frequency to save trained models")
args = parser.parse_args()
print(args)


def main():
    
    # seed the random number generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    save_path = 'logs/{}'.format(args.exp_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    transform = transforms.Compose([transforms.ToPILImage(), 
                                transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.ToPILImage(), 
                                       transforms.ToTensor()])

    xy_train = H5Dataset(feat_path = train_feat_path, 
                    lbl_path = train_label_path, 
                    transform = transform, 
                    target_transform = target_transform)

    xy_val = H5Dataset(feat_path = val_feat_path, 
                    lbl_path = val_label_path, 
                    transform = transform, 
                    target_transform = target_transform)

    train_dataloader = DataLoader(xy_train, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(xy_val, batch_size=32, shuffle=True)

    dataloaders = {'train' : train_dataloader, 'val' : val_dataloader}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    

    model_ft = resnet18()
    num_classes = 2
    model_ft.fc = nn.Linear(512, num_classes)




    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                             save_freq = args.save_freq, exp_name = args.exp_name, num_epochs=25)




if __name__ == '__main__':
    
    main()

    
    

    