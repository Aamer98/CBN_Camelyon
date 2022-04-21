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
import pickle
import pandas as pd


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_freq, exp_name):
    since = time.time()

    train_accuracies = []
    train_loss = []
    val_accuracies = []
    val_loss = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_accuracies.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc)
                val_loss.append(epoch_loss)                

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        if num_epochs%epoch==0:
            torch.save(model.state_dict(), './logs/{}/epoch_{}.pth'.format(exp_name, epoch))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), './logs/{}/best_model.pth'.format(exp_name, epoch))

    df = pd.DataFrame(list(zip(train_accuracies, train_loss, val_accuracies, val_loss)), columns = ['train_accuracies', 'train_loss', 'val_accuracies', 'val_loss'])
    df.to_csv('./logs/{}/train_logs.csv'.format(exp_name))

    return model
    
