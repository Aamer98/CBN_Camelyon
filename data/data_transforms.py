from torchvision import datasets, models, transforms
import torch
import torchvision


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

target_transform = transforms.Compose([transforms.ToTensor()])