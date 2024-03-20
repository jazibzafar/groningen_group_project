##
import os

import numpy as np
from ultralytics import YOLO
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from tifffile import imread


def img_loader(path):
    return imread(path)


class DatasetSegmentation(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.tile_path = os.path.join(self.data_path, 'tiles')
        self.mask_path = os.path.join(self.data_path, 'masks')
        self.tile_list = os.listdir(self.tile_path)  # this is a list
        self.mask_list = os.listdir(self.mask_path)

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, index):
        tile_name = self.tile_list[index]
        mask_name = self.mask_list[index]
        
        tile_path = os.path.join(self.tile_path, tile_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        
        tile = imread(tile_path)
        tile = ToTensor()(tile)
        torch.permute(tile, (2, 0, 1))
        mask = imread(mask_path)
        return tile, mask


class yolo_model(nn.Module):
    def __init__(self, in_channels= 4):
        super().__init__()
        model = YOLO("yolov8m-seg.pt")
        self.inp = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias= False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*(list(model.children())[1:-1]))
        self.out = nn.Conv2d(48, 1, 1)

    def forward(self, x):
        x = F.silu(self.inp(x))
        x = F.silu(self.backbone(x))
        x = F.silu(self.out(x))
        return x


path = "./treecover_segmentation_aerial_goettingen"
data = DatasetSegmentation(path)

train_loader = DataLoader(data,batch_size=1, shuffle=True)
model = yolo_model(in_channels=4)
#model = yolo_model(3)

# model.info(detailed=True)
# print(model)
# model = YOLO("yolov8m-seg.pt")
# newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
# print(newmodel)
# print(list(model.children())[-1])

##
print("Welcome to the trees_groningen")
print(torch.cuda.is_available(), torch.cuda.device_count())
num_epochs = 10
optimizer = optim.Adam(model.parameters())
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:  # Assuming train_loader is your DataLoader
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = nn.MSELoss()(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        running_loss += loss.item()