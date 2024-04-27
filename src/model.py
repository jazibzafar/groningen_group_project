import os

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import DatasetSegmentation, calculate_performance, GoeTransform


class YoloModel(nn.Module):
    def __init__(self, model_name: str = "coco", in_channels: int = 4):
        super().__init__()
        #TODO: change model loading paths
        if model_name == "coco":
            model = YOLO("./models/yolov8m-seg.pt")
        elif model_name == "retrained_coco":
            model = YOLO("yolov8m-seg.yaml")
        elif model_name == "retrained_empty":
            model = YOLO("yolov8m-seg.yaml")
        else:
            model = YOLO("yolov8m-seg.yaml")

        self.inp = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias= False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*(list(model.children())[1:-1]))
        self.out_1 = nn.Conv2d(48, 1, 1)
        self.out_2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = F.silu(self.inp(x))
        x = F.silu(self.backbone(x))
        x = F.silu(self.out_1(x))
        x = F.silu(self.out_2(x))
        return x
    
    def train(self,
              path = "./data/goettingen",
              input_size: int = 256,
              batch_size: int = 64,
              num_epochs: int = 100,
              optimizer_class: str = "adam",
              loss: str = "l2"):
        
        print(f"Creating dataset and loader for {path}")
        augment = GoeTransform(input_size=input_size)
        data = DatasetSegmentation(path, transform=augment)
        train_loader = DataLoader(data,batch_size=batch_size, shuffle=True)

        if optimizer_class == "adam":
            optimizer = optim.Adam(self.parameters())
        elif optimizer_class == "sgd":
            optimizer = optim.SGD(self.parameters())
        else:
            optimizer = optim.Adam(self.parameters())

        if loss == "l2":
            criterion = nn.MSELoss()
        elif loss == "l1":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        print(f"Starting training on {path}")
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            print("Epoch {}/{}".format(epoch+1, num_epochs))
            for inputs, labels in train_loader:  # Assuming train_loader is your DataLoader
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
                running_loss += loss.item()
                acc = calculate_performance(labels, outputs)
            print(f"Loss: {running_loss} Acc: {acc}")
        pass

    def test(self):
        pass
    def predict(self):
        pass
    def save(self):
        pass