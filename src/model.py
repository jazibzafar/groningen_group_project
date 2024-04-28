import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.utils import DatasetSegmentation, calculate_performance, GoeTransform, EarlyStopping


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

        self.data_generator = torch.Generator().manual_seed(2804)

    def forward(self, x):
        x = F.silu(self.inp(x))
        x = F.silu(self.backbone(x))
        x = F.silu(self.out_1(x))
        x = F.silu(self.out_2(x))
        return x
    
    def train_on_data(self,
              path = "./data/goettingen",
              input_size: int = 256,
              batch_size: int = 64,
              num_epochs: int = 100,
              optimizer_class: str = "adam",
              loss: str = "l2"):
        
        training_loss = []
        training_acc = []
        validation_loss = []
        validation_acc = []
        best_vloss = 1000
            
        print(f"Creating training and validation datasets and loaders for {path}")
        augment = GoeTransform(input_size=input_size)
        data = DatasetSegmentation(path, transform=augment)
        train_data, validation_data, _ = random_split(data, [0.7, 0.1, 0.2], generator=self.data_generator)
        del data
        
        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_data,batch_size=batch_size, shuffle=True)
        
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
            acc= 0.0
            running_vloss = 0.0
            vacc = 0.0
            print("Epoch {}/{}".format(epoch+1, num_epochs))

            # Training one epoch
            self.train(True)
            for inputs, labels in train_loader:  # Assuming train_loader is your DataLoader
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                outputs = outputs.squeeze(1)  # labels are bs x w x h while outputs are bs x 1 x w x h; this corrects it
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
                running_loss += loss.item()
                acc += calculate_performance(labels, outputs)
            running_loss /= len(train_loader)
            acc /= len(train_loader)

            # Validating
            self.eval()
            with torch.no_grad():
                for vinputs, vlabels in validation_loader:
                    voutputs = self(vinputs)
                    voutputs = voutputs.squeeze(1)
                    vloss = criterion(voutputs, vlabels)
                    running_vloss += vloss
                    vacc += calculate_performance(vlabels, voutputs)
            running_vloss /= len(validation_loader)
            vacc /= len(validation_loader)

            # appending average epoch losses and acc to their specific lists
            training_loss.append(running_loss)
            training_acc.append(acc)
            validation_loss.append(running_vloss)
            validation_acc.append(vacc)
            print(f"Training Loss: {running_loss} Training Acc: {acc} Validation Loss: {running_vloss} Validation Acc: {vacc}")
            
            # Saving best model
            if running_vloss < best_vloss:
                best_vloss = running_vloss
                if path.split("\\")[-1] == "sliced":
                    save_path = os.path.join(os.getcwd(), "models", path.split("\\")[-2])
                else:
                    save_path = os.path.join(os.getcwd(), "models", path.split("\\")[-1])
                if not(Path(save_path).exists()):
                    os.mkdir(save_path)
                self.save(save_path)
            
            # Performing Early stop
            early_stopping = EarlyStopping(tolerance=20, min_delta=0)
            early_stopping(running_loss, running_vloss)
            if early_stopping.early_stop:
                print("Early stopped at epoch:", epoch)
                break
        pass

    def test(self,
             path = "./data/goettingen",
             input_size: int = 256):
        acc = 0.0
        if path.split("\\")[-1] == "sliced":
            model_path = os.path.join(os.getcwd(), "models", path.split("\\")[-2], "best_model")
        else:
            model_path = os.path.join(os.getcwd(), "models", path.split("\\")[-1], "best_model")
        if Path(model_path).exists():
            print("Loading Best Model for task")
            self.load_state_dict(torch.load(model_path))
        print(f"Creating test datasets and loaders for {path}")
        data = DatasetSegmentation(path)
        _, _, test_data = random_split(data, [0.7, 0.1, 0.2], generator=self.data_generator)
        del data

        test_loader = DataLoader(test_data,batch_size=1, shuffle=True)

        print("Starting Testing")
        self.eval()
        with torch.no_grad():
            for vinputs, vlabels in test_loader:
                voutputs = self(vinputs)
                voutputs = voutputs.squeeze(1)
                acc += calculate_performance(vlabels, voutputs)
        acc /= len(test_loader)
        print(f"Testing Acc: {acc}")

    def predict(self):
        pass
    def save(self, save_path):
        torch.save(self.state_dict(), save_path + "/best_model")
        pass