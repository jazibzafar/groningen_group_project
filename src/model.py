import os
from pathlib import Path

import time
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
    def __init__(self, args):
        super().__init__()
        self.args = args
        # initialize the network
        self.inp, self.backbone, self.out_1, self.out_2 = self.on_init_network(args.model_name,
                                                                               args.in_channels)
        # initialize the data
        self.train_data, self.validation_data, self.test_data = self.on_init_data(args.data_path,
                                                                                  args.input_size)
        # initialize the empty lists to store losses, these will be exported when saved.
        self.training_loss = []
        self.training_acc = []
        self.validation_loss = []
        self.validation_acc = []

    @staticmethod
    def on_init_network(model_name, in_channels):
        #TODO: change model loading paths
        if model_name == "coco":
            model = YOLO("./models/yolov8m-seg.pt")
        elif model_name == "pretrained_coco":
            model = YOLO("yolov8m-seg.yaml")
        elif model_name == "pretrained_empty":
            model = YOLO("yolov8m-seg.yaml")
        else:
            model = YOLO("yolov8m-seg.yaml")

        inp = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias= False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        backbone = nn.Sequential(*(list(model.children())[1:-1]))
        out_1 = nn.Conv2d(48, 1, 1)
        out_2 = nn.Upsample(scale_factor=2)
        return inp, backbone, out_1, out_2

    @staticmethod
    def on_init_data(data_path, input_size):
        data_generator = torch.Generator().manual_seed(2804)
        augment = GoeTransform(input_size=input_size)
        data = DatasetSegmentation(data_path, transform=augment)
        train_data, validation_data, test_data = random_split(data,
                                                              [0.7, 0.1, 0.2],
                                                              generator=data_generator)
        return train_data, validation_data, test_data

    def forward(self, x):
        x = F.silu(self.inp(x))
        x = F.silu(self.backbone(x))
        x = F.silu(self.out_1(x))
        x = F.silu(self.out_2(x))
        return x
    
    def train_on_data(self):
        best_vloss = 1000
        # ready the loaders
        train_loader = DataLoader(self.train_data,batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_data,batch_size=self.args.batch_size, shuffle=True)
        # set the optimizer
        if self.args.optimizer_class == "adam":
            optimizer = optim.Adam(self.parameters())
        elif self.args.optimizer_class == "sgd":
            optimizer = optim.SGD(self.parameters())
        else:
            optimizer = optim.Adam(self.parameters())
        # set the losses
        if self.args.loss == "l2":
            criterion = nn.MSELoss()
        elif self.args.loss == "l1":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        # begin training
        print(f"Starting training on {self.args.data_path}")
        start_time = time.time()
        for epoch in range(self.args.num_epochs):
            running_loss = 0.0
            acc= 0.0
            running_vloss = 0.0
            vacc = 0.0
            print("Epoch {}/{}".format(epoch+1, self.args.num_epochs))
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
            self.training_loss.append(running_loss)
            self.training_acc.append(acc)
            self.validation_loss.append(running_vloss)
            self.validation_acc.append(vacc)
            print(f"[train loss: {running_loss:.4f}] [train iou {acc:.4f}] [val loss: {running_vloss:.4f}] [val iou {vacc:.4f}]")
            # Saving best model
            if running_vloss < best_vloss:
                best_vloss = running_vloss
                if not(Path(self.args.save_path).exists()):
                    os.mkdir(self.args.save_path)
                self.save(self.args.save_path)
            # Performing Early stop
            early_stopping = EarlyStopping(tolerance=20, min_delta=0)
            early_stopping(running_loss, running_vloss)
            if early_stopping.early_stop:
                print("Early stopped at epoch:", epoch)
                break
        end_time = time.time()
        train_time = end_time - start_time
        print(f"training {self.args.num_epochs} took {train_time:.2f} seconds.")
        print(f"an average of {train_time / self.args.num_epochs:.2f} seconds/epoch.")
        pass

    def test(self):
        acc = 0.0
        # if self.args.path.split("\\")[-1] == "sliced":
        #     model_path = os.path.join(os.getcwd(), "models", self.path.split("\\")[-2], "best_model")
        # else:
        #     model_path = os.path.join(os.getcwd(), "models", self.path.split("\\")[-1], "best_model")
        if Path(self.args.model_path).exists():
            print("Loading Best Model for task")
            self.load_state_dict(torch.load(self.args.model_path))
        # print(f"Creating test datasets and loaders for {self.path}")
        # data = DatasetSegmentation(path)
        # _, _, test_data = random_split(data, [0.7, 0.1, 0.2], generator=self.data_generator)
        # del data

        test_loader = DataLoader(self.test_data,batch_size=1, shuffle=True)

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

    def save(self, save_path,):
        torch.save(self.state_dict(), save_path + "best_model.ckpt")
        self.write_list_to_file(save_path + "train_loss.txt", self.training_loss)
        self.write_list_to_file(save_path + "val_loss.txt", self.validation_loss)
        self.write_list_to_file(save_path + "train_iou.txt", self.training_acc)
        self.write_list_to_file(save_path + "val_iou.txt", self.validation_acc)
        pass

    @staticmethod
    def write_list_to_file(path, list_to_write):
        with open(path, 'w') as f:
            for line in list_to_write:
                f.write(f"{line}\n")
        return
