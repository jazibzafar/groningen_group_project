import os
from pathlib import Path

import time
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

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
        # a fixed subset of test dataset for prediction
        self.predict_data = Subset(self.test_data, [0, 1, 2])


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
                                                              [0.8, 0.1, 0.1],
                                                              generator=data_generator)
        return train_data, validation_data, test_data

    def forward(self, x):
        x = F.silu(self.inp(x))
        x = F.silu(self.backbone(x))
        x = F.silu(self.out_1(x))
        x = F.silu(self.out_2(x))
        return x

    def train_validate_one_epoch(self, train_loader, validation_loader, epoch, optimizer, criterion):
        running_loss = 0.0
        acc = 0.0
        running_vloss = 0.0
        vacc = 0.0
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
        return running_loss, acc, running_vloss, vacc

    def train_on_data(self):
        best_vloss = 1000
        # ready the loaders
        train_loader = DataLoader(self.train_data,batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_data,batch_size=self.args.batch_size, shuffle=True)
        # set the optimizer and losses
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        # lists for losses and accuracies
        training_loss = []
        validation_loss = []
        training_accuracy = []
        validation_accuracy = []
        testing_accuracy = []
        # begin training
        print(f"Starting training on {self.args.data_path}")
        start_time = time.time()
        for epoch in range(self.args.num_epochs):
            print("Epoch {}/{}".format(epoch + 1, self.args.num_epochs))
            train_loss, train_acc, train_vloss, train_vacc = self.train_validate_one_epoch(train_loader,
                                                                                           validation_loader,
                                                                                           epoch,
                                                                                           optimizer,
                                                                                           criterion)
            # appending average epoch losses and acc to their specific lists
            training_loss.append(train_loss)
            training_accuracy.append(train_acc)
            validation_loss.append(train_vloss)
            validation_accuracy.append(train_vacc)
            print(f"[train loss: {train_loss:.4f}] [train iou {train_acc:.4f}] [val loss: {train_vloss:.4f}] [val iou {train_vacc:.4f}]")
            self.save(self.args.save_path)
            # test the data to get test loss and accuracy
            test_acc = self.test()
            testing_accuracy.append(test_acc)
            print(f"[test iou: {test_acc:.4f}]")

            # Performing Early stop
            early_stopping = EarlyStopping(tolerance=20, min_delta=0)
            early_stopping(train_loss, train_vloss)
            if early_stopping.early_stop:
                print("Early stopped at epoch:", epoch)
                break
        end_time = time.time()
        train_time = end_time - start_time
        print(f"training {self.args.num_epochs} took {train_time:.2f} seconds.")
        print(f"an average of {train_time / self.args.num_epochs:.2f} seconds/epoch.")
        return training_loss, training_accuracy, validation_loss, validation_accuracy, testing_accuracy

    def test(self):
        acc = 0.0
        # if Path(self.args.save_path).exists():
        #     print("Loading Best Model for task")
        #     self.load_state_dict(torch.load(self.args.save_path))

        test_loader = DataLoader(self.test_data,batch_size=1, shuffle=False)
        print("Starting Testing")
        self.eval()
        with torch.no_grad():
            for vinputs, vlabels in test_loader:
                voutputs = self(vinputs)
                voutputs = voutputs.squeeze(1)
                acc += calculate_performance(vlabels, voutputs)
        acc /= len(test_loader)
        print(f"Testing iou: {acc}")
        return acc

    def train_only_output_layer(self):
        for i, param in enumerate(self.parameters()):
            if i < 3:  # Adjust the number based on the layers you want to freeze
                param.requires_grad = False
        trL, trA, vaL, vaA, teA = self.train_on_data()
        self.write_list_to_file(self.args.save_path + "leval_train_loss.txt", trL)
        self.write_list_to_file(self.args.save_path + "leval_train_acc.txt", trA)
        self.write_list_to_file(self.args.save_path + "leval_val_loss.txt", vaL)
        self.write_list_to_file(self.args.save_path + "leval_val_acc.txt", vaA)
        self.write_list_to_file(self.args.save_path + "leval_test_loss.txt", trA)
        # return trL, trA, vaL, vaA, teA
        return

    def reload_model(self):
        self.inp, self.backbone, self.out_1, self.out_2 = self.on_init_network(self.args.model_name,
                                                                               self.args.in_channels)
        return

    def predict(self):
        # if Path(self.args.save_path).exists():
        #     print("Loading Best Model for task")
        #     self.load_state_dict(torch.load(self.args.save_path))

        predict_loader = DataLoader(self.predict_data, batch_size=1, shuffle=False)
        p_tiles = []
        p_masks = []
        p_preds = []
        print("Starting Testing")
        self.eval()
        with torch.no_grad():
            for vinputs, vlabels in predict_loader:
                voutputs = self(vinputs)
                voutputs = voutputs.squeeze(1)
                p_tiles.append(vinputs)
                p_masks.append(vlabels)
                p_preds.append(voutputs)
        return p_tiles, p_masks, p_preds

    def pretrain_with_bangalore(self):
        pretrain_path = "./bangalore"
        pretrain_augment = GoeTransform(input_size=self.args.input_size)
        pretrain_data = DatasetSegmentation(pretrain_path)
        # not validating just pre-training
        train_loader = DataLoader(pretrain_data, batch_size=self.args.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        pretrain_loss = []
        pretrain_iou  = []
        start_time = time.time()
        for epoch in range(self.args.pt_num_epochs):
            running_loss = 0.0
            acc = 0.0
            print("Epoch {}/{}".format(epoch + 1, self.args.pt_num_epochs))
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
            pretrain_loss.append(running_loss)
            pretrain_iou.append(acc)
            print(f"[pretrain loss: {running_loss:.4f}] [pretrain iou {acc:.4f}]")
        end_time = time.time()
        tt = end_time - start_time
        print(f"pretraining {self.args.args.pt_num_epochs} took {tt:.2f} seconds.")
        print(f"an average of {tt / self.args.pt_num_epochs:.2f} seconds/epoch.")
        # save the pretrain model
        torch.save(self.state_dict(), self.args.pt_save_path + "pretrain_model.ckpt")
        self.write_list_to_file(self.args.pt_save_path+"pretrain_loss.txt", pretrain_loss)
        self.write_list_to_file(self.args.pt_save_path+"pretrain_iou.txt", pretrain_iou)
        # moving on to regular training
        print(f"pretraining complete: training on normal dataset.")
        trL, trA, vaL, vaA, teA = self.train_on_data()
        self.write_list_to_file(self.args.save_path + "india_train_loss.txt", trL)
        self.write_list_to_file(self.args.save_path + "india_train_acc.txt", trA)
        self.write_list_to_file(self.args.save_path + "india_val_loss.txt", vaL)
        self.write_list_to_file(self.args.save_path + "india_val_acc.txt", vaA)
        self.write_list_to_file(self.args.save_path + "india_test_loss.txt", trA)
        return

    def save(self, save_path,):
        torch.save(self.state_dict(), save_path + "best_model.ckpt")
        return

    @staticmethod
    def write_list_to_file(path, list_to_write):
        with open(path, 'w') as f:
            for line in list_to_write:
                f.write(f"{line}\n")
        return
