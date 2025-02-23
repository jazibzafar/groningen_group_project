import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import albumentations as A
from tifffile import imread
import numpy as np


def img_loader(path):
    return imread(path)

def delete_list(lst):
    del lst[:]
    del lst


def calculate_performance(predictions, targets):
    intersection = predictions*targets
    union = predictions + targets - intersection
    numerator = intersection.sum()
    denominator = union.sum()
    return numerator/denominator


class GoeTransform:
    def __init__(self, input_size: int):
        self.input_size = input_size

        self.transforms = A.Compose([
            A.RandomCrop(height=self.input_size,
                         width=self.input_size,
                         always_apply=True),
            A.HorizontalFlip(p=0.1),  # 0.5
            A.RandomRotate90(p=0.1),  # 0.2
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
                                       contrast_limit=(0.2, 0.3),
                                       p=0.02),  # 0.1
            A.RandomGamma(gamma_limit=(100, 140), p=0.02),  # 0.1
            A.RandomToneCurve(scale=0.1, p=0.02)  # 0.1
        ])

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop

class BanTransform:
    def __init__(self, input_size: int):
        self.input_size = input_size

        self.transforms = A.CenterCrop(height=self.input_size,
                                      width=self.input_size,
                                      always_apply=True)

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop

class BanTransformLabel:
    def __init__(self, input_size: int):
        self.input_size = input_size

        self.transforms = A.CenterCrop(height=self.input_size,
                                      width=self.input_size,
                                      always_apply=True)

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = torch.Tensor((self.transforms(image=image)['image']))
        return crop


class DatasetSegmentation(Dataset):
    def __init__(self, data_path, key=None, transform=None):
        super().__init__()
        self.data_path = data_path
        self.key = key
        self.transform = transform
        self.label_tr = BanTransformLabel(input_size=256)
        self.tile_path = os.path.join(self.data_path, 'tiles/')
        self.mask_path = os.path.join(self.data_path, 'masks/')
        self.tile_list = sorted(os.listdir(self.tile_path), key=len)  # this is a list
        self.mask_list = sorted(os.listdir(self.mask_path), key=len)

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, index):
        tile_name = self.tile_list[index]
        mask_name = self.mask_list[index]
        
        tile_path = os.path.join(self.tile_path, tile_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        
        tile = imread(tile_path)
        # ensure channels == 4
        if self.key == "bangalore":
            if tile.shape[0] > 4:
                tile = tile[0:4, :, :]
                tile = np.transpose(tile, axes=(1,2,0)).astype(np.int32)

        if self.transform:
            tile = self.transform(image=tile)

        mask = imread(mask_path)
        # mask = ToTensor()(mask)  # normalizes the mask which we don't want, instead
        # mask = torch.Tensor(mask)  # this does not normalize
        mask = self.label_tr(mask)
        return tile, mask


class EarlyStopping:
    def __init__(self, tolerance=10, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True