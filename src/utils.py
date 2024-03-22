import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from tifffile import imread

def img_loader(path):
    return imread(path)

def delete_list(lst):
    del lst[:]
    del lst


def jaccard_index(predictions, targets):
    intersection = predictions*targets
    union = predictions + targets - intersection
    numerator = intersection.sum()
    denominator = union.sum()
    return numerator/denominator


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
        mask = ToTensor()(mask)
        return tile, mask