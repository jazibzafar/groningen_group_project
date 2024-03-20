from torch.utils.data import Dataset, DataLoader
from tifffile import imread
import os


def img_loader(path):
    return imread(path)

class DatasetSegmentation(Dataset):
    def _init_(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.tile_path = os.path.join(self.data_path, 'tiles')
        self.mask_path = os.path.join(self.data_path, 'masks')
        self.tile_list = os.listdir(self.tile_path)  # this is a list
        self.mask_list = os.listdir(self.mask_path)

    def _len_(self):
        return len(self.tile_list)

    def _getitem_(self, index):
        tile_name = self.tile_list[index]
        mask_name = self.mask_list[index]
        
        tile_path = os.path.join(self.tile_path, tile_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        
        tile = imread(tile_path)
        mask = imread(mask_path)
        return tile, mask
    
path = "./treecover_segmentation_aerial_goettingen"
data = DatasetSegmentation(path)