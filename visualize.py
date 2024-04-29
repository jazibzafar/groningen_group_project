##
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread
from src.utils import DatasetSegmentation
from src.model import YoloModel
from dataclasses import dataclass


##
# helper functions
def img_loader(path):
    return imread(path)


def to_rgb(img_in):
    if img_in.shape == 0:
        pass



# indices = ['29', '88', '358', '297', '417', '518', '157', '589']
# path_tiles = "./data/goettingen/predict/tiles"
# path_masks = "./data/goettingen/predict/masks"
# tile_list = sorted(os.listdir(path_tiles), key=len)
# mask_list = sorted(os.listdir(path_masks), key=len)
##
predict_path = "./data/goettingen/predict"
data = DatasetSegmentation(predict_path)
dataloader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)

@dataclass
class Args:
    model_name: str = "coco"
    in_channels: int = 4
    input_size: int = 256
    data_path: str = "./data/goettingen/sliced/"
    save_path: str = "./output/"
    ckpt_path: str = "./output/best_model.ckpt"
    batch_size: int = 8
    num_epochs: int = 10
    optimizer_class: str = "adam"
    loss: str = "l2"


args = Args()
model = YoloModel(args = args)
model.load_state_dict(torch.load(args.ckpt_path))

##
model.eval()
with torch.no_grad():
    for tiles, masks in dataloader:
        inp = tiles.permute(0, 3, 1, 2)
        pred = model(inp)

