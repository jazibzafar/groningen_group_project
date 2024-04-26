##
from src.utils import img_loader
import os
from tifffile import imwrite
from pathlib import Path
import numpy as np


def slice_image(sample, slicesize: int):
    output = []
    for x in range(0, sample.shape[0], slicesize):
        for y in range(0, sample.shape[1], slicesize):
            output.append(sample[x:x + slicesize, y:y + slicesize, :])  # HWC
    return output


def slice_mask(sample, slicesize: int):
    output = []
    for x in range(0, sample.shape[0], slicesize):
        for y in range(0, sample.shape[1], slicesize):
            output.append(sample[x:x + slicesize, y:y + slicesize])  # HWC
    return output


def data_slicer(path: str = "./data/goettingen/", slice_size: int = 256):
    # join the paths and stuff
    path_tiles = os.path.join(path, 'tiles')
    path_masks = os.path.join(path, 'masks')
    path_sliced = path + "sliced"
    path_tiles_sliced = os.path.join(path_sliced, 'tiles')
    path_masks_sliced = os.path.join(path_sliced, 'masks')

    if not (Path(path_sliced).exists()):
        os.mkdir(path_sliced)
    if not(Path(path_tiles_sliced).exists()):
        os.mkdir(path_tiles_sliced)
    if not(Path(path_masks_sliced).exists()):
        os.mkdir(path_masks_sliced)
    # generate a list of indices
    tile_numbers = np.arange(38) + 1
    # generate the list of tiles and masks
    tile_list = [os.path.join(path_tiles, f"tile_{i}.tif") for i in tile_numbers]
    mask_list = [os.path.join(path_masks, f"mask_{i}.tif") for i in tile_numbers]
    # slice the tiles
    sliced_tile_list = []
    sliced_mask_list = []
    for tile in range(len(tile_list)):
        img = img_loader(tile_list[tile])
        msk = img_loader(mask_list[tile])
        sliced_tile_list.extend(slice_image(img, slice_size))
        sliced_mask_list.extend(slice_mask(msk, slice_size))

    # save the tiles to disk
    for i in range(len(sliced_tile_list)):
        imwrite(os.path.join(path_tiles_sliced, f"tile_{i}.tif"), sliced_tile_list[i])
        # since tile list and mask list have the same length:
        imwrite(os.path.join(path_masks_sliced, f"mask_{i}.tif"), sliced_mask_list[i])
    return sliced_tile_list, sliced_mask_list
