from src.utils import img_loader, delete_list

import os
from pathlib import Path
from tifffile import imwrite
from tqdm import tqdm
import numpy as np


def augment_dataset(path: str = "./data/goettingen"):
    all_tiles_path = os.path.join(path, 'tiles')
    all_masks_path = os.path.join(path, 'masks')
    path_augmented = path + "_augmented"
    all_tiles_path_augmented = os.path.join(path_augmented, 'tiles')
    all_masks_path_augmented = os.path.join(path_augmented, 'masks')

    if not(Path(path_augmented).exists()):
        os.mkdir(path_augmented)
    if not(Path(all_tiles_path_augmented).exists()):
        os.mkdir(all_tiles_path_augmented)
    if not(Path(all_masks_path_augmented).exists()):
        os.mkdir(all_masks_path_augmented)
    #TODO: change limit for doing augmentation
    if len(os.listdir(all_tiles_path_augmented)) < 2000:
        print("Starting Augmentation")
        for img_no in tqdm(range(1, len(os.listdir(all_tiles_path)) + 1)):
            tile_list = [img_loader(os.path.join(all_tiles_path,"tile_" + str(img_no) + ".tif"))]
            mask_list = [img_loader(os.path.join(all_masks_path,"mask_" + str(img_no) + ".tif"))]
            tile_list.append(np.flipud(tile_list[0]))
            tile_list.append(np.fliplr(tile_list[0]))
            mask_list.append(np.flipud(mask_list[0]))
            mask_list.append(np.fliplr(mask_list[0]))

            temp = []
            for tile in tile_list:
                temp.append(tile[0:512, 0:512,:])
                temp.append(tile[0:512, 512:1024,:])
                temp.append(tile[512:1024, 512:1024,:])
                temp.append(tile[512:1024, 0:512,:])
                temp.append(tile[256:768, 256:768,:])
            tile_list[:] = [*tile_list, *temp]
            del temp[:]

            for mask in mask_list:
                temp.append(mask[0:512, 0:512])
                temp.append(mask[0:512, 512:1024])
                temp.append(mask[512:1024, 512:1024])
                temp.append(mask[512:1024, 0:512])
                temp.append(mask[256:768, 256:768])

            mask_list[:] = [*mask_list, *temp]
            del temp[:]

            for tile in tile_list:
                temp.append(np.rot90(tile, k=1, axes=(0,1)))
                temp.append(np.rot90(tile, k=2, axes=(0,1)))
                temp.append(np.rot90(tile, k=3, axes=(0,1)))
            
            tile_list[:] = [*tile_list, *temp]
            del temp[:]

            for mask in mask_list:
                temp.append(np.rot90(mask, k=1, axes=(0,1)))
                temp.append(np.rot90(mask, k=2, axes=(0,1)))
                temp.append(np.rot90(mask, k=3, axes=(0,1)))
            
            mask_list[:] = [*mask_list, *temp]
            delete_list(temp)

            for index in range(0,len(tile_list)):
                resized = tile_list[index].copy()
                if resized.shape[0] == 1024:
                    resized.resize((512,512,4))
                if resized.shape[2] == 1:
                    print("SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA") 
                if index == 0:
                    imwrite(os.path.join(all_tiles_path_augmented, "tile_" + str(img_no) + ".tif"), resized)
                else:
                    imwrite(os.path.join(all_tiles_path_augmented, "tile_" + str(img_no) + "_" + str(index) + ".tif"), resized)
                del resized
            
            delete_list(tile_list)

            for index in range(0,len(mask_list)):
                resized = mask_list[index].copy()
                if resized.shape[0] == 1024:
                    resized.resize((512,512)) 
                if index == 0:
                    imwrite(os.path.join(all_masks_path_augmented, "mask_" + str(img_no) + ".tif"), resized)
                else:
                    imwrite(os.path.join(all_masks_path_augmented, "mask_" + str(img_no) + "_" + str(index) + ".tif"), resized)
                del resized
            
            delete_list(mask_list)

            del mask
            del tile
    else:
        pass
    print("Augmentation Finished")