##
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from src.model import YoloModel
plt.rcParams['figure.figsize'] = (5,5)
import pickle

##
def to_rgb(img_in):
    img_in = img_in[0]
    img_out =img_in[0:3, :, :]
    img_out = img_out.permute(1, 2, 0)
    return img_out.numpy()


def show_img(img, title=''):
    plt.imshow(img)
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


##
# testing if the tiles are the same
path_direct_tiles = "./pickled/direct_tiles.pkl"
path_pretrain_tiles = "./pickled/pretrain_tiles.pkl"

with open(path_direct_tiles, "rb") as f:
    direct_tiles = pickle.load(f)

with open(path_pretrain_tiles, "rb") as f:
    pretrain_tiles = pickle.load(f)


# ##
# tile1 = to_rgb(direct_tiles[0])
# show_img(tile1, "")
#
# tile2 = to_rgb(pretrain_tiles[2])
# show_img(tile2, "")
#
# ##
# path_direct_masks = "./pickled/direct_masks.pkl"
# with open(path_direct_masks, "rb") as f:
#     direct_masks = pickle.load(f)
# mask1 = direct_masks[0]
# show_img(mask1[0])
#
# mask2 = direct_masks[2]
# show_img(mask2[0])
#
##
path_direct_preds = "./pickled/direct_preds.pkl"
with open(path_direct_preds, "rb") as f:
    direct_preds = pickle.load(f)

pred1 = direct_preds[0]
show_img(pred1[0])

pred2 = direct_preds[2]
show_img(pred2[0])

##
path_pretrain_preds = "./pickled/pretrain_preds.pkl"
with open(path_pretrain_preds, "rb") as f:
    pretrain_preds = pickle.load(f)

preda = pretrain_preds[0]
show_img(preda[0])

predb = pretrain_preds[2]
show_img(predb[0])



# pred1 = test_preds[0]
# show_img(pred1[0], "")
##
#
# im1 = np.arange(100).reshape((10, 10))
# im2 = im1.T
# im3 = np.flipud(im1)
# im4 = np.fliplr(im2)
#
# fig = plt.figure(figsize=(4,2))
# plt.suptitle("Original Image", fontsize=12)
# plt.axis('off')
#
# ax1 = fig.add_subplot(1, 3, 1)
# ax1.imshow(im1)
# ax1.set_title("im1")
# ax1.axis('off')
#
# ax2 = fig.add_subplot(1, 3, 2)
# ax2.imshow(im2)
# ax2.set_title("im2")
# ax2.axis('off')
#
# ax3 = fig.add_subplot(1, 3, 3)
# ax3.imshow(im3)
# ax3.set_title("im3")
# ax3.axis('off')
#
# fig.tight_layout()
# plt.show()
# ##
#
# def viz_imgs(tile, mask, pred):
#     # prepare data
#     tile = to_rgb(tile)
#
#     plt.figure(figsize=(5,3))
#     ax1 = fig.add_subplot(1, 3, 1)
#     ax1.imshow(tile)
#     ax1.set_title("Tile")
#     ax1.axis('off')
#
#     ax2 = fig.add_subplot(1, 3, 2)
#     ax2.imshow(mask)
#     ax2.set_title("Mask")
#     ax2.axis('off')
#
#     ax3 = fig.add_subplot(1, 3, 3)
#     ax3.imshow(pred)
#     ax3.set_title("Pred")
#     ax3.axis('off')
#
#     fig.tight_layout()
#     plt.show()
#     return
