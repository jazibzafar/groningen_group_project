##
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from src.model import YoloModel
plt.rcParams['figure.figsize'] = (5,5)

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
class Args:
    model_name: str = "coco"
    in_channels: int = 4
    input_size: int = 256
    data_path: str = "./data/goettingen/sliced/"
    save_path: str = "./output/"
    ckpt_path: str = "./output/best_model.ckpt"
    pred_path: str = "./data/goettingen/predict"
    batch_size: int = 32
    num_epochs: int = 100
    optimizer_class: str = "adam"
    loss: str = "l2"

args = Args()
model = YoloModel(args = args)
test_tiles, test_masks, test_preds = model.test()

##
tile1 = to_rgb(test_tiles[0])
show_img(tile1, "")
mask1 = test_masks[0]
show_img(mask1[0])
pred1 = test_preds[0]
show_img(pred1[0], "")
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
