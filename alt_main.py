import matplotlib.pyplot as plt
from src.model import YoloModel
from src.slicer import data_slicer
from dataclasses import dataclass
plt.rcParams['figure.figsize'] = (5,5)
import pickle

@dataclass
class Args:
    model_name: str = "coco"
    in_channels: int = 4
    input_size: int = 256
    data_path: str = "./data/goettingen/sliced/"
    save_path: str = "./output_final/"
    ckpt_path: str = "./output_30apr/best_model.ckpt"
    pred_path: str = "./data/goettingen/predict"
    batch_size: int = 64
    num_epochs: int = 80
    optimizer_class: str = "adam"
    loss: str = "l2"
    pt_num_epochs: int = 40
    pt_save_path: str = "./output_final/"


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


def test_main():
    # args = get_arguments()
    args = Args()
    # data_slicer()
    model = YoloModel(args = args)

    # Train directly on data
    tL, tA, vL, vA, test_acc = model.train_on_data()
    model.write_list_to_file(model.args.save_path + "direct_tL.txt", tL)
    model.write_list_to_file(model.args.save_path + "direct_tA.txt", tA)
    model.write_list_to_file(model.args.save_path + "direct_vL.txt", vL)
    model.write_list_to_file(model.args.save_path + "direct_vA.txt", vA)
    model.write_list_to_file(model.args.save_path + "direct_test_acc.txt", test_acc)
    direct_tiles, direct_masks, direct_preds = model.predict()
    model.reload_model()
    model.pretrain_with_bangalore()
    pretrain_tiles, pretrain_masks, pretrain_preds = model.predict()

    with open("./pickled/direct_tiles.pkl", "wb") as f:
        pickle.dump(direct_tiles, f)

    with open("./pickled/direct_masks.pkl", "wb") as f:
        pickle.dump(direct_masks, f)

    with open("./pickled/direct_preds.pkl", "wb") as f:
        pickle.dump(direct_preds, f)

    with open("./pickled/pretrain_tiles.pkl", "wb") as f:
        pickle.dump(pretrain_tiles, f)

    with open("./pickled/pretrain_masks.pkl", "wb") as f:
        pickle.dump(pretrain_masks, f)

    with open("./pickled/pretrain_preds.pkl", "wb") as f:
        pickle.dump(direct_preds, f)
    return


if __name__ == '__main__':
    test_main()