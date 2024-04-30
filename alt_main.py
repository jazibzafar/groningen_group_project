import matplotlib.pyplot as plt

from src.model import YoloModel
from src.slicer import data_slicer
from dataclasses import dataclass


@dataclass
class Args:
    model_name: str = "coco"
    in_channels: int = 4
    input_size: int = 256
    data_path: str = "./data/goettingen/sliced/"
    save_path: str = "./output/"
    ckpt_path: str = "./output_30apr/best_model.ckpt"
    pred_path: str = "./data/goettingen/predict"
    batch_size: int = 32
    num_epochs: int = 1
    optimizer_class: str = "adam"
    loss: str = "l2"
    pt_num_epochs: int = 1
    pt_save_path: str = "./pt_output/"


def test_main():
    # args = get_arguments()
    args = Args()
    # data_slicer()
    model = YoloModel(args = args)
    # Linear Eval on CoCo
    # model.train_only_output_layer()  # lin eval; losses/accuracies saved
    # leval_tiles, leval_masks, leval_preds = model.predict()
    # model.reload_model()
    # # Train directly on data
    tL, tA, vL, vA, test_acc = model.train_on_data()
    model.write_list_to_file(model.args.save_path + "direct_tL.txt", tL)
    model.write_list_to_file(model.args.save_path + "direct_tA.txt", tA)
    model.write_list_to_file(model.args.save_path + "direct_vL.txt", vL)
    model.write_list_to_file(model.args.save_path + "direct_vA.txt", vA)
    model.write_list_to_file(model.args.save_path + "direct_test_acc.txt", test_acc)
    direct_tiles, direct_masks, direct_preds = model.predict()
    model.reload_model()
    pretrain_tiles, pretrain_masks, pretrain_preds = model.predict()

    return


if __name__ == '__main__':
    test_main()