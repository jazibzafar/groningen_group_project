import matplotlib.pyplot as plt

from src.model import YoloModel
from src.slicer import data_slicer
from dataclasses import dataclass
from visualize import to_rgb


@dataclass
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


def test_main():
    # args = get_arguments()
    args = Args()
    # data_slicer()
    model = YoloModel(args = args)
    # model.train_on_data()
    test_tiles, test_masks, test_preds = model.test()
    return test_tiles, test_masks, test_preds


if __name__ == '__main__':
    test_tiles, test_masks, test_preds = test_main()