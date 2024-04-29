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
    ckpt_path: str = "./output/best_model.ckpt"
    batch_size: int = 64
    num_epochs: int = 10
    optimizer_class: str = "adam"
    loss: str = "l2"


def test_main():
    # args = get_arguments()
    args = Args()
    data_slicer()
    model = YoloModel(args = args)
    model.train_on_data()
    model.test()


if __name__ == '__main__':
    test_main()
