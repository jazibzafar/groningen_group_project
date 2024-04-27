import os

import argparse
from numpy import arange

from src.model import YoloModel
from src.data_augmentation import augment_dataset
from src.utils import DatasetSegmentation, GoeTransform
from src.arguments import get_arguments


class Args:
    def __init__(self):
        self.model = "coco"
        self.in_channels=4
        self.data_path = "./data/oettingen/sliced/"
        self.batch_size=32
        self.num_epochs=1

def test_main():
    # args = get_arguments()
    args = Args()
    model = YoloModel(model_name=args.model, in_channels=args.in_channels)
    model.train(path=args.data_path,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                optimizer_class='adam',
                loss='l2')


if __name__ == '__main__':
    test_main()
