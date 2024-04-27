import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="coco",
                        help='What model to use (empty, coco, retrained_empty, retrained_coco)')
    parser.add_argument('--data_dir', type=str, default="./data/goettingen/sliced/",
                        help='path to the dataset directory')
    parser.add_argument('--input-channels', type=int, default=4,
                        help='How many color channels does the dataset have.')
    parser.add_argument('--input-size', type=int, default=256,
                        help='the size of the input image')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='No. epochs')
    parser.add_argument('--out_dir', type=str, default="./out/",
                        help='directory to save checkpoints and results.')
