import os

import argparse
from numpy import arange

from src.model import yolo_model
from src.data_augmentation import augment_dataset

# function that determines the function call parameters and returns them
# NOTE: The function below specifies the inputs to the model. For instance, if you want to run it from the terminal,
# you would run it like this:
# python3.exe main.py --model "coco" --dataset "goettingen" ...
def get_arguments():
    parser = argparse.ArgumentParser(description='Deep Learning Project')

    # arguments allowed in function call
    parser.add_argument('--model', type=str, default="coco", help='What model to use (empty, coco, retrained_empty, retrained_coco)')

    parser.add_argument('--dataset', type=str, default="goettingen", help='What dataset to be used  for training (goettingen or india)')

    parser.add_argument('--on-augmented', type=bool, default=True, help='Whether to run training on augmented data')

    parser.add_argument('--input-channels', type=int, default=4, help='How many color channels do ')

    parser.add_argument('--input-size', type=int, default=256, help='the size of the input image')

    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')

    parser.add_argument('--epochs', type=int, default=100, help='No. epochs')

    parser.add_argument('--optimizer', type=str, default="adam", help='What optimizer to use (adam or sgd)')

    parser.add_argument('--loss', type=str, default="l2", help='What loss to use (l1 or l2)')
    
    arguments = parser.parse_args()

    # raising errors for wrong args input
    if arguments.optimizer not in ["adam", "sgd"]:
        raise parser.error("The optimizer should be either 'adam' or 'sgd' for --optimizer")
    
    if arguments.dataset not in ["goettingen", "india"]:
        raise parser.error("The dataset should be either 'goettingen' or 'india' for --optimizer")
    
    if arguments.model not in ["empty", "coco", "retrained_empty", "retrained_coco"]:
        raise parser.error("The model should be either 'empty', 'coco', 'retrained_empty' or 'retrained_coco' for --model")
    
    if arguments.loss not in ["l1", "l2"]:
        raise parser.error("The loss should be either 'l1' or 'l2' for --loss")
    
    if arguments.dataset == "india":
        print("Warning: This dataset will not be augmented!!!")

    # returning each function call parameter
    return arguments

# NOTE: This is the function that is called to run the training and validation of the model.
def run(model_name: str = "coco",
        dataset_name: str = "goettingen",
        train_on_augmented: bool = True,
        in_channels: int = 4,
        batch_size: int = 64,
        num_epochs: int = 100,
        optimizer: str = "adam",
        loss: str = "l2"
        ):

    # NOTE: Here we declare what model we are using
    model = yolo_model(model_name= model_name, in_channels=in_channels)

    # NOTE: The dataset is declared here.
    if dataset_name == "goettingen":
        if train_on_augmented:
            data_path = os.path.join(os.getcwd(), "data", "goettingen_augmented")
        else:
            data_path = os.path.join(os.getcwd(), "data", "goettingen")
    elif dataset_name == "india":
        data_path = os.path.join(os.getcwd(), "data", "india")
    else:
        data_path = os.path.join(os.getcwd(), "data", "goettingen")

    # NOTE: This is the training function. When model.train() is called, the training begins.
    model.train(path=data_path, 
                batch_size=batch_size, 
                num_epochs=num_epochs,
                optimizer_class=optimizer,
                loss=loss)


# NOTE: This part of the code tells the python interpreter what to do.
if __name__ == '__main__':
    # 1. You get the arguments for the training.
    arguments = get_arguments()
    augment_dataset()

    # 2. You run the training using the arguments called above.
    # To not perform a part of the task mention it in run command line. For example: -no-training
    run(
        model_name=arguments.model,
        dataset_name=arguments.dataset,
        train_on_augmented=arguments.on_augmented,
        in_channels=arguments.input_channels,
        batch_size=arguments.batch_size,
        num_epochs=arguments.epochs,
        optimizer=arguments.optimizer,
        loss=arguments.loss
        )