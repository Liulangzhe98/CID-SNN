# Imports
import torch
from torch.utils.data import DataLoader as DL

from argparse import ArgumentParser, Namespace
from SNN import *
from configure import Configure
from helper import *
from splitter import *
from trainer import train_model, preload_training
from tester import *


def create_arg_parser() -> Namespace:
    """Function that takes care of the argument parsing 

    Returns:
        Namespace: Namespace of all option
    """
    parser = ArgumentParser(description="Train or load a siamese neural network for camera identification")
    parser.add_argument("--dev", action="store_true",
                        help="Runs in development mode")
    parser.add_argument("--size", default="small", const="small",  required=True,
                        nargs="?", choices=SiameseNetwork.layers,
                        help="Runs with different transformations on the images and different layers in the CNN")
    parser.add_argument("--mode", default="create", const="create",  required=True,
                        nargs="?", choices=["create", "test", "both"],
                        help="Create/test model or do both (default: %(default)s)")

    parser.add_argument("--save", action="store_true",
                        help="Stores the model in the Models folder and keeps track of the parameters used")
    parser.add_argument("--load", nargs="?", const=None, default=None,
                        help="Loads the model from the Models folder")
    parser.add_argument("--ID", nargs=1, required=True,
                        help="Used for storing the log ID into the save file")

    return parser.parse_args()


def main(args: Namespace) -> None:
    """Function that controls the flow of the program

    Args:
        args (Namespace): All preselected option to run the program
    """

    # Init
    # TODO: Make a nice printout for the config class
    config = Configure(args)
    CS = config.CROP_SIZE

    for k, v in vars(config).items():
        v = str(v).replace("\n", "")
        print(f" - {k:20}| {v}")

    # Data prep
    train_set, valid_set, test_set = data_splitter(Path(config.DATA_FOLDER))
    random.seed(20052022)

    train_set = random.sample(train_set,
                              k=math.floor(len(train_set)*config.SUBSET_SIZE))
    valid_set = random.sample(valid_set,
                              k=math.floor(len(valid_set)*config.SUBSET_SIZE))
    test_set = random.sample(test_set,
                             k=math.floor(len(test_set)*config.SUBSET_VAL_SIZE))

    print(
        f" - Train : {len(train_set):>5} images\n"
        f" - Val   : {len(valid_set):>5} images\n"
        f" - Test  : {len(test_set):>5} images")
    print(f" {'-'*25} ")

    SNN_model = SiameseNetwork(CS).to(config.DEVICE)
    print(f"The SNN architecture summary: \n{SNN_model}")
    print(f" {'='*25} \n")

    # Preloading images to RAM for training/validation
    if getattr(args, "mode") in ["create", "both"]:
        train_data, valid_data = preload_training(
            config.Transform, train_set, valid_set)

    # Preloading images to RAM for testing
    if getattr(args, "mode") in ["test", "both"]:
        test_data = preload_testing(config.Transform, test_set)

    # Training and testing of the model
    if getattr(args, "mode") in ["create", "both"]:
        train_dl = DL(train_data, shuffle=True, num_workers=8, batch_size=15)
        valid_dl = DL(valid_data, shuffle=False, num_workers=8, batch_size=8)

        train_model(SNN_model, train_dl, valid_dl, config)

        # Saving the model when specified
        if getattr(args, "save"):
            save_model(config, SNN_model)

    if getattr(args, "mode") in ["test", "both"]:
        test_dl = DL(test_data, shuffle=False, num_workers=2, batch_size=1)

        # Loading a pretrained model when specified
        if getattr(args, "load"):
            SNN_model = torch.load(config.loaded_model,
                                   map_location=config.DEVICE)
        test_model(SNN_model, test_dl, config)

if __name__ == "__main__":
    main(create_arg_parser())
