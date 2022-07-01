# Imports
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import torch

import time
import argparse
import json

from SNN import *
from helper import *
from splitter import *
from trainer import train_model
from tester import *


# Constants
config = {
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "CROP_SIZE": "small",
    "MAX_EPOCH": 50,
    "SUBSET_SIZE": 1.0,       # Used for testing and not blowing up my laptop
    "SUBSET_VAL_SIZE": 1.0,
    "DATA_FOLDER": "Project/Dresden/natural",
    "RESULT_FOLDER": "Project/Results",
    "MODELS_FOLDER": "Project/Models",
    "SAVED_FILE": "Project/saved.json",
    "HYPER": {
        "LR": 0.00025,
        "DECAY": 0.90,
    }
}
transform = {
    "small":  tf.Compose([
        tf.CenterCrop((100, 100)),
        tf.ToTensor()
    ]),
    "medium": tf.Compose([
        tf.CenterCrop((200, 200)),
        tf.ToTensor()
    ]),
    "large":  tf.Compose([
        tf.CenterCrop((800, 800)),
        tf.ToTensor()
    ])
}


def main(args):
    # Init
    SIZE = getattr(args, "size")

    if getattr(args, 'dev'):  # Local settings
        print("\u001b[31m == RUNNING IN DEV MODE == \u001b[0m")

        print(args)
        config["MAX_EPOCH"] = 15
        config['SUBSET_SIZE'] = 0.02
        config['SUBSET_VAL_SIZE'] = 0.1
        config["DATA_FOLDER"] = "Dresden/natural"
        config["RESULT_FOLDER"] = "Results"
        config["MODELS_FOLDER"] = "Models"
        config["SAVED_FILE"] = "saved.json"
    else:  # Peregrine settings
        print(" == RUNNING IN PEREGRINE MODE == ")

    transformation = transform[SIZE]
    config['CROP_SIZE'] = SIZE

    for k, v in config.items():
        print(f" - {k} : {v}")
    print(f" - TRANSFORMATION :  {transformation}")

    saved_object = json.load(open(config['SAVED_FILE'], "r"))
    save_ID = len(list(filter(
        lambda x: x.startswith(SIZE),
        saved_object.keys())))
    loaded_model = None

    if (suffix := getattr(args, 'load')):  # There is a string after the --load
        load_name = f"{SIZE}_{suffix}"
        if load_name not in saved_object.keys():
            print(f"Could not load the requested model: {load_name}")
            exit(1)
        loaded_model = saved_object[load_name]['Model']

    # Data prep
    train_set, valid_set, test_set = data_splitter(Path(config["DATA_FOLDER"]))
    random.seed(20052022)

    train_set = random.sample(train_set,
                              k=math.floor(len(train_set)*config['SUBSET_SIZE']))
    valid_set = random.sample(valid_set,
                              k=math.floor(len(valid_set)*config['SUBSET_SIZE']))
    test_set = random.sample(test_set,
                             k=math.floor(len(test_set)*config['SUBSET_VAL_SIZE']))

    print(
        f" - Train : {len(train_set):>5} images\n"
        f" - Val   : {len(valid_set):>5} images\n"
        f" - Test  : {len(test_set):>5} images\n")

    SNN_model = SiameseNetwork(SIZE).to(config["DEVICE"])
    print(f"The SNN architecture summary: \n{SNN_model}")
    print(f" {'='*25} ")

    SNN_train_data, SNN_valid_data, SNN_train_data = None, None, None
    if getattr(args, 'mode') in ['create', 'both']:
        # ==== PRE LOADING ====
        start = time.time()
        print("Pre loading training images")
        SNN_train_data = DresdenSNNDataset(
            image_paths=train_set, transform=transformation)
        SNN_valid_data = DresdenSNNDataset(
            image_paths=valid_set, transform=transformation)
        print(f"Loading all images took: {time.time()-start:.4f}s")

    if getattr(args, 'mode') in ['test', 'both']:
        # ==== PRE LOADING ====
        start = time.time()
        print("Pre loading testing images")
        SNN_test_data = DresdenSNNDataset(
            image_paths=test_set, transform=transformation)
        print(f"Loading all images took: {time.time()-start:.4f}s")

    # Training and testing of the model
    if getattr(args, 'mode') in ['create', 'both']:
        train_dataloader = DataLoader(SNN_train_data,
                                      shuffle=True, num_workers=8, batch_size=25)
        valid_dataloader = DataLoader(SNN_valid_data,
                                      shuffle=False, num_workers=8, batch_size=8)

        # ==== ACTUAL TRAINING OF THE MODEL====
        train_model(SNN_model, train_dataloader, valid_dataloader, config)

        if getattr(args, 'save'):
            m = config['MODELS_FOLDER']
            torch.save(SNN_model,
                       f"{m}/model_{SIZE}_{save_ID}.pth")
            print(f"Saving the model at: {m}/model_{SIZE}_{save_ID}.pth")
            saved_object[f"{SIZE}_{save_ID}"] = create_save_object(
                config, save_ID, SNN_model, str(transformation))

            json.dump(saved_object, open(config["SAVED_FILE"], "w"), indent=2)

    if getattr(args, 'mode') in ['test', 'both']:
        test_dataloader = DataLoader(SNN_test_data,
                                     shuffle=False, num_workers=2, batch_size=1)

        # ==== ACTUAL TESTING OF THE MODEL ====
        if getattr(args, 'load'):
            SNN_model = torch.load(
                f"{config['MODELS_FOLDER']}/{loaded_model}",
                map_location=config["DEVICE"])
        test_with_loader(SNN_model, test_dataloader, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SNN model for camera identification')
    parser.add_argument('--dev', action='store_true',
                        help='Runs in development mode')
    parser.add_argument('--size', default='small', const='small',
                        nargs='?', choices=['small', 'medium', 'large'],
                        help='Runs with larger cropped images')
    parser.add_argument('--mode', default='create', const='create',
                        nargs='?', choices=['create', 'test', 'both'],
                        help='Create/test model or do both (default: %(default)s)')

    # TODO: Needs work and probably file name for loading
    parser.add_argument('--save', action='store_true',
                        help='Stores the model in the Models folder and keeps track of the parameters used')
    parser.add_argument('--load', nargs='?', const=None, default=None,
                        help='Loads the model in the Models folder and keeps track of the parameters used')

    main(parser.parse_args())
