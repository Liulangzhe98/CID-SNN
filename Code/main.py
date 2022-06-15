# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

import time
import argparse

from SNN import *
from helper import *
from splitter import *
from trainer import train_model
from validator import validate_model, validate_model_with_loader

# Constants
config = {
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "MAX_EPOCH" : 20,         # TODO: Make this into an early termination value
    "SUBSET_SIZE" : 1.0,       # Used for testing and not blowing up my laptop
    "SUBSET_VAL_SIZE": 1.0,
    "DATA_FOLDER" : "Project/Dresden/natural",
    "RESULT_FOLDER" : "Project/Results",
    "MODELS_FOLDER" : "Project/Models",
    "Transform" : {
        "small": transforms.Compose([transforms.CenterCrop((100, 100)),transforms.ToTensor()]),
        "large" : transforms.Compose([transforms.CenterCrop((200, 200)),transforms.ToTensor()]),
    }
}


def main(args):
    # Init 
    if getattr(args, 'dev'): # Local settings
        print("\u001b[31m == RUNNING IN DEV MODE == \u001b[0m")

        print(args)
        config["MAX_EPOCH"] = 10
        config['SUBSET_SIZE'] = 0.05
        config['SUBSET_VAL_SIZE'] = 0.1
        config["DATA_FOLDER"] = "Dresden/natural"
        config["RESULT_FOLDER"] = "Results"
        config["MODELS_FOLDER"] = "Models"
    else: # Peregrine settings
        print(" == RUNNING IN PEREGRINE MODE == ")

    transformation = config["Transform"]['large' if getattr(args, 'large') else 'small']

    print(f" - Working with device: {config['DEVICE']}\n - Within the folder: {config['DATA_FOLDER']}")
    for k, v in config.items():
        print(f" - {k} : {v}")

    train_set, test_set = data_splitter(Path(config["DATA_FOLDER"]), 0.8)
    random.seed(20052022)  
    
    train_set = random.sample(train_set, k=math.floor(len(train_set)*config['SUBSET_SIZE']))
    # train_set = random.sample(train_set, k=10)
    
    test_set = random.sample(test_set,   k=math.floor(len(test_set)*config['SUBSET_VAL_SIZE'])) 

    print(f" - Train : {len(train_set)} pairs\n - Test  : {len(test_set)} pairs")

    # Load the training dataset
    # TODO: Set the right workers and batch size

    SNN_model = (SiameseNetworkLarger if getattr(args, 'large') else SiameseNetwork)().to(config["DEVICE"])
    print(f"The SNN architecture summary: \n{SNN_model}")
    print(f" {'='*25} ")

    # TODO: Mode selection for better time usage on peregrine
    if getattr(args, 'mode') in ['create', 'both']:
        start = time.time()
        print("Pre loading train images")
        train_dataloader = DataLoader(
            DresdenSNNDataset(image_paths=train_set, transform=transformation),
            shuffle=False, num_workers=8, batch_size=64)
        print(f"Loading all images took: {time.time()-start:.4f}s")
        train_model(SNN_model, train_dataloader, config)
  
    #TODO: Make a json file to keep track of the models 
    if getattr(args, 'save'):
        torch.save(SNN_model, f"{config['MODELS_FOLDER']}/model_{'large' if getattr(args, 'large') else 'small'}.pth")

    if getattr(args, 'load'):
        SNN_model = torch.load(f"{config['MODELS_FOLDER']}/model_{'large' if getattr(args, 'large') else 'small'}.pth", map_location=config["DEVICE"])
        
    if getattr(args, 'mode') in ['validate', 'both']:
        start = time.time()
        print("Pre loading test images")
        SNN_data = DresdenSNNDataset(image_paths=test_set, transform=transformation)
        test_dataloader = DataLoader(
            SNN_data,
            shuffle=False, num_workers=2, batch_size=1)
        print(f"Loading all images took: {time.time()-start:.4f}s")
        # validate_model(SNN_model, test_dataloader, SNN_data, config)

        validate_model_with_loader(SNN_model, test_dataloader, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SNN model for camera identification')
    parser.add_argument('--dev', action='store_true',
        help='Runs in development mode')
    parser.add_argument('--large', action='store_true', 
        help='Runs with larger cropped images')
    parser.add_argument('--mode', default='create', const='create',
        nargs='?', choices=['create', 'validate', 'both'],
        help='Create/validate model or do both (default: %(default)s)')

    # TODO: Needs work and probably file name for loading    
    parser.add_argument('--save', action='store_true',
        help='Stores the model in the Models folder and keeps track of the parameters used')
    parser.add_argument('--load', action='store_true',
        help='Loads the model in the Models folder and keeps track of the parameters used')
       
    main(parser.parse_args())