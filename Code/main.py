# Imports
from matplotlib import image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

import time
import datetime
import argparse

from SNN import *
from helper import *
from splitter import *
from trainer import train_model
from tester import *

# Constants
config = {
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "CROP_SIZE": "small",
    "MAX_EPOCH" : 50,         # TODO: Make this into an early termination value
    "SUBSET_SIZE" : 1.0,       # Used for testing and not blowing up my laptop
    "SUBSET_VAL_SIZE": 1.0,
    "DATA_FOLDER" : "Project/Dresden/natural",
    "RESULT_FOLDER" : "Project/Results",
    "MODELS_FOLDER" : "Project/Models",
    "Transform" : {
        "small": transforms.Compose([transforms.CenterCrop((100, 100)),transforms.ToTensor()]),
        "medium" : transforms.Compose([transforms.CenterCrop((200, 200)),transforms.ToTensor()]),
        "large" : transforms.Compose([transforms.CenterCrop((800, 800)),transforms.ToTensor()]),
    }
}


def main(args):
    BEGIN_TIME = time.time()

    # Init 
    if getattr(args, 'dev'): # Local settings
        print("\u001b[31m == RUNNING IN DEV MODE == \u001b[0m")

        print(args)
        config["MAX_EPOCH"] = 15
        config['SUBSET_SIZE'] = 0.002
        config['SUBSET_VAL_SIZE'] = 0.1
        config["DATA_FOLDER"] = "Dresden/natural"
        config["RESULT_FOLDER"] = "Results"
        config["MODELS_FOLDER"] = "Models"
    else: # Peregrine settings
        print(" == RUNNING IN PEREGRINE MODE == ")

    transformation = config["Transform"][getattr(args, 'size')]
    config['CROP_SIZE'] = getattr(args, "size")

    for k, v in config.items():
        print(f" - {k} : {v}")

    train_set, val_set, test_set = data_splitter(Path(config["DATA_FOLDER"]), 0.8, 0.9)
    random.seed(20052022)  
    
    train_set = random.sample(train_set, k=math.floor(len(train_set)*config['SUBSET_SIZE']))
    val_set = random.sample(val_set,     k=math.floor(len(val_set)*config['SUBSET_SIZE']))
    test_set = random.sample(test_set,   k=math.floor(len(test_set)*config['SUBSET_VAL_SIZE'])) 

    print(f" - Train : {len(train_set):>5} images\n - Val   : {len(val_set):>5} images\n - Test  : {len(test_set):>5} images")

   
    SNN_model = SiameseNetwork(config['CROP_SIZE']).to(config["DEVICE"])
    print(f"The SNN architecture summary: \n{SNN_model}")
    print(f" {'='*25} ")

    SNN_train_data = None
    SNN_val_data   = None
    SNN_test_data  = None
    train_dataloader = None
    val_dataloader   = None
    test_dataloader  = None

    # ==== PRE LOADING ====
    if getattr(args, 'mode') in ['create', 'both']:
        start = time.time()
        print("Pre loading training images")
        SNN_train_data = DresdenSNNDataset(image_paths=train_set, transform=transformation)

        SNN_val_data   = DresdenSNNDataset(image_paths=val_set, transform=transformation)

        train_dataloader = DataLoader(SNN_train_data,
            shuffle=True, num_workers=8, batch_size=32)
        val_dataloader   = DataLoader(SNN_val_data, 
            shuffle=False, num_workers=8, batch_size=8)
        print(f"Loading all images took: {time.time()-start:.4f}s")

   
    if getattr(args, 'mode') in ['test', 'both']:
        start = time.time()
        print("Pre loading testing images")
        SNN_test_data = DresdenSNNDataset(image_paths=test_set, transform=transformation)
        test_dataloader = DataLoader(SNN_test_data,
            shuffle=False, num_workers=2, batch_size=1)
        print(f"Loading all images took: {time.time()-start:.4f}s")
    # =====================



    if getattr(args, 'mode') in ['create', 'both']:
        train_model(SNN_model, train_dataloader, val_dataloader, config)
        #TODO: Make a json file to keep track of the models 
        if getattr(args, 'save'):
            torch.save(SNN_model, f"{config['MODELS_FOLDER']}/model_{getattr(args, 'size')}.pth")

        del train_dataloader
    
    

    if getattr(args, 'mode') in ['test', 'both']:
        if getattr(args, 'load'):
            SNN_model = torch.load(f"{config['MODELS_FOLDER']}/model_{getattr(args, 'size')}.pth", map_location=config["DEVICE"])
        # test_model(SNN_model, test_dataloader, SNN_test_data, config)
        test_with_loader(SNN_model, test_dataloader, config)
   
    print(f"Run sucessful and took: {str(datetime.timedelta(seconds=time.time()-BEGIN_TIME))[:-3]}")
    
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
    parser.add_argument('--load', action='store_true',
        help='Loads the model in the Models folder and keeps track of the parameters used')
       
    main(parser.parse_args())