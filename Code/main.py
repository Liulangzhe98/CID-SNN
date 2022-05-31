# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch import device, optim
import torch.nn.functional as F

from os import path
import time
import argparse
import json

from SNN import *
from helper import *
from splitter import *

# Constants
config = {
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "MAX_EPOCH" : 20,         # TODO: Make this into an early termination value
    "SUBSET_SIZE" : 30,       # Used for testing and not blowing up my laptop
    "DATA_FOLDER" : "Project/Dresden/natural",
    "RESULT_FOLDER" : "Project/Results",
    "MODELS_FOLDER" : "Project/Models",
}


# Resize the images and transform to tensors
# TODO: The resize might remove pattern noise ???
# TODO: (100, 100) resize makes it so that the layers are applied correctly
transformation = transforms.Compose([
        # transforms.Resize((640,480)),
        transforms.CenterCrop((100, 100)),
        transforms.ToTensor()
    ])


def main(args):
    start_time = time.time()
  
    print(f" - Working with device: {config['DEVICE']}\n - Within the folder: {config['DATA_FOLDER']}")
    for k, v in config.items():
        print(f" - {k} : {v}")


    train_set, test_set = data_splitter(Path(config["DATA_FOLDER"]), 0.8)
    if getattr(args, 'dev'):
        train_set = random.sample(train_set, k=config["SUBSET_SIZE"])
        test_set = random.sample(test_set, k=len(test_set)//100)
    else:
        # peregrine settings
        train_set = random.sample(train_set, k=len(train_set)//2)
        test_set = random.sample(test_set, k=len(test_set)) 

    # Load the training dataset
    # TODO: Make sure these values are working for our dataset
    train_dataloader = DataLoader(
        DresdenSNNDataset(image_paths=train_set, transform=transformation),
        shuffle=True, num_workers=8, batch_size=64)

    SNN_model = SiameseNetwork().to(config["DEVICE"])
    print(f"The SNN architecture summary: \n{SNN_model}")
    print(f" {'='*25} ")

    train_model(SNN_model, train_dataloader, start_time)
    # torch.save(SNN_model, f"{config['MODELS_FOLDER']}/model.pth")

    # SNN_model = torch.load(f"{config['MODELS_FOLDER']}/model.pth", map_location=config["DEVICE"])

    # # TODO: Make sure these values are working for our dataset
    test_dataloader = DataLoader(
        DresdenSNNDataset(image_paths=test_set, transform=transformation), 
        shuffle=True, num_workers=2, batch_size=1)

    validate_model(SNN_model, test_dataloader)


def validate_model(net: SiameseNetwork, dataloader: DataLoader):
    print(" === Validation results === ")
    device = config["DEVICE"]
    dataiter = iter(dataloader)

    x0, _, _, f_name0, _ = next(dataiter)
    name = f_name0[0].split('/')[0]

    for i in range(3):
        models_checked = {}
        for _ in range(3): # TODO: Maybe not hardcode -> now it is 9 cameras + avg
            same_histo = []
            diff_histo = []

            while name in models_checked.keys():
                x0, _, _, f_name0, _ = next(dataiter)
                name = f_name0[0].split('/')[0]
            print(f"  Validating on : {name}")
            
            for _, x1, _, _, f_name1 in dataiter:
                name_other = f_name1[0].split('/')[0]
                output1, output2 = net(x0.to(device), x1.to(device))
                euclidean_distance = F.pairwise_distance(output1, output2)

                if name == name_other:
                    same_histo.append(euclidean_distance.item())
                else:
                    diff_histo.append(euclidean_distance.item())
            models_checked[name] = {
                "same" : same_histo,
                "diff" : diff_histo
            }
            dataiter = iter(dataloader)

        models_checked["Summed"] = {
            "same" : [x for (_, v) in models_checked.items() for x in v['same']],
            "diff" : [x for (_, v) in models_checked.items() for x in v['diff']]
        }

            # histo_makers(same_histo, diff_histo, name, config["RESULT_FOLDER"]+ "/histo_together.png")  
        multiple_histo(models_checked, config["RESULT_FOLDER"]+f"/histo_multiple_{i}.png")


def train_model(net: SiameseNetwork, dataloader: DataLoader, start_time: float = None):
    DEVICE = config["DEVICE"]
    MAX_EPOCH = config["MAX_EPOCH"]
    
    optimizer = optim.Adam(net.parameters(), lr = 0.0005 )
    criterion = ContrastiveLoss()
    counter = []
    loss_history = [] 
    print(" === Training results === ")

    # Iterate throught the epochs
    for epoch in range(MAX_EPOCH):
        print(F"Currently at epoch {epoch+1:>{len(str(MAX_EPOCH))}}/{MAX_EPOCH} | Total time spent: {time.time()-start_time:>7.2f}s")

        # Iterate over batches
        avg_loss = 0
        for i, (img0, img1, label, _, _) in enumerate(dataloader, 0):
            # Send the images and labels to CUDA
            img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()


            avg_loss += loss_contrastive.item()
            # # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f" Current loss {loss_contrastive.item():.4f}")
        counter.append(epoch)
        loss_history.append(avg_loss/(i+1))
        print()
    save_plot(counter, loss_history, config["RESULT_FOLDER"])


def init(args):
    if getattr(args, 'dev'): # Local settings
        print("\u001b[31m == RUNNING IN DEV MODE == \u001b[0m")
        config["MAX_EPOCH"] = 5
        config["DATA_FOLDER"] = "Dresden/natural"
        config["RESULT_FOLDER"] = "Results"
        config["MODELS_FOLDER"] = "Models"
    else: # Peregrine settings
        print(" == RUNNING IN PEREGRINE MODE == ")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SNN model for camera identification')
    parser.add_argument('--dev', action='store_true',
        help='Runs in development mode')
    parser.add_argument('--epoch' , type=int, default=25,
        help='Determine the amount of epochs') 
    
    init(parser.parse_args())
    main(parser.parse_args())                            