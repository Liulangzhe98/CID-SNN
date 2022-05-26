# Imports
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn.functional as F

from os import path
import time
import argparse
import statistics


from SNN import *
from helper import *
from splitter import *

# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#TODO: Make this change for local and peregrine usage
MAX_EPOCH = 25       # TODO: Make this into an early termination value
SUBSET_SIZE = 25      # Used for testing and not blowing up my laptop
# Making sure that the program can run both locally and within peregrine

# Resize the images and transform to tensors
# TODO: The resize might remove pattern noise ???
# TODO: (100, 100) resize makes it so that the layers are applied correctly
transformation = transforms.Compose([
        # transforms.Resize((640,480)),
        transforms.CenterCrop((100, 100)),
        transforms.ToTensor()
    ])

DATA_FOLDER = "Dresden/natural"
if not path.exists(DATA_FOLDER):
    DATA_FOLDER = "Project/Dresden/natural"


def main(args):
    start_time = time.time()
    if getattr(args, 'dev'):
        print("\u001b[31m == RUNNING IN DEV MODE == \u001b[0m")

    print(f"Working with device: {DEVICE}\nWithin the folder: {DATA_FOLDER}")

    train_set, test_set = data_splitter(Path(DATA_FOLDER), 0.8)
    if getattr(args, 'dev'):
        train_set = random.sample(train_set, k=SUBSET_SIZE)
        test_set = random.sample(test_set, k=len(test_set)//25)
    else:
        # peregrine settings
        train_set = random.sample(train_set, k=len(train_set)//4)
        test_set = random.sample(test_set, k=len(test_set)) 
        

    # Load the training dataset
    # TODO: Make sure these values are working for our dataset
    train_dataloader = DataLoader(
        DresdenSNNDataset(image_paths=train_set, transform=transformation),
        shuffle=True, num_workers=8, batch_size=64)

    SNN_model = SiameseNetwork().to(DEVICE)

    print(f"The SNN network summary: \n{SNN_model}")
    train_model(SNN_model, train_dataloader, start_time)
    torch.save(SNN_model, f"Models/model.pth")


    # SNN_model = torch.load("Models/model.pth")

    # TODO: Make sure these values are working for our dataset
    test_dataloader = DataLoader(
        DresdenSNNDataset(image_paths=test_set, transform=transformation), 
        shuffle=True, num_workers=2, batch_size=1)

    validate_model(SNN_model, test_dataloader)


def validate_model(net: SiameseNetwork, dataloader: DataLoader):
    print(" === Validation results === ")
    print("  Lower Dissimilarity ( < 1) means they are probably from the same camera")
    # Grab one image that we are going to test
    dataiter = iter(dataloader)
    x0, _, _, f_name0, _ = next(dataiter)
    name = f_name0[0].split('/')[0]

    same_histo = []
    diff_histo = []
 
    for i, (_, x1, label2, _, f_name1) in enumerate(dataiter, 1):
        name_other = f_name1[0].split('/')[0]
        # print(f"[{i:5} / {len(dataiter)-1}] : {name} vs {name_other}")
        
        output1, output2 = net(x0.to(DEVICE), x1.to(DEVICE))
        euclidean_distance = F.pairwise_distance(output1, output2)

        # print(f'Dissimilarity: {euclidean_distance.item():>5.2f}', end = " | ")
        # print(f"{f_name0[0]} vs {f_name1[0]:<20}")
        if name == name_other:
            same_histo.append(euclidean_distance.item())
        else:
            diff_histo.append(euclidean_distance.item())
        # imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
    histo_makers(same_histo, diff_histo, name)
    print(f"  For the same camera pairs: mean = {statistics.mean(same_histo):5.2f} with std = {statistics.stdev(same_histo):5.2f}")
    print(f"  For the diff camera pairs: mean = {statistics.mean(diff_histo):5.2f} with std = {statistics.stdev(diff_histo):5.2f}")



def train_model(net: SiameseNetwork, dataloader: DataLoader, start_time: float = None):
    optimizer = optim.Adam(net.parameters(), lr = 0.0005 )
    criterion = ContrastiveLoss()
    counter = []
    loss_history = [] 
    iteration_number = 0
    print(" === Training results === ")

    # Iterate throught the epochs
    for epoch in range(MAX_EPOCH):
        print(F"Currently at epoch {epoch+1:>{len(str(MAX_EPOCH))}}/{MAX_EPOCH} | Total time spent: {time.time()-start_time:>7.2f}s")

        # Iterate over batches
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

            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f" Current loss {loss_contrastive.item():.4f}")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        print()
    print()
    save_plot(counter, loss_history)
    # show_plot(counter, loss_history)

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(
        description='SNN model for camera identification')

    parser.add_argument('--dev', action='store_true',
        help='Runs in development mode')
    parser.add_argument('--epoch' , type=int, default=25,
        help='Determine the amount of epochs') 
    print(type(parser.parse_args()))
    main(parser.parse_args())                            