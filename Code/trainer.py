# Imports
from torch.utils.data import DataLoader
from torch import optim

import torch.nn as nn

import time

from SNN import *
from helper import *
from splitter import *

def train_one_epoch(net, dataloader, loss_func, optimizer, e_time, config):
    running_loss = 0.
    
    DEVICE = config["DEVICE"]

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (img0, img1, label, f1, f2) in enumerate(dataloader, 1): #img0, img1 and label are list of images
        # Send the images and labels to CUDA
        img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
        print(img0.size())

        
        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        prediction = net(img0, img1)
       
        # Pass the outputs of the networks and label into the loss function & backpropagate
        loss = loss_func(prediction, label)
        loss.backward()
        exit()

        # Optimize
        optimizer.step()
        
        running_loss += loss.item()

        if torch.cuda.is_available():
            del img0, img1, label

            torch.cuda.synchronize()
        
        # Every 100 batches print out the loss
        if i % 10 == 0 :
            print(f"   Current loss {loss.item():.4f}  | Time spent so far this epoch: {time.time()-e_time:>7.2f}s")

    return running_loss / i


def train_model(net: SiameseNetwork, t_data: DataLoader, v_data: DataLoader, config):
    MAX_EPOCH = config["MAX_EPOCH"]
    DEVICE = config["DEVICE"]
    optimizer = optim.Adam(net.parameters(), lr = 0.0003)
    decayRate = 0.90
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decayRate)

    loss_func = nn.MSELoss()
    counter = []
    loss_history = [] 

    print(" === Training results === ")
    start_time = time.time()

    # Iterate throught the epochs
    for epoch in range(MAX_EPOCH):
        # random.seed(22052022) # TODO: remove, was only for debugging
        print(F"Currently at epoch {epoch+1:>{len(str(MAX_EPOCH))}}/{MAX_EPOCH} | Total time spent: {time.time()-start_time:>7.2f}s")
        
        net.train(True)
        avg_loss = train_one_epoch(net, t_data, loss_func, optimizer, time.time(), config)
        
        net.train(False)
        running_vloss = 0.0

        with torch.no_grad():
            for i, (img0, img1, label, f1, f2) in enumerate(v_data, 1): #img0, img1 and label are list of images
                img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
                prediction = net(img0, img1)
                vloss = loss_func(prediction, label)
                running_vloss += vloss

                if i % 10 == 0:
                    torch.cuda.empty_cache()
        
        avg_vloss = running_vloss / i
        print(f"Loss T: {avg_loss:.4f} | Loss V: {avg_vloss:.4f}")

        lr_scheduler.step()
        counter.append(epoch)
        loss_history.append((avg_loss, avg_vloss.item()))
        print()
    save_plot(counter, loss_history, ["Avg loss (Train)", "Avg loss (Val)"], config["RESULT_FOLDER"])