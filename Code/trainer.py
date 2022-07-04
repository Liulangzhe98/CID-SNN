# Imports
from torch.utils.data import DataLoader as DL
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

    # img0, img1 are Tensors in the format:
    #   [<batch size>, <in channel>, <img data> <img data>]
    for i, (img0, img1, label, f1, f2) in enumerate(dataloader, 1):
        # Send the images and labels to CUDA
        img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        prediction = net(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        #  and backpropagate
        loss = loss_func(prediction, label)
        loss.backward()

        # Optimize
        optimizer.step()

        running_loss += loss.item()

        # TODO: Not sure if this is still needed
        if torch.cuda.is_available():
            del img0, img1, label
            torch.cuda.synchronize()

        # Every 100 batches print out the loss
        if i % 100 == 0:
            print(
                f"   Current loss {loss.item():.4f} |"
                f" Time spent so far this epoch: {time.time()-e_time:>7.2f}s")

    return running_loss / i


def validation(net, DEVICE, v_data, loss_func):
    running_vloss = 0.0

    with torch.no_grad():
        # img0, img1 and label are list of images
        i = 0
        for i, (img0, img1, label, f1, f2) in enumerate(v_data, 1):
            img0, img1, label = img0.to(DEVICE), img1.to(
                DEVICE), label.to(DEVICE)
            prediction = net(img0, img1)
            vloss = loss_func(prediction, label)
            running_vloss += vloss

    return running_vloss / i



def train_model(net: SiameseNetwork, t_data: DL, v_data: DL, config):
    MAX_EPOCH = config["MAX_EPOCH"]
    DEVICE = config["DEVICE"]
    UID = config['UID']
    optimizer = optim.Adam(net.parameters(), lr=config['HYPER']["LR"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config['HYPER']["DECAY"])

    loss_func = nn.MSELoss()
    counter = []
    loss_history = []
    
    the_last_loss = 100
    patience = 2
    trigger_times = 0

    print(" === Training results === ")
    start_time = time.time()

    # Iterate throught the epochs
    for epoch in range(MAX_EPOCH):
        # random.seed(22052022) # TODO: remove, was only for debugging
        print(
            f"Currently at epoch {epoch+1:>{len(str(MAX_EPOCH))}}/{MAX_EPOCH} |"
            f" Total time spent: {time.time()-start_time:>7.2f}s")

        net.train(True)
        avg_loss = train_one_epoch(
            net, t_data, loss_func, optimizer, time.time(), config)

        net.train(False)
        avg_vloss = validation(net, DEVICE, v_data, loss_func)

        lr_scheduler.step()
        counter.append(epoch)
        loss_history.append((avg_loss, avg_vloss.item()))
        print(f"Loss T: {avg_loss:.4f} | Loss V: {avg_vloss:.4f}")

        if avg_vloss > the_last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early Stopping!\nStart to test process.')
                return
        the_last_loss = avg_vloss

    
        print()

    save_plot(counter, loss_history, ["Avg loss (Train)", "Avg loss (Val)"], 
        f'{config["RESULT_FOLDER"]}/loss_graph_{UID}.png')
   
