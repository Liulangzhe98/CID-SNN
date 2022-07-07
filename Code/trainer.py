# Used for typing annotation
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configure import Configure
    from SNN import SiameseNetwork


# Imports
from torch.utils.data import DataLoader as DL
from torch import optim, device
import torch.nn as nn


import time

from SNN import *
from helper import *
from splitter import *


def train_one_epoch(net: SiameseNetwork, dataloader: DL,
                    optimizer: optim, e_time: float, DEVICE: device) -> float:
    """ Function that will train the model for one epoch

    Args:
        net (SiameseNetwork): 
            The model to be trained
        dataloader (Dataloader): 
            The training dataloader that provides the images in batches
        optimizer (torch.optim): 
            The optimizer used for training
        e_time (float): 
            The start time of this epoch
        DEVICE (device): 
            The device used GPU when available

    Returns:
        float: Return the average loss of this epoch
    """

    running_loss = 0.
    loss_func = nn.MSELoss()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (img0, img1, label, f1, f2) in enumerate(dataloader, 1):
        # img0, img1 are Tensors in the format:
        #   [<batch size>, <in channel>, <img data> <img data>]

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

        # Every 100 batches print out the loss
        if i % 100 == 0:
            print(
                f"   Current loss {loss.item():.4f} |"
                f" Time spent so far this epoch: {time.time()-e_time:>7.2f}s")

    return running_loss / i


def validation(net: SiameseNetwork, v_data: DL, DEVICE: device) -> float:
    """Function that will validate the model during training

    Args:
        net (SiameseNetwork):  
            The model to be validated
        v_data (DataLoader): 
            The training dataloader that provides the images in batches
        DEVICE (device): 
            The device used GPU when available

    Returns:
        float: Return the average validation loss of this epoch
    """

    running_vloss = 0.0
    loss_func = nn.MSELoss()

    with torch.no_grad():
        i = 0
        for i, (img0, img1, label, f1, f2) in enumerate(v_data, 1):
            # img0, img1 are Tensors in the format:
            #   [<batch size>, <in channel>, <img data> <img data>]
            img0, img1, label = (
                img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE))
            prediction = net(img0, img1)
            vloss = loss_func(prediction, label)
            running_vloss += vloss

    return running_vloss / i


def finished_training(loss_history: list, config: Configure) -> None:
    save_plot(loss_history, config.LOSS_GRAPH)


def train_model(net: SiameseNetwork, t_data: DL, v_data: DL, config: Configure) -> None:
    MAX_EPOCH = config.MAX_EPOCH
    optimizer = optim.Adam(net.parameters(), lr=config.HYPER["LR"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.HYPER["DECAY"])

    loss_history = []

    the_last_loss = 100
    patience = 3
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
            net, t_data, optimizer, time.time(), config.DEVICE)

        net.train(False)
        avg_vloss = validation(net, v_data, config.DEVICE)

        lr_scheduler.step()
        loss_history.append((avg_loss, avg_vloss.item()))
        print(f"Loss T: {avg_loss:.4f} | Loss V: {avg_vloss:.4f}")

        if avg_vloss > the_last_loss:
            trigger_times += 1
            print("Trigger Times:", trigger_times)

            if trigger_times >= patience:
                print("Early Stopping!\nStart to test process.")
                finished_training(loss_history, config)
                return
        else:
            trigger_times = 0
        the_last_loss = avg_vloss

        print()

    finished_training(loss_history, config)


def preload_training(transformation: tf, train_set: list,
                     valid_set: list) -> Tuple[SNNDataset, SNNDataset]:
    """Wrapper for the preloading of training/validation images

    Args:
        transformation (torchvision.transformation): 
                Transformations that should be applied to the images
        train_set (list) : 
                The list of image paths for training
        valid_set (list) : 
                The list of image paths for validation

    Returns:
        tuple(SNNDataset, SNNDataset): 
            Return the SNNDataset object of both the training and validation set
    """

    start = time.time()
    print("Pre loading training images")
    train_data = SNNDataset(image_paths=train_set, transform=transformation)
    valid_data = SNNDataset(image_paths=valid_set, transform=transformation)
    print(f"Loading all images took: {time.time()-start:.4f}s")
    return train_data, valid_data