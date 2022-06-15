# Imports
from torch.utils.data import DataLoader
from torch import optim

import time

from SNN import *
from helper import *
from splitter import *

def train_model(net: SiameseNetwork, dataloader: DataLoader, config):
    DEVICE = config["DEVICE"]
    MAX_EPOCH = config["MAX_EPOCH"]
    net.train(True)
    optimizer = optim.Adam(net.parameters(), lr = 0.0005)
    loss = torch.nn.MSELoss()
    counter = []
    loss_history = [] 
    print(" === Training results === ")
    start_time = time.time()

    # Iterate throught the epochs
    for epoch in range(MAX_EPOCH):
        e_time = time.time()
        # random.seed(22052022) # TODO: remove, was only for debugging
        print(F"Currently at epoch {epoch+1:>{len(str(MAX_EPOCH))}}/{MAX_EPOCH} | Total time spent: {time.time()-start_time:>7.2f}s")
        # Iterate over batches
        avg_loss = 0
        i = 0
        for i, (img0, img1, label, f1, f2) in enumerate(dataloader, 0): #img0, img1 and label are list of images
            # Send the images and labels to CUDA
            img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            prediction = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            output = loss(prediction, label)

            # for (correct, f1_, f2_, pred) in zip(label, f1, f2, prediction):
                # print(f"Breaking: {f1_.split('/')[0]:20} vs {f2_.split('/')[0]:20} | {pred} -> {correct}")
            
            # Calculate the backpropagation
            output.backward()

            # Optimize
            optimizer.step()

            avg_loss += output.item()
            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f"   Current loss {output.item():.4f}  | Time spent so far this epoch: {time.time()-e_time:>7.2f}s")
        print(f" Loss this epoch: {avg_loss/(i+1):.4f}")
        counter.append(epoch)
        loss_history.append(avg_loss/(i+1))
        print()
    save_plot(counter, loss_history, config["RESULT_FOLDER"])