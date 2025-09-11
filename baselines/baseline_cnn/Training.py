import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF

from baseline_cnn import Classifier as b_cnn
from config import *

import tqdm

file_path = f"./baseline_cnn/logs/"

def compute_divergence(spk_list, spk_soft_list):

    num_time_steps = len(spk_list)
 
    divergence = 0
    for t in range(num_time_steps):
        
    
        for spk, spk_soft in zip(spk_list[t], spk_soft_list[t]):
            divergence += torch.mean( (spk - spk_soft)**2 )
    
    divergence = divergence / num_time_steps

    return divergence


def main():

    #
    # Device
    #
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    # Augmentation
    #

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    #
    # Dataset
    #
    
    data_path = "./../FashionMNIST"
    train_ds = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform, download=True)


    #
    # Data loaders
    #

    train_loader = DataLoader(train_ds, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_THREADS, 
                              shuffle=True, 
                              drop_last=True
                        )
    
    test_loader = DataLoader(test_ds, 
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_THREADS, 
                             shuffle=True, 
                             drop_last=True
                        )

    #
    # Logging
    #

    writer = SummaryWriter(file_path)


    
    #
    # Init Model
    #

    model = b_cnn.Classifier()
    model.to(device)

    #
    # Train loop
    #
    for epoch in range(NUM_EPOCHS + 1):
        
        print(f"Epoch {epoch}")

        # Epoch 0 = no training steps are performed 
        # test based on train data
        # -> Determinate initial train_loss and train_accuracy
        if epoch == 0:

            continue
            train_loss, train_accuracy = model.test(train_loader, device)



        else:

            model.train()

            for x, target in tqdm.tqdm(train_loader, position=0, leave=True):
                # Transfer data to GPU (if available)
                x, target = x.to(device), target.to(device)

                # Reset gradients
                model.optimizer.zero_grad()

                # Forward pass
                predictions = model(x)

                # Calc loss
                loss = model.cce_loss(predictions, target)

                # Backprob
                loss.backward()

                # Update parameters
                model.optimizer.step()

                #
                # Update metrics
                #

                # Loss
                model.loss_metric.update(loss)
                
                # Accuracy
                predicated_labels = torch.argmax(predictions, dim=1)
                accuracy_batch = model.accuracy(predicated_labels, target)
                model.accuracy_metric.update(accuracy_batch)

   
            train_loss = model.loss_metric.compute()
            train_accuracy = model.accuracy_metric.compute()

        test_loss, test_accuracy = model.test(device, test_loader)

        #
        # Output
        #
        print(f"    train_loss: {train_loss}")
        print(f"     test_loss: {test_loss}")
        print(f"train_accuracy: {train_accuracy}")
        print(f" test_accuracy: {test_accuracy}")
  
        #
        # Logging
        #
        writer.add_scalars("Loss",
                            { "Train" : train_loss, "Test" : test_loss },
                            epoch)
        
        writer.add_scalars("Accuracy",
                            { "Train" : train_accuracy, "Test" : test_accuracy },
                            epoch)
        
        writer.flush()

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")