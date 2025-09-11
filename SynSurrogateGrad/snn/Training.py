import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF

from snntorch import spikegen

from Classifier import *

import tqdm

NUM_EPOCHS = 32
BATCH_SIZE = 128
NUM_THREADS = 16


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

    file_path = f"./logs/"

    writer = SummaryWriter(file_path)


    
    #
    # Init Model
    #

    model = Classifier()
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

            train_loss, train_accuracy, train_divergence_layerwise, train_divergence = model.test(train_loader, device)

        else:

            model.train()

            for x, targets in tqdm.tqdm(train_loader, position=0, leave=True):
                # Transfer data to GPU (if available)
                x, targets = x.to(device), targets.to(device)

                # Reset gradients
                model.optimizer.zero_grad()

                # Forward pass

                x = spikegen.rate(x, num_steps=model.num_steps)
                x = x.view(model.num_steps, BATCH_SIZE, -1)

                #
                # Conti
                #

                _, conti_act = model(x, is_spike=False)

                spk_rec, spike_act = model(x, is_spike=True)


                divergence_layerwise = compute_divergence(conti_act, spike_act, layerwise=True)
                divergence = torch.mean(divergence_layerwise)
                divergence.backward(retain_graph=True)


                # Calc loss
                loss = model.cce_rate_loss(spk_rec, targets)

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
                accuracy = SF.accuracy_rate(spk_rec, targets)
                model.accuracy_metric.update(accuracy)

                # Divergence

                with torch.no_grad():
                    #divergence_layerwise = compute_divergence(spk_list, spk_soft_list, layerwise=True)
                    
                    for layer_idx in range(model.num_layers):
                        model.divergence_layer_metric_list[layer_idx].update(divergence_layerwise[layer_idx].cpu())

            train_loss = model.loss_metric.compute()
            train_accuracy = model.accuracy_metric.compute()

            train_divergence_layerwise = torch.zeros(size=(model.num_layers,))
            for layer_idx in range(model.num_layers):
                train_divergence_layerwise[layer_idx] = model.divergence_layer_metric_list[layer_idx].compute()

            train_divergence = torch.mean(train_divergence_layerwise)

        test_loss, test_accuracy, test_divergence_layerwise, test_divergence = model.test(test_loader, device)

        #
        # Output
        #
        print(f"      train_loss: {train_loss}")
        print(f"       test_loss: {test_loss}")
        print(f"  train_accuracy: {train_accuracy}")
        print(f"   test_accuracy: {test_accuracy}")
        print(f"train_divergence: {train_divergence}")
        print(f" test_divergence: {test_divergence}")

        #
        # Logging
        #
        writer.add_scalars("Loss",
                            { "Train" : train_loss, "Test" : test_loss },
                            epoch)
        
        writer.add_scalars("Accuracy",
                            { "Train" : train_accuracy, "Test" : test_accuracy },
                            epoch)
        
        for layer_idx in range(model.num_layers):
            writer.add_scalars(f"Divergence_layer_{layer_idx}",
                            { "Train" : train_divergence_layerwise[layer_idx], "Test" : test_divergence_layerwise[layer_idx] },
                            epoch)
        
        writer.add_scalars("Divergence",
                            { "Train" : train_divergence, "Test" : test_divergence },
                            epoch)
        
        writer.flush()

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")