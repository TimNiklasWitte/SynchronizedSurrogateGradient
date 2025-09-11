import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF

from baseline.scnn import Classifier as b_scnn
from snntorch import spikegen
from config import *

import tqdm

file_path = f".baselines/baseline_scnn/logs/"




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

    model = b_scnn.Classifier()
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
                x = spikegen.rate(x, num_steps=model.num_steps)
                spk_rec, spike_act, spike_soft = model(x)
                # Calc loss
                loss = model.cce_loss(spk_rec, target)

                # divergence
                divergence_layerwise = b_scnn.compute_divergence(spike_act, spike_soft, layerwise=True)
                divergence = torch.mean(divergence_layerwise)
                divergence.backward(retain_graph=True)

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
                accuracy = SF.accuracy_rate(spk_rec, target)
                model.accuracy_metric.update(accuracy)
                
                # Divergence

                with torch.no_grad():
                    #divergence_layerwise = compute_divergence(spk_list, spk_soft_list, layerwise=True)
                    
                    for layer_idx in range(model.num_spike_layer):
                        model.div_layer_metrics[layer_idx].update(divergence_layerwise[layer_idx].cpu())


   
            train_loss = model.loss_metric.compute()
            train_accuracy = model.accuracy_metric.compute()

            train_divergence_layerwise = torch.zeros(size=(model.num_spike_layer,))
            for layer_idx in range(model.num_spike_layer):
                train_divergence_layerwise[layer_idx] = model.div_layer_metrics[layer_idx].compute()

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
        
        for layer_idx in range(model.num_spike_layer):
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