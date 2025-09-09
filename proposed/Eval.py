import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from Classifier import *

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
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.Normalize((0.1307,), (0.3081,))  # mean & std for MNIST      
    ])

    #
    # Dataset
    #
    
    data_path = "./mnist_data"
    test_ds = torchvision.datasets.MNIST(data_path, train=False, transform=transform, download=True)

    #
    # Data loaders
    #
 
    test_loader = DataLoader(test_ds, 
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_THREADS, 
                             shuffle=True, 
                             drop_last=True
                        )
    
    #
    # Init Model
    #

    model = Classifier(num_layers=32)
    model.to(device)

  
    with torch.no_grad():

        num_batches = 0
        differences = torch.zeros(size=(model.num_steps, model.num_layers), device=device)
        for x, targets in tqdm.tqdm(test_loader, position=0, leave=True):
            
        
            # Transfer data to GPU (if available)
            x, targets = x.to(device), targets.to(device)

        
            x = x.view(BATCH_SIZE, -1)

            spks, spks_soft = model(x)


            for t in range(model.num_steps):
                for layer_idx in range(model.num_layers):
                    differences[t, layer_idx] += torch.mean((spks[t][layer_idx] - spks_soft[t][layer_idx])**2)

            
            num_batches += 1

    differences = differences / num_batches
  
    differences = torch.mean(differences, dim=0)
    
    differences = differences.detach().cpu().numpy()

    plt.plot(differences)
    plt.xlabel("Layer idx")
    plt.ylabel("MSE between spk and spk_soft")
    plt.grid()
    plt.tight_layout()

    plt.savefig("./mse_between_spk_and_spk_soft.png", dpi=200)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")