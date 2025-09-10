import torch.nn as nn
import torch.optim as optim
from snntorch import surrogate


#3 Dense Layer 
#MNIST 28x28 -> 64->64->64->10 
IMG_W_H = 28

INPUT_DIM = IMG_W_H*IMG_W_H
HIDDEN_DIM = 64
OUTPUT_DIM = 10

ACTIVATION = nn.Sigmoid()
BATCH_SIZE = 128
OPTIMIZER = optim.Adam
LOSS = nn.CrossEntropyLoss()
LEARNING_RATE = 0.0001
NUM_EPOCHS = 32

NUM_THREADS = 16

BETA = 0.9
NUM_STEPS = 25
SPIKE_GRAD = surrogate.fast_sigmoid()