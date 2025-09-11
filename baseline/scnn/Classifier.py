import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy

import torch.nn.functional as F

from config import *
from snntorch import spikegen

THRESHOLD = 1.0        # matches snn.Leaky default 
SLOPE     = 25.0       # match our surrogate slope

import torch

def compute_divergence(spk_list, spk_soft_list, layerwise=False):

    num_time_steps = len(spk_list)
    num_layers = len(spk_list[0])

    divergence = torch.zeros(size=(num_layers,)).cuda()
    for t in range(num_time_steps):
        
    
        for layer_idx, (spk, spk_soft) in enumerate(zip(spk_list[t], spk_soft_list[t])):
           
            divergence[layer_idx] += torch.mean( torch.abs(spk - spk_soft) )
    
    divergence = divergence / num_time_steps
    
    if not layerwise:
        divergence = torch.mean(divergence)

    return divergence

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()

        self.num_steps = NUM_STEPS
        self.num_spike_layer = 3

        self.layers = nn.Sequential(
            nn.Conv2d(1, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD),

            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD),

            nn.Flatten(),
            nn.Linear(HIDDEN_DIM*int((IMG_W_H/4)*(IMG_W_H/4)), HIDDEN_DIM),
            ACTIVATION,
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, output=True),
        )

        self.cce_loss = SPIKE_LOSS
        self.optimizer = OPTIMIZER( self.parameters(), LEARNING_RATE)


        self.accuracy = Accuracy(task="multiclass", num_classes=OUTPUT_DIM)

        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()
        self.div_layer_metrics = [MeanMetric() for _ in range( self.num_spike_layer)]


    def forward(self, x):
        spk_out_list = []

        spk_list = []
        spk_soft_list = []
    
         # reset hidden states
        for layer in self.layers:
            if isinstance(layer, snn.Leaky):
                layer.reset_hidden()

        for step in range(self.num_steps):
 
            cur_x = x[step,:]
            spk_step = []
            spk_soft_step = []
            last_spk = None

            for layer in self.layers:
                if isinstance(layer, snn.Leaky):
                    spk, mem = layer(cur_x)  # spk: binary, mem: membrane potential
                    # soft spikes from membrane
                    spk_soft = torch.sigmoid((mem - THRESHOLD)*SLOPE)  # threshold=1.0, slope=25
                    spk_step.append(spk)
                    spk_soft_step.append(spk_soft)
                    cur_x = spk
                    last_spk = spk
                else:
                    cur_x = layer(cur_x)

            spk_out_list.append(last_spk)
            spk_list.append(spk_step)
            spk_soft_list.append(spk_soft_step)

        return torch.stack(spk_out_list, dim=0), spk_list, spk_soft_list

    
    @torch.no_grad
    def test(self, test_loader, device):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()

        for layer_idx in range( self.num_spike_layer):
            self.div_layer_metrics[layer_idx].reset()

        for x, target in test_loader:
        
            x, target = x.to(device), target.to(device)

            
            x = spikegen.rate(x, num_steps=self.num_steps)

            spk_rec, spk_list, spk_soft_list = self(x)

            div_layerwise = compute_divergence(spk_list, spk_soft_list, layerwise=True)

            loss = self.cce_loss(spk_rec, target)

            self.loss_metric.update(loss)

            accuracy_batch = SF.accuracy_rate(spk_rec, target)
            self.accuracy_metric.update(accuracy_batch)

            for i, d in enumerate(div_layerwise):
                self.div_layer_metrics[i].update(d.cpu())


        test_loss = self.loss_metric.compute()
        test_accuracy = self.accuracy_metric.compute()

        div_layerwise = torch.zeros(size=( self.num_spike_layer,))
        for layer_idx in range( self.num_spike_layer):
            div_layerwise[layer_idx] = self.div_layer_metrics[layer_idx].compute()


        self.loss_metric.reset()
        self.accuracy_metric.reset()
        
        divergence = torch.mean(div_layerwise)
        for layer_idx in range( self.num_spike_layer):
            self.div_layer_metrics[layer_idx].reset()

        return test_loss, test_accuracy, div_layerwise, divergence