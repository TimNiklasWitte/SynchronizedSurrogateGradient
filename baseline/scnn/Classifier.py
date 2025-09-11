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

def soft_from_mem(mem_rec, threshold=THRESHOLD, slope=SLOPE):
    # mem_rec: (T, B, num_classes) –> soft spike probability in [0,1]
    return torch.sigmoid((mem_rec - threshold) * slope)

def divergence_mse(spk_rec, mem_rec):
    spk_soft = soft_from_mem(mem_rec)
    # both (T, B, C); MSE averaged over all dims
    return F.mse_loss(spk_rec.float(), spk_soft)

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()

        self.num_steps = NUM_STEPS

        self.layers = nn.Sequential(
            nn.Conv2d(1, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=True),

            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=True),

            nn.Flatten(),
            nn.Linear(HIDDEN_DIM*int((IMG_W_H/4)*(IMG_W_H/4)), HIDDEN_DIM),
            ACTIVATION,
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=True, output=True),
        )

        self.cce_loss = SPIKE_LOSS
        self.optimizer = OPTIMIZER( self.parameters(), LEARNING_RATE)


        self.accuracy = Accuracy(task="multiclass", num_classes=OUTPUT_DIM)

        self.loss_metric = MeanMetric()
        self.div_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()


    def forward(self, x):
        mem_rec = []
        spk_rec = []
    
         # reset hidden states
        for layer in self.layers:
            if isinstance(layer, snn.Leaky):
                layer.reset_hidden()

        for step in range(self.num_steps):
 
            cur_x = x[step,:]
            spk_out, mem_out = self.layers(cur_x)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        # stack along time dimension: (T, B, num_classes)
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

    
    @torch.no_grad
    def test(self, device, test_loader):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()
        self.div_metric.reset()

        for x, target in test_loader:
        
            x, target = x.to(device), target.to(device)

            x = spikegen.rate(x, num_steps=self.num_steps)

            spk_rec, mem_rec = self(x)
            div = divergence_mse(spk_rec, mem_rec)
            self.div_metric.update(div)
            loss = self.cce_loss(spk_rec, target)
            predictions = mem_rec.mean(0)

            self.loss_metric.update(loss)

            predicated_labels = torch.argmax(predictions, dim=1)
            accuracy_batch = self.accuracy(predicated_labels, target)
            self.accuracy_metric.update(accuracy_batch)


        test_loss = self.loss_metric.compute()
        test_accuracy = self.accuracy_metric.compute()

        self.loss_metric.reset()
        self.accuracy_metric.reset()
        
        test_div = self.div_metric.compute()
        self.div_metric.reset()
        return test_loss, test_accuracy, test_div