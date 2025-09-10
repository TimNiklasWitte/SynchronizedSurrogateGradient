import torch
import torch.nn as nn
import snntorch as snn

from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy

from config import *

class Classifier(torch.nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()



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

    
        self.cce_loss = LOSS
        self.optimizer = OPTIMIZER( self.parameters(), LEARNING_RATE)


        self.accuracy = Accuracy(task="multiclass", num_classes=OUTPUT_DIM)

        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()


    def forward(self, x):
        mem_rec = []
        spk_rec = []
    

        for step in range(self.num_steps):
 
            spk_out, mem_out = self.layers(x)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)
    
    @torch.no_grad
    def test(self, device, test_loader):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()

        for x, target in test_loader:
        
            x, target = x.to(device), target.to(device)

            predictions = self(x)
            loss = self.cce_loss(predictions, target)

            self.loss_metric.update(loss)

            predicated_labels = torch.argmax(predictions, dim=1)
            accuracy_batch = self.accuracy(predicated_labels, target)
            self.accuracy_metric.update(accuracy_batch)


        test_loss = self.loss_metric.compute()
        test_accuracy = self.accuracy_metric.compute()

        self.loss_metric.reset()
        self.accuracy_metric.reset()
        
        return test_loss, test_accuracy