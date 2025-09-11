import torch
import torch.nn as nn

from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy

from config import *

class Classifier(torch.nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()



        self.layers = nn.Sequential(
            nn.Conv2d(1, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            ACTIVATION,
            nn.MaxPool2d(2,2),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            ACTIVATION,
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(HIDDEN_DIM*int((IMG_W_H/4)*(IMG_W_H/4)), HIDDEN_DIM),
            ACTIVATION,
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    
        self.cce_loss = LOSS
        self.optimizer = OPTIMIZER( self.parameters(), LEARNING_RATE)


        self.accuracy = Accuracy(task="multiclass", num_classes=OUTPUT_DIM)

        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()


    def forward(self, x):
        x = self.layers(x)
        return x
    
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