import torch
import torch.nn as nn

from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy

from config import *

class Classifier(torch.nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            ACTIVATION,
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            ACTIVATION,
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            ACTIVATION,
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    
        self.cce_loss = LOSS
        self.optimizer = OPTIMIZER( self.parameters(), LEARNING_RATE)


        self.accuracy = Accuracy(task="multiclass", num_classes=OUTPUT_DIM)

        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()


    def forward(self, x):
        x = x.view(x.size(0), -1)
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