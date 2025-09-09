import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from torchmetrics import MeanMetric

import tqdm

from Leaky import *


class Classifier(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        self.num_layers = num_layers

        num_inputs = 28*28
        self.num_hidden = 64
        self.num_outputs = 10

        # Temporal Dynamics
        self.num_steps = 25
        beta = 0.95


        # Initialize layers


        self.layer_list = nn.ModuleList(
            [nn.Linear(num_inputs, self.num_hidden)] +
            [nn.Linear(self.num_hidden, self.num_hidden) for _ in range(num_layers - 1)] +
            [nn.Linear(self.num_hidden, self.num_outputs)]
        )  

        self.lif = Leaky(beta=beta)




        self.cce_rate_loss = SF.ce_rate_loss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        #
        # Metrics
        #
        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()

    def forward(self, x):
        
        batch_size = x.shape[0]
        # Initialize hidden states at t=0

        mem_list = (
            [torch.zeros(size=(batch_size, self.num_hidden)).cuda()] + 
            [torch.zeros(size=(batch_size, self.num_hidden)).cuda() for _ in range(self.num_layers - 1)] +
            [torch.zeros(size=(batch_size, self.num_outputs)).cuda()]
        )

        spk_rec = []
        spk_soft_rec = []

        for step in range(self.num_steps):
            
            spk_rec_tmp = []
            spk_soft_rec_tmp = []

            x_tmp = x.clone()
            for idx, layer in enumerate(self.layer_list):
           
                x_tmp = layer(x_tmp)
                
                spk, spk_soft, mem_list[idx] = self.lif(x_tmp, mem_list[idx])

                spk_rec_tmp.append(spk)
                spk_soft_rec_tmp.append(spk_soft)

            spk_rec.append(spk_rec_tmp)
            spk_soft_rec.append(spk_soft_rec_tmp)

        return spk_rec, spk_soft_rec

    
    @torch.no_grad
    def test(self, test_loader, device):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()

        for x, targets in tqdm.tqdm(test_loader, position=0, leave=True):
        
            x, targets = x.to(device), targets.to(device)

            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

            spk_rec = self(x)
            loss = self.cce_rate_loss(spk_rec, targets)

            self.loss_metric.update(loss)

            # Accuracy
            accuracy = SF.accuracy_rate(spk_rec, targets)
            self.accuracy_metric.update(accuracy)


        test_loss = self.loss_metric.compute()
        test_accuracy = self.accuracy_metric.compute()

        self.loss_metric.reset()
        self.accuracy_metric.reset()
        
        return test_loss, test_accuracy
    