import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from torchmetrics import MeanMetric

import tqdm

from Leaky import *


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()



        # Temporal Dynamics
        self.num_steps = 25
        beta = 0.95


        # Initialize layers

        self.linear_1 = nn.Linear(28*28, 64)
        self.lif_1 = Leaky(beta=beta)

        self.linear_2 = nn.Linear(64, 64)
        self.lif_2 = Leaky(beta=beta)

        self.linear_3 = nn.Linear(64, 10)
        self.lif_3 = Leaky(beta=beta)

        self.cce_rate_loss = SF.ce_rate_loss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        #
        # Metrics
        #
        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()

    def forward(self, x):
        
        batch_size = x.shape[0]

        mem_1 = torch.zeros(size=(batch_size, 64)).cuda()
        mem_2 = torch.zeros(size=(batch_size, 64)).cuda()
        mem_3 = torch.zeros(size=(batch_size, 10)).cuda()

        spk_out_list = []

        spk_list = []
        spk_soft_list = []

        for step in range(self.num_steps):
 
            x_dash = self.linear_1(x)
            spk_1, spk_1_soft, mem_1 = self.lif_1(x_dash, mem_1)

            spk_1 = self.linear_2(spk_1)
            spk_2, spk_2_soft, mem_2 = self.lif_2(spk_1, mem_2)

            spk_3 = self.linear_3(spk_2)
            spk_out, spk_out_soft, mem_3 = self.lif_3(spk_3, mem_3)
            
            #print(spk_1.shape, spk_2.shape, spk_out.shape)
            #print(spk_1_soft.shape, spk_2_soft.shape, spk_out_soft.shape)
            #exit(0)
            spk_list.append([spk_1, spk_2, spk_out])
            spk_soft_list.append([spk_1_soft, spk_2_soft, spk_out_soft])

    
            spk_out_list.append(spk_out)

        return torch.stack(spk_out_list, dim=0), spk_list, spk_soft_list

    
    @torch.no_grad
    def test(self, test_loader, device):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()

        for x, targets in tqdm.tqdm(test_loader, position=0, leave=True):
        
            x, targets = x.to(device), targets.to(device)

            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

            spk_rec, _, _ = self(x)
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
    