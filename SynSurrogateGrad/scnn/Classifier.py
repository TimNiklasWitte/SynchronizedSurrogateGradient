import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from torchmetrics import MeanMetric
from snntorch import spikegen
from config import *

import tqdm

from SynSurrogateGrad.scnn.SynchronizedLeaky import SynchronizedLeaky
from SynSurrogateGrad.scnn.SynchronizedLeakyConv2D import SynchronizedLeakyConv2D
from SynSurrogateGrad.scnn.Divergence import compute_divergence


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()



        # Temporal Dynamics
        self.num_steps = NUM_STEPS
        beta = BETA

        self.syn_lif_conv_1 = SynchronizedLeakyConv2D(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=3, beta=beta)
        self.syn_lif_conv_2 = SynchronizedLeakyConv2D(in_channels=HIDDEN_DIM, out_channels=HIDDEN_DIM, kernel_size=3, beta=beta)
        
        self.syn_lif_1 = SynchronizedLeaky(in_features=int(HIDDEN_DIM*(IMG_W_H/4)*(IMG_W_H/4)), 
                                           out_features=HIDDEN_DIM, 
                                           beta=beta
                                        )
        
        self.syn_lif_2 = SynchronizedLeaky(in_features=HIDDEN_DIM, 
                                           out_features=OUTPUT_DIM, 
                                           beta=beta
                                        )

        # self.syn_lif_2 = SynchronizedLeaky(in_features=64, 
        #                                    out_features=64, 
        #                                    beta=beta
        #                                 )
        
        # self.syn_lif_3 = SynchronizedLeaky(in_features=64, 
        #                                    out_features=64, 
        #                                    beta=beta
        #                                 )

        # self.syn_lif_4 = SynchronizedLeaky(in_features=64, 
        #                                    out_features=10, 
        #                                    beta=beta
        #                                 )

        self.cce_rate_loss = SPIKE_LOSS

        self.optimizer = OPTIMIZER(self.parameters(), lr=LEARNING_RATE)

        #
        # Metrics
        #
        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()

        # Divergence
        self.num_layers = 4
        self.divergence_layer_metric_list = [
            MeanMetric() for _ in range(self.num_layers)
        ]


    def forward(self, x, is_spike):
        
        batch_size = x.shape[1]

        mem_1 = torch.zeros(size=(batch_size, HIDDEN_DIM, int(IMG_W_H/2),  int(IMG_W_H/2))).cuda()
        mem_2 = torch.zeros(size=(batch_size, HIDDEN_DIM, int(IMG_W_H/4),  int(IMG_W_H/4))).cuda()
        mem_3 = torch.zeros(size=(batch_size, HIDDEN_DIM)).cuda()
        mem_4 = torch.zeros(size=(batch_size, OUTPUT_DIM)).cuda()

        preds_list = []
        out_layerwise_list = []
      
        for step in range(self.num_steps):
            out_1, mem_1 = self.syn_lif_conv_1(x[step, ...], mem_1, is_spike)
            out_2, mem_2 = self.syn_lif_conv_2(out_1, mem_2, is_spike)

            out_2 = out_2.view(batch_size, -1)
            
            out_3, mem_3 = self.syn_lif_1(out_2, mem_3, is_spike)
            out_4, mem_4 = self.syn_lif_2(out_3, mem_4, is_spike)

            out_layerwise_list.append([out_1, out_2, out_3, out_4])

            preds_list.append(out_4)

        return torch.stack(preds_list, dim=0), out_layerwise_list

    
    @torch.no_grad
    def test(self, test_loader, device):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()

        for layer_idx in range(self.num_layers):
            self.divergence_layer_metric_list[layer_idx].reset()

        for x, targets in tqdm.tqdm(test_loader, position=0, leave=True):
        
            x, targets = x.to(device), targets.to(device)

            x = spikegen.rate(x, num_steps=self.num_steps)

            _, conti_act = self(x, is_spike=False)

            spk_rec, spike_act = self(x, is_spike=True)

            loss = self.cce_rate_loss(spk_rec, targets)

            #
            # Update metrics
            #

            # Loss
            self.loss_metric.update(loss)

            # Accuracy
            accuracy = SF.accuracy_rate(spk_rec, targets)
            self.accuracy_metric.update(accuracy)

            # Divergence
            divergence_layerwise = compute_divergence(spike_act, conti_act, layerwise=True)
            
            for layer_idx in range(self.num_layers):
                self.divergence_layer_metric_list[layer_idx].update(divergence_layerwise[layer_idx].cpu())

        
        test_loss = self.loss_metric.compute()
        test_accuracy = self.accuracy_metric.compute()
        
        divergence_layerwise = torch.zeros(size=(self.num_layers,))
        for layer_idx in range(self.num_layers):
            divergence_layerwise[layer_idx] = self.divergence_layer_metric_list[layer_idx].compute()

        divergence = torch.mean(divergence_layerwise)

        self.loss_metric.reset()
        self.accuracy_metric.reset()

        for layer_idx in range(self.num_layers):
            self.divergence_layer_metric_list[layer_idx].reset()
    
        return test_loss, test_accuracy, divergence_layerwise, divergence
    