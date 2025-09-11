import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

# Leaky neuron model, overriding the backward pass with a custom function
class SynchronizedLeakyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, beta, threshold=1.0):
        super(SynchronizedLeakyConv2D, self).__init__()

        #
        # Parameters
        #

        # spike
        self.w_spike = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        init.xavier_uniform_(self.w_spike)

        self.b_spike = nn.Parameter(torch.zeros(out_channels))

        # conti
        self.w_hat_conti = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        init.xavier_uniform_(self.w_hat_conti)

        self.b_hat_conti = nn.Parameter(torch.zeros(out_channels))

        self.pool = torch.nn.MaxPool2d(2,2)

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.SigmoidSurrogate.apply

    # the forward function is called each time we call Leaky
    def forward(self, x, mem, is_spike):
        
        #
        # spike
        #

        spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
        out = spk 

        if is_spike:
            
            self.w_spike.requires_grad_(True)
            self.b_spike.requires_grad_(True)

            self.w_hat_conti.requires_grad_(False)
            self.b_hat_conti.requires_grad_(False)

            x = F.conv2d(x, weight=self.w_spike, bias=self.b_spike, stride=1, padding=1)

        #
        # conti
        #

        else:
            
            self.w_spike.requires_grad_(False)
            self.b_spike.requires_grad_(False)

            self.w_hat_conti.requires_grad_(True)
            self.b_hat_conti.requires_grad_(True)

            w_conti = self.w_spike + self.w_hat_conti
            b_conti = self.b_spike + self.b_hat_conti


            x = F.conv2d(x, weight=w_conti, bias=b_conti, stride=1, padding=1)

            out = F.sigmoid(mem-self.threshold)
        

        x = self.pool(x)

        reset = (self.beta * spk * self.threshold).detach() 
        mem = self.beta * mem + x - reset 

        return out, mem

    # Forward pass: Heaviside function
    # Surrogate gradient with sigmoid derivative
    class SigmoidSurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            
            spk = (mem > 0).float()  
            ctx.save_for_backward(mem)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors
            # sigmoid derivative as surrogate
            sig = torch.sigmoid(mem)
            grad = sig * (1 - sig) * grad_output
            return grad