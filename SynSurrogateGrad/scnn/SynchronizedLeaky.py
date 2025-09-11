import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

# Leaky neuron model, overriding the backward pass with a custom function
class SynchronizedLeaky(nn.Module):
    def __init__(self, in_features, out_features, beta, threshold=1.0):
        super(SynchronizedLeaky, self).__init__()

        #
        # Parameters
        #

        # spike
        self.w_spike = nn.Parameter(torch.empty(in_features, out_features))
        init.xavier_uniform_(self.w_spike)

        self.b_spike = nn.Parameter(torch.zeros(out_features))

        # conti
        self.w_hat_conti = nn.Parameter(torch.empty(in_features, out_features))
        init.xavier_uniform_(self.w_hat_conti)

        self.b_hat_conti = nn.Parameter(torch.zeros(out_features))


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

            x = x @ self.w_spike + self.b_spike

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


            x = x @ w_conti + b_conti

            out = F.sigmoid(mem-self.threshold)
        
        
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