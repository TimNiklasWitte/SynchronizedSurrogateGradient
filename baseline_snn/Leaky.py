import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F

import numpy as np

# Leaky neuron model, overriding the backward pass with a custom function
class Leaky(nn.Module):
    def __init__(self, beta, threshold=1.0):
        super(Leaky, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.SigmoidSurrogate.apply

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
      spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
      
      spk_soft = F.sigmoid(mem-self.threshold)
      
      
      reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
      mem = self.beta * mem + input_ - reset  # Eq (1)
      return spk, spk_soft, mem

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