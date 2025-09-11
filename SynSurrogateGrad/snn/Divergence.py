import torch

def compute_divergence(spk_list, spk_soft_list, layerwise=False):

    num_time_steps = len(spk_list)
    num_layers = len(spk_list[0])

    divergence = torch.zeros(size=(num_layers,)).cuda()
    for t in range(num_time_steps):
        
    
        for layer_idx, (spk, spk_soft) in enumerate(zip(spk_list[t], spk_soft_list[t])):
           
            divergence[layer_idx] += torch.mean( torch.abs(spk - spk_soft) )
    
    divergence = divergence / num_time_steps
    
    if not layerwise:
        divergence = torch.mean(divergence)

    return divergence