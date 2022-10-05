import torch
import torch.nn as nn

def UncertaintyLoss(std, size):
    return torch.norm(std, 1) / size

def Uncertainty2Loss(std):
    return torch.norm(std, dim=1).mean(0)

