import torch
import torchvision
import numpy as np
import random
import os
from platform import python_version


# ----------------------------------------------------------------------------
class AverageMeter(object):
    def __init__(self, alpha=0.95):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.alpha = alpha

    def update(self, val):
        self.val = val
        self.count += 1
        if self.count == 1:
            self.avg = self.val
        else:
            self.avg = self.alpha * self.avg + (1-self.alpha) * self.val

# ----------------------------------------------------------------------------
def init_random_seed(rand_number, printing=False):
    random.seed(rand_number)
    np.random.seed(rand_number)
    torch.manual_seed(rand_number)
    torch.cuda.manual_seed(rand_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if printing:
        print('initialized with random seed = {}'.format(rand_number))

def get_device(num_device=0):
    use_cuda = torch.cuda.is_available()
    print('available cuda devices: {}'.format(range(torch.cuda.device_count())))
    device = torch.device("cuda:{}".format(num_device) if use_cuda else "cpu")
    print('currently selected device: {}'.format(torch.cuda.current_device()))
    return device


# ----------------------------------------------------------------------------
def test(device, G, H, dataloader):
    G.eval(), H.eval()
    correct = 0
    tot_data = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logit = H(G(images, dropout=False), dropout=False)
            preds = logit.max(1, keepdim=True)[1]
            correct += preds.eq(labels.view_as(preds)).sum().item()
            tot_data += len(labels)

        accuracy = 1. * correct / tot_data

    G.train(), H.train()
    return accuracy
