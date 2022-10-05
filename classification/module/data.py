import numpy as np
from torchvision import datasets, transforms

def get_mnist_dataset(train_xform=transforms.ToTensor(), test_xform=transforms.ToTensor()):
    dset = {}
    dset['train'] = datasets.MNIST(path['mnist'], True, train_xform, download='False')
    dset['test'] = datasets.MNIST(path['mnist'], False, test_xform, download='False')
    return dset

def get_svhn_dataset(train_xform=transforms.ToTensor(), test_xform=transforms.ToTensor()):
    dset = {}
    dset['train'] = datasets.SVHN(path['svhn'], 'train', train_xform, download='False')
    dset['test'] = datasets.SVHN(path['svhn'], 'test', test_xform, download='False')
    return dset
