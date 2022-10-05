import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from module.network import *
from module.loss import *
from module.util import *
from module.data import *
from module.muda import *
from module.transform import *

def main():

    arg = {}
    path = {}
    dset = {}
    loader = {}
    domain = {}
    ##### TO BE EDITED ###############################
    path['mnist'] = 'dataset/mnist/'
    path['svhn'] = 'dataset/svhn/'
    source_model_path = 'model/svhn_source_ckpt.pth'
    target_model_path = 'model/svhn2mnist_ckpt.pth'
    ##################################################
    device = get_device(num_device=0)
    #-----------------------
    opt_Lu = {0: 'L1', 1: 'L2'}
    #-----------------------
    domain[1] = {0: 'mnist', 1: 'svhn'}
    exp = 1   #{0: 'Digit-0', 1: 'Digit-1', 2: 'Office-31', 3: 'Office-Home', 4: 'VisDA', 5: 'Sign'}
    #-----------------------
    source = 1
    target = 0
    #-----------------------
    arg['c_s'] = 1.0
    arg['c_u'] = 1.0
    arg['optLu'] = 1  # {0: 'L1', 1: 'L2'}
    #-----------------------
    arg['test_period'] = 30
    arg['alpha'] = 0.5 # alpha in exponential moving average
    arg['save'] = False
    #-----------------------
    n_epoch = 50
    n_bs_tr = 256
    n_bs_te = 256
    n_class = 10
    #-----------------------
    arg['fix'] = True
    arg['iLR'] = 2e-4
    arg['fLR'] = 2e-4
    #-----------------------
    arg['sam'] = 10 # number of Monte-carlo dropout sampling
    #-----------------------
    number_of = {0: 60000, 1: 73257}
    n_iteration = int( (number_of[target]/n_bs_tr) * n_epoch )
    #-----------------------
    G = MidG().to(device)
    H = MidH().to(device)

    G.load_state_dict(torch.load(source_model_path+'_g.pth', map_location=lambda storage, loc:storage))
    H.load_state_dict(torch.load(source_model_path+'_h.pth', map_location=lambda storage, loc:storage))

    # ==========================================================================================================
    arg['n_cls'] = n_class
    arg['n_batch'] = n_bs_tr
    arg['n_iter'] = n_iteration

    # set optimizers & schedulers
    arg['op_g'] = optim.Adam(G.parameters(), lr=arg['iLR'], weight_decay=5e-4)
    arg['op_h'] = optim.Adam(H.parameters(), lr=arg['iLR'], weight_decay=5e-4)

    arg['scheduler_g'] = optim.lr_scheduler.CosineAnnealingLR(arg['op_g'], T_max=arg['n_iter'], eta_min=arg['fLR'])
    arg['scheduler_h'] = optim.lr_scheduler.CosineAnnealingLR(arg['op_h'], T_max=arg['n_iter'], eta_min=arg['fLR'])

    arg['criterion'] = nn.CrossEntropyLoss()


    # ==========================================================================================================
    ## set data loaders
    dset['train_t'] = datasets.MNIST(path['mnist'], True, transform_w1, download=False)
    dset['test_t'] = datasets.MNIST(path['mnist'], False, transform_n1, download=False)
    dset['train_s'] = datasets.SVHN(path['svhn'], 'train', transform_w2, download=False)
    dset['test_s'] = datasets.SVHN(path['svhn'], 'test', transform_n2, download=False)
    #-----------------------
    loader['train_t'] = DataLoader(dset['train_t'], n_bs_tr, shuffle=True, num_workers=4)
    loader['test_t'] = DataLoader(dset['test_t'], n_bs_te, shuffle=False, num_workers=4)
    loader['train_s'] = DataLoader(dset['train_s'], n_bs_tr, shuffle=True, num_workers=4)
    loader['test_s'] = DataLoader(dset['test_s'], n_bs_te, shuffle=False, num_workers=4)


    # ==========================================================================================================
    ## check initial performance & run

    accu_s = test(device, G, H, loader['test_s'])
    accu_t = test(device, G, H, loader['test_t'])
    print('\nInitial accuracy : source = {:.4f}, target = {:.4f}'.format(accu_s, accu_t))
    print('source model path = {}\n'.format(source_model_path))

    acc_s, acc_t, max_t, max_it = [], [], [], []

    for arg['rnd'] in [2022, 2023, 2024]:
        init_random_seed(arg['rnd'], printing=True)

        testloaders = [loader['test_s'], loader['test_t']]
        trainloaders = [loader['train_s'], loader['train_t']]

        G.load_state_dict(torch.load(source_model_path+'_g.pth', map_location=lambda storage, loc:storage))
        H.load_state_dict(torch.load(source_model_path+'_h.pth', map_location=lambda storage, loc:storage))
        accu = muda(device, G, H, trainloaders, testloaders, target_model_path, arg)

        acc_s.append(accu[0][-1])
        acc_t.append(accu[1][-1])
        max_t.append(np.max(accu[1]))
        max_it.append(np.argmax(accu[1])+1)

        accu_s = test(device, G, H, loader['test_s'])
        accu_t = test(device, G, H, loader['test_t'])
        print('Adapted accuracy : source = {:.4f}, target = {:.4f}\n'.format(accu_s, accu_t))

    print("-------------------------------------------------------------------------------------------------")
    print("[S] {:.4f} +- {:.4f} [T] {:.4f} +- {:.4f} (max) {:.4f} @ iteration {:.0f}".format(
        np.mean(acc_s), np.std(acc_s), np.mean(acc_t), np.std(acc_t), np.mean(max_t), np.mean(max_it)))
    print("-------------------------------------------------------------------------------------------------")



if __name__ == "__main__":
    main()
