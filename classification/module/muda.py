import torch
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm
from module.loss import *
from module.util import *

def muda(device, G, H, trainloader, testloader, save_path, param):
    """
    - trainloader: [0] train_s, [1] train_t
    - testloader: [0] test_s, [1] test_t
    """
    trloader = {'S': trainloader[0], 'T': trainloader[1]}
    loader = {'S': testloader[0], 'T': testloader[1]}
    tr_s_batch = iter(trloader['S'])
    tr_t_batch = iter(trloader['T'])

    accu_s_list, accu_t_list = [], []
    best_accu, best_idx = 0., 0

    accu_s_ = AverageMeter(alpha=param['alpha'])
    accu_t_ = AverageMeter(alpha=param['alpha'])
    loss_s_ = AverageMeter(alpha=param['alpha'])
    loss_u_ = AverageMeter(alpha=param['alpha'])


    LR = param['iLR']
    pbar = tqdm(range(param['n_iter']))

    since = time.time()
    best_idx = 0
    for idx in pbar:
        if param['fix'] == False:
            for par in param['op_g'].param_groups:
                LR = par['lr']

        try:
            data_s, label_s = tr_s_batch.next()
        except:
            tr_s_batch = iter(trloader['S'])
            data_s, label_s = tr_s_batch.next()

        try:
            data_t, label_t = tr_t_batch.next() # label_t is not used during UDA
        except:
            tr_t_batch = iter(trloader['T'])
            data_t, label_t = tr_t_batch.next()

        if len(label_s) < param['n_batch'] or len(label_t) < param['n_batch']:
            continue

        data_s, label_s = data_s.to(device), label_s.to(device)
        data_t, label_t = data_t.to(device), label_t.to(device)

        G.train()
        H.train()

        #-------------------------------------------
        param['op_g'].zero_grad()
        param['op_h'].zero_grad()

        ft_s = G(data_s, dropout=False)
        logit_s = H(ft_s, dropout=False)
        loss_s = param['criterion'](logit_s, label_s)

        loss_s.backward()
        param['op_h'].step()
        param['op_g'].step()

        loss_s_.update(loss_s.item())

        #-------------------------------------------
        param['op_g'].zero_grad()
        param['op_h'].zero_grad()

        ft_s = G(data_s, dropout=False)
        logit_s = H(ft_s, dropout=False)
        loss_s = param['criterion'](logit_s, label_s)

        loss_s = param['c_s'] * loss_s
        loss_s.backward(retain_graph=True)
        param['op_h'].step()

        #-------------------------------------------
        pred_list = []
        for _ in range(param['sam']):
            ft_t = G(data_t, dropout=True)
            pred_list.append(torch.unsqueeze(F.softmax(H(ft_t, dropout=True), dim=1), dim=0))

        pred_t = torch.cat(pred_list, dim=0).mean(dim=0)
        pred_t_std = torch.cat(pred_list, dim=0).std(dim=0)

        if param['optLu'] == 0:
            loss_u = UncertaintyLoss(pred_t_std, param['n_batch'])
        else:  # 1
            loss_u = Uncertainty2Loss(pred_t_std)

        loss_u = param['c_u'] * loss_u
        loss_u.backward()
        param['op_g'].step()

        loss_u_.update(loss_u.item())

        #-------------------------------------------
        ### accumulate test result of each iteration
        if idx % param['test_period'] == 0:
            accu_s = test(device, G, H, loader['S'])
            accu_t = test(device, G, H, loader['T'])

        accu_s_.update(accu_s)
        accu_t_.update(accu_t)

        accu_s_list.append(accu_s_.avg)
        accu_t_list.append(accu_t_.avg)

        post_str = "[S]{:.4f} [T]{:.4f} (*{:.4f} @{}), loss=[Ls]{:.4f} [Lu]{:.4f} (LR={:.6f})".format(
            accu_s_.avg, accu_t_.avg, best_accu, best_idx+1, loss_s_.avg, loss_u_.avg, LR)

        if accu_t_.avg > best_accu:
            if param['save'] == True:
                torch.save(G.state_dict(), save_path+'_g.pth')
                torch.save(H.state_dict(), save_path+'_h.pth')
            best_accu = accu_t_.avg
            best_idx = idx
            post_str += '!!!'

        pbar.set_postfix(accu=post_str)

        if param['fix'] == False:
            param['scheduler_g'].step()
            param['scheduler_h'].step()

    if param['save'] == True:
        torch.save(G.state_dict(), save_path+'_g_last.pth')
        torch.save(H.state_dict(), save_path+'_h_last.pth')

    train_time = time.time() - since
    print('Training completed ({:.0f}m {:.0f}s) Max target accuracy = {:.4f} @{}'.format(
        train_time//60, train_time%60, best_accu, best_idx+1))

    return accu_s_list, accu_t_list
