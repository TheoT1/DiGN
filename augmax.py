# AugMax model construction.
# from https://github.com/VITA-Group/AugMax
#
# Last updated: May 6 2022

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

class AugMaxModule(nn.Module):
    def __init__(self, device='cuda'):
        super(AugMaxModule, self).__init__()
        self.device = device

    def forward(self, xs, m, q):
        '''
        Inputs:
            xs: tuple of Tensors. len(x)=3. xs = (x_ori, x_aug1, x_aug2, x_aug3). x_ori.size()=(N,C,W,H)
            m: Tensor. m.size=(N)
            q: Tensor. q.size()=(N,3). w = softmax(q)
        Outputs:
            x_mix: Tensor. x_mix.size()=(N,C,W,H)
        '''

        x_ori = xs[0]
        w = torch.nn.functional.softmax(q, dim=1) # w.size()=(N,3)

        N = x_ori.size()[0]

        x_mix = torch.zeros_like(x_ori).to(self.device)
        for i, x_aug in enumerate(xs[1:]):
            wi = w[:,i].view((N,1,1,1)).expand_as(x_aug)
            x_mix += wi * x_aug 

        m = m.view((N,1,1,1)).expand_as(x_ori)
        x_mix = (1-m) * x_ori + m * x_mix

        return x_mix