import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGBasis(nn.Module):
    def __init__(self, nbasis=9, specular_rgb=False):
        super().__init__()
        self.nbasis = nbasis
        self.specular_rgb = specular_rgb
        self.lobe = nn.Parameter(torch.tensor([np.exp(i) for i in range(2,11)],dtype=torch.float32))
        self.lobe.requires_grad_(False)


    def forward(self, v, n, l, albedo, weights):
        '''
        :param  v: [N, 3]
        :param  n: [N, 3]
        :param  l: [N, 3]
        :param  albedo: [N, 3]
        :param  weights: [N, nbasis]
        '''
        h = F.normalize(l + v, dim=-1)   # [N,3]
        D = torch.exp(self.lobe[None,].clamp(min=0) * ((h*n).sum(-1, keepdim=True) - 1))  # [N, 9]
        if self.specular_rgb:
            specular = (weights.view(-1,3,self.nbasis) * D[:,None]).sum(-1).clamp(min=0.)  # [N, 3]
        else:
            specular = (weights * D).sum(-1, keepdim=True).clamp(min=0.)  # [N, 1]
        lambert = albedo
        brdf = lambert + specular.expand_as(lambert)
        return brdf, specular