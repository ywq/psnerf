import torch
import numpy as np
import sys
import torch.nn.functional as F
from torch.nn import init


def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def MAE(vec1, vec2, mask=None, normalize=True):
    '''
    Input : N x 3  or  H x W x 3 .   [-1,1]
    Output : MAE, AE
    '''
    vec1, vec2 = vec1.copy().astype(np.float32), vec2.copy().astype(np.float32)
    if normalize:
        norm1 = np.linalg.norm(vec1, axis=-1)
        norm2 = np.linalg.norm(vec2, axis=-1)
        vec1 /= norm1[...,None] + 1e-5
        vec2 /= norm2[...,None] + 1e-5
        vec1[norm1==0] = 0
        vec2[norm2==0] = 0
    dot_product = (vec1 * vec2).sum(-1).clip(-1, 1)
    if mask is not None:
        dot_product = dot_product[mask.astype(bool)]
    angular_err = np.arccos(dot_product) * 180.0 / np.pi
    l_err_mean  = angular_err.mean()
    return l_err_mean, angular_err
