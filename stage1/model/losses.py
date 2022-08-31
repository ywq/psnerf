import torch
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, full_weight, grad_weight, norm_weight=1.0, mask_weight=1.0, device=None):
        super().__init__()
        self.full_weight = full_weight
        self.grad_weight = grad_weight
        self.norm_weight = norm_weight
        self.mask_weight = mask_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.mask_loss = nn.BCELoss(reduction='mean')
        self.device = device
    
    def get_rgb_full_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])
        return rgb_loss

    def get_smooth_loss(self, diff_norm):
        if diff_norm is None or diff_norm.shape[0]==0:
            return torch.tensor(0.0).to(self.device).float()
        else:
            return diff_norm.mean()
    
    def get_mask_loss(self, mask, mask_gt):
        return self.mask_loss(mask, mask_gt)

    def forward(self, out_dict, rgb_gt, normal_gt=None, norm_mask=None, mask=None, mask_gt=None, mask_valid=None):
        rgb_gt = rgb_gt.to(self.device)
        rgb_pred = out_dict['rgb']
        diff_norm = out_dict['diff_norm']
        normal = out_dict.get('normal_pred',None)
        
        if self.full_weight != 0.0:
            rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt)
        else:
            rgb_full_loss = torch.tensor(0.0).to(self.device).float()

        if diff_norm is not None and self.grad_weight != 0.0:
            grad_loss = self.get_smooth_loss(diff_norm)
        else:
            grad_loss = torch.tensor(0.0).to(self.device).float()
        
        loss = self.full_weight * rgb_full_loss + \
               self.grad_weight * grad_loss
            
        loss_term = {
            'fullrgb_loss': rgb_full_loss,
            'grad_loss': grad_loss,
        }
        if normal is not None and normal_gt is not None:
            if norm_mask.sum()>0:
                normal_loss = self.l1_loss(normal[norm_mask], normal_gt[norm_mask]) / float(normal[norm_mask].shape[0])
                loss += self.norm_weight * normal_loss
                loss_term['normal_loss'] = normal_loss

        # add mask loss
        if mask is not None and mask_gt is not None:
            lmask = self.mask_loss(mask[mask_valid].clamp(0,1), mask_gt[mask_valid])
            loss += self.mask_weight * lmask
            loss_term['mask_loss'] = lmask

        loss_term['loss'] = loss

        if torch.isnan(loss):
            breakpoint()

        return loss_term


