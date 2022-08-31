import torch
from torch import nn
from torch.nn import functional as F


class MainLoss(nn.Module):
    def __init__(self, sg_rgb_weight, loss_type='L1', 
                        albedo_smooth_weight=0,rough_smooth_weight=0,
                        vis_weight=1.0):
        super().__init__()
        self.sg_rgb_weight = sg_rgb_weight
        self.albedo_smooth_weight = albedo_smooth_weight
        self.rough_smooth_weight = rough_smooth_weight
        self.vis_weight = vis_weight

        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!')

        self.smooth_loss = nn.L1Loss(reduction='mean')

    def get_rgb_loss(self, sg_rgb_values, rgb_gt, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        mask = mask.expand(rgb_gt.shape[0],-1)
        sg_rgb_values = sg_rgb_values[mask].reshape((-1, 3))
        rgb_gt = rgb_gt[mask].reshape((-1, 3))

        sg_rgb_loss = self.img_loss(sg_rgb_values, rgb_gt)

        return sg_rgb_loss

    def get_vis_loss(self, vis_values, vis_gt, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        mask = mask.expand(vis_gt.shape[0],-1)
        vis_values = vis_values[mask].reshape((-1, ))
        vis_gt = vis_gt[mask].reshape((-1, ))

        vis_loss = self.img_loss(vis_values, vis_gt)
        return vis_loss

    def get_smooth_loss(self, x, x_jitter, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0.:
            return torch.tensor(0.0).cuda().float()

        mask = mask.expand(x.shape[0],-1)
        return self.smooth_loss(x[mask],x_jitter[mask])

    def forward(self, model_outputs, ground_truth, model_input=None):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        sg_rgb_loss = self.get_rgb_loss(model_outputs['sg_rgb_values'], rgb_gt, network_object_mask, object_mask)
        loss = self.sg_rgb_weight * sg_rgb_loss
        
        albedo_loss, rough_loss = None, None
        if 'albedo_jitter' in [*model_outputs] and self.albedo_smooth_weight>0:
            albedo_loss = self.get_smooth_loss(model_outputs['albedo_values'], model_outputs['albedo_jitter'], network_object_mask, object_mask)
            loss += self.albedo_smooth_weight * albedo_loss 
        if 'rough_jitter' in [*model_outputs] and self.rough_smooth_weight>0:
            rough_loss = self.get_smooth_loss(model_outputs['rough_values'], model_outputs['rough_jitter'], network_object_mask, object_mask)
            loss += self.rough_smooth_weight * rough_loss
            
        lterm =  {
            'sg_rgb_loss': sg_rgb_loss,
            'albedo_smooth_loss': albedo_loss,
            'rough_smooth_loss': rough_loss,
        }
        if 'visibility' in [*model_input] and 'visibility' in [*model_outputs]:
            if 'vis_train_gt' in model_input and 'light_vis_train' in model_input and 'vis_train' in model_outputs:
                vis_loss = self.get_vis_loss(model_outputs['vis_train'][...,0], model_input['vis_train_gt'], network_object_mask, object_mask)
            elif 'light_vis_train' in model_input and 'vis_train' in model_outputs:
                vis_loss = self.get_vis_loss(model_outputs['vis_train'][...,0], model_input['visibility'], network_object_mask, object_mask)
            else:
                vis_loss = self.get_vis_loss(model_outputs['visibility'][...,0], model_input['visibility'], network_object_mask, object_mask)
            loss += self.vis_weight * vis_loss
            lterm['vis_loss'] = vis_loss
        lterm['loss'] = loss
        
        return lterm



class NormalLoss(nn.Module):
    def __init__(self, normal_weight, normal_smooth_weight=0):
        super().__init__()
        self.normal_weight = normal_weight
        self.normal_smooth_weight = normal_smooth_weight

        self.normal_loss = nn.MSELoss(reduction='mean')
        self.smooth_loss = nn.L1Loss(reduction='mean')

    def get_normal_loss(self, normal_values, normal_gt, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        # mask = mask.squeeze()
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        normal_values = normal_values[mask].reshape((-1, 3))
        normal_gt = normal_gt[mask].reshape((-1, 3))
        norm_loss = self.normal_loss(normal_values, normal_gt)

        return norm_loss

    def get_smooth_loss(self, x, x_jitter, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0.:
            return torch.tensor(0.0).cuda().float()
        return self.smooth_loss(x[mask],x_jitter[mask])

    def forward(self, model_outputs):
        normal_gt = F.normalize(model_outputs['normal_values'],dim=-1)
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        norm_loss = self.get_normal_loss(model_outputs['normal_pred'], normal_gt, network_object_mask, object_mask)
        loss = self.normal_weight * norm_loss
        
        normal_smooth_loss = None
        if 'normal_jitter' in model_outputs and self.normal_smooth_weight>0:
            normal_smooth_loss = self.get_smooth_loss(model_outputs['normal_pred'], model_outputs['normal_jitter'], network_object_mask, object_mask)
            loss += self.normal_smooth_weight * normal_smooth_loss 
            

        return {
            'loss': loss,
            'normal_loss': norm_loss,
            'normal_smooth_loss': normal_smooth_loss,
        }
