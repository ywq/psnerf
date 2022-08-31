import os
import torch
from .common import (
    get_tensor_values, sample_patch_points, arange_pixels
)
import logging
from .losses import Loss
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')
from utils.tools import MAE

to_img = lambda x: (x.astype(np.float32).clip(0,1) * 255).astype(np.uint8)
to_numpy = lambda x: x.detach().cpu().numpy()
to_hw = lambda x, h, w: x.reshape(w,h,-1).permute(1,0,2)

class Trainer(object):
    def __init__(self, model, optimizer, cfg_all, device=None, **kwargs):
        cfg = cfg_all['training']
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_training_points = cfg['n_training_points']
        self.n_eval_points = cfg['n_training_points']
        self.overwrite_visualization = True
        self.normal_loss = cfg.get('normal_loss', False)
        self.normal_after = cfg.get('normal_after', -1)
        self.cfg = cfg
        self.angle = self.cfg.get('normal_angle',None)
        self.normal_weight_decay = self.cfg.get('normal_weight_decay',False)
        self.mask_loss = self.cfg.get('mask_loss',False)
        self.mask_loss_type = self.cfg.get('mask_loss_type','acc')
        self.rendering_technique = cfg['type']

        self.loss = Loss(
            cfg['lambda_l1_rgb'], 
            cfg['lambda_normals'],
            cfg.get('lambda_normloss', 1.0),
            cfg.get('lambda_mask', 1.0),
            device=device,
        )

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.model.train()
        self.optimizer.zero_grad()

        loss_dict = self.compute_loss(data, it=it)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        return loss_dict
    
    def render_visdata(self, data_loader, it, out_render_path):
        self.model.eval()
        save_img = []
        for di, data in enumerate(data_loader):
            if di>=2: break
            (img, mask, world_mat, camera_mat, scale_mat, img_idx, normal, norm_mask, mask_valid) = \
                self.process_data_dict(data)

            h, w = img.shape[-2:] #resolution
            p_loc, pixels = arange_pixels(resolution=(h, w))
            pixels = pixels.to(self.device)
            ploc = p_loc.to(self.device)

            img_iter = [to_img(to_numpy(img))[0].transpose(1,2,0)]

            with torch.no_grad():
                rgb_pred, norm_pred,mask_pred, mask_acc = [], [],[],[]
                for ii, pixels_i in enumerate(torch.split(ploc, 1024, dim=1)):
                    out_dict = self.model(
                                    pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                                    add_noise=False, eval_=True, it=it,)
                    rgb_pred.append(out_dict['rgb'])
                    norm_pred.append(out_dict['normal_pred'])
                    mask_pred.append(out_dict['mask_pred'])
                    mask_acc.append(out_dict['acc_map'])
                rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h,w))
                img_iter.append(to_img(rgb_pred))
                mask_pred = to_numpy(to_hw(torch.cat(mask_pred, dim=0),h,w)).repeat(3,axis=-1)
                # normal
                norm_pred = to_numpy(to_hw(torch.cat(norm_pred, dim=1),h,w))
                norm_pred = np.einsum('ij,hwi->hwj', to_numpy(world_mat)[0,:3,:3]*np.array([[1,-1,-1]]),norm_pred)
                img_iter.append(to_img(norm_pred/2.+0.5))
                if normal is not None:
                    img_iter.append(to_img(to_numpy(normal[0].permute(1,2,0))/2.+0.5))
                    error = MAE(norm_pred,to_numpy(normal[0].permute(1,2,0)).clip(-1,1))[1]/45
                    img_iter.append(to_img(cm(error.clip(0,1)*(to_numpy(mask.bool())[0,0]|mask_pred[...,0]))[...,:3]))
                # mask
                img_iter.append(to_img(mask_pred))
                mask_acc = to_numpy(to_hw(torch.cat(mask_acc, dim=1),h,w)).repeat(3,axis=-1)
                img_iter.append(to_img(mask_acc))

            with torch.no_grad():
                rgb_pred = \
                    [self.model(
                        pixels_i, camera_mat, world_mat, scale_mat, 'phong_renderer', 
                        add_noise=False, eval_=True, it=it)['rgb']
                        for ii, pixels_i in enumerate(torch.split(ploc, 1024, dim=1))]
           
                rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h,w))
                img_iter.append(to_img(rgb_pred)) 
            
            save_img.append(np.concatenate(img_iter, axis=-2))
        save_img = np.concatenate(save_img, axis=0)
        save_img = Image.fromarray(save_img.astype(np.uint8)).convert("RGB")
        save_img.save(out_render_path)
        self.model.train()
        return 

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device
       
        img = data.get('img').to(device)
        img_idx = data.get('img.idx')
        batch_size, _, h, w = img.shape
        mask_img = data.get('img.mask', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        normal = data.get('img.normal').to(device) if self.normal_loss else None
        norm_mask = data.get('img.norm_mask').unsqueeze(1).to(device) if self.normal_loss else None
        mask_valid = data.get('img.mask_valid', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)

        return (img, mask_img, world_mat, camera_mat, scale_mat, img_idx, normal, norm_mask, mask_valid)

    def compute_loss(self, data, eval_mode=False, it=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        n_points = self.n_eval_points if eval_mode else self.n_training_points
        (img, mask_img, world_mat, camera_mat, scale_mat, img_idx, normal, norm_mask, mask_valid) = self.process_data_dict(data)

        # Shortcuts
        device = self.device
        batch_size, _, h, w = img.shape

        # Assertions
        assert(((h, w) == mask_img.shape[2:4]) and (n_points > 0))

        # Sample pixels
        if n_points >= h*w:
            p = arange_pixels((h, w), batch_size)[0].to(device)
            mask_gt = mask_img.bool().reshape(batch_size,-1).to(torch.float32)
            mask_valid = mask_valid.bool().reshape(batch_size,-1)
            norm_mask_gt = norm_mask.bool() if self.normal_loss else None
            # pix = None
            pix = p
        else:
            p, pix = sample_patch_points(batch_size, n_points,
                                    patch_size=1.,
                                    image_resolution=(h, w),
                                    continuous=False,
                                    )
            p = pix.to(device) 
            pix = pix.to(device) 
            mask_gt = get_tensor_values(mask_img, pix.clone()).bool().reshape(batch_size,-1).to(torch.float32)
            mask_valid = get_tensor_values(mask_valid*1., pix.clone()).bool().reshape(batch_size,-1)
            norm_mask_gt = get_tensor_values(norm_mask, pix.clone()).bool().squeeze(-1) if self.normal_loss else None

        out_dict = self.model(
            p, camera_mat, world_mat, scale_mat, 
            self.rendering_technique, it=it, 
            eval_=eval_mode
        )
        
        rgb_gt = get_tensor_values(img, pix.clone())
        normal_gt = None
        if self.normal_loss and it>=self.normal_after:
            normal_gt = get_tensor_values(normal, pix.clone())
            if self.angle is not None:
                norm_mask_gt[normal_gt[...,-1]<np.cos(np.deg2rad(self.angle))] = False
            normal_gt = torch.einsum('bij,bnj->bni', world_mat[:,:3,:3]*torch.tensor([[[1,-1,-1]]],dtype=torch.float32).to(device), normal_gt)

        mask_pred = out_dict.get('acc_map',None)
        if not self.mask_loss:
            mask_gt = None

        loss_dict = self.loss(out_dict, rgb_gt, normal_gt, norm_mask_gt, mask_pred, mask_gt, mask_valid)
        return loss_dict
