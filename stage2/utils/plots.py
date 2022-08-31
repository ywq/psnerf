import numpy as np
import plotly.graph_objs as go
import torch
import torchvision
import trimesh
from PIL import Image
from skimage import measure

from utils import rend_util
import os
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')


def plot_micro(gamma, model_out_all ,rgb_gt_all, path, epoch, img_res, model_input_all=None):
    path_img = path
    tonemap_img = lambda x: torch.pow(x, 1./gamma)
    clip_img = lambda x: torch.clamp(x, min=0., max=1.)
    # arrange data to plot
    plot_all = []
    for model_outputs, rgb_gt, model_input in zip(model_out_all, rgb_gt_all, model_input_all) :
        batch_size, num_samples, _ = rgb_gt.shape
        rgb_gt = clip_img(tonemap_img(rgb_gt.cuda()))
        network_object_mask = model_outputs['network_object_mask'].reshape(batch_size, num_samples, 1)
        rgb_gt[~network_object_mask[...,0]] = 1
        sg_rgb_eval = model_outputs['sg_rgb_values'].reshape(batch_size, num_samples, 3)
        sg_rgb_eval = clip_img(tonemap_img(sg_rgb_eval))
        normal = model_outputs['normal_values'].reshape(batch_size, num_samples, 3)
        normal = clip_img((normal + 1.) / 2.)

        diffuse_albedo = model_outputs['sg_diffuse_albedo_values'].reshape(batch_size, num_samples, 3)
        diffuse_albedo /= diffuse_albedo[diffuse_albedo!=1].max()   ## scale
        diffuse_albedo = clip_img(diffuse_albedo)
        rough = model_outputs['sg_specular_rgb_values'].reshape(batch_size, num_samples, 3)
        rough = clip_img(rough)

        if 'normal_pred' in [*model_outputs]:
            normal_pred = model_outputs['normal_pred'].reshape(batch_size, num_samples, 3)
            normal_pred = clip_img((normal_pred + 1.) / 2.)
            normal = torch.cat([normal, normal_pred],0)

        visibility = torch.ones_like(rgb_gt)
        vis_buffer = torch.ones_like(rgb_gt)
        if 'visibility' in [*model_outputs]:
            visibility = model_outputs['visibility'].reshape(batch_size, num_samples, 3)
            if 'visibility' in [*model_input]:
                vis_buffer = model_input['visibility'].reshape(batch_size, num_samples, 1).repeat(1,1,3)

        output_vs_gt = torch.cat((normal, diffuse_albedo, rough, 
                                clip_img(vis_buffer), clip_img(visibility), sg_rgb_eval, rgb_gt, 
                                model_outputs['object_mask'].reshape(batch_size, num_samples)[...,None].expand(rgb_gt.shape),
                                network_object_mask.expand(rgb_gt.shape),
                                ), dim=0)
        plot_all.append(output_vs_gt)

        if 'sg_weight' in [*model_outputs]:
            sg_weight = model_outputs['sg_weight'].view(num_samples,3,9).permute(2,0,1).clamp(0,1)
            if 'normal_pred' in [*model_outputs]:
                normal_mae = torch.from_numpy(cm(model_outputs['normal_mae'].cpu().numpy()/180.)[...,:3]).cuda().view(1,num_samples,3).clamp(0,1)
                sg_weight = torch.cat([normal_mae,sg_weight],dim=0)
            plot_all.append(sg_weight)
    plot_all = torch.stack(plot_all, dim=1).view(-1,*output_vs_gt.shape[1:])
    output_vs_gt_plot = lin2img(plot_all, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=len(model_out_all)*2 if 'sg_weight' in [*model_outputs] else len(model_out_all)).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path_img, epoch))

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
