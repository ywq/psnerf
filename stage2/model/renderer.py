import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import get_embedder
from model.microfacet import Microfacet
from model.sgbasis import SGBasis

act_fns = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'softplus': F.softplus,
}

class Normal_Network(nn.Module):
    def __init__(self, din, dout, W, depth, skip_at=[]):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(din, W)] 
                                    + [nn.Linear(W+din if i in skip_at else W, W) for i in range(depth-1)]
                                    + [nn.Linear(W, dout)] )
        self.skip_at = skip_at

    def forward(self, x):
        y = x
        for li, lyr in enumerate(self.linears):
            y = lyr(y)
            y = F.relu(y) if li!=len(self.linears)-1 else y
            if li in self.skip_at:
                y = torch.cat([y,x],-1)
        return y

class Network(nn.Module):
    def __init__(self, din, dout, W, depth, skip_at=[]):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(din, W)] 
                                    + [nn.Linear(W+din if i in skip_at else W, W) for i in range(depth-1)]
                                    + [nn.Linear(W, dout)] )
        self.skip_at = skip_at

    def forward(self, x):
        y = x
        for li, lyr in enumerate(self.linears):
            y = lyr(y)
            y = F.relu(y) if li!=len(self.linears)-1 else torch.sigmoid(y)
            if li in self.skip_at:
                y = torch.cat([y,x],-1)
        return y


class PSNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        ''' known:      pts2l: LxNx3
                        pts2c: Nx3
                        normal: Nx3
            unknown:    albedo: Nx3
                        rough: NxS
        '''
        self.conf = conf
        self.render_model = conf.get_string('train.render_model', default='sgbasis')
        if self.render_model == 'microfacet':
            self.microfacet = Microfacet(f0 = conf.get_float('brdf.fresnel_f0', default = 0.05))
        elif self.render_model == 'sgbasis':
            nbasis = conf.get_int('train.nbasis', default=9)
            self.specular_rgb=conf.get_bool('train.specular_rgb', default=False)
            self.sgbasis = SGBasis(nbasis=nbasis, specular_rgb=self.specular_rgb)

        self.embedder, dim_emb = get_embedder(conf.get_int('brdf.net.n_freqs_xyz'))
        W = conf.get_int('brdf.net.mlp_width')
        depth = conf.get_int('brdf.net.mlp_depth')
        skip_at = conf.get_int('brdf.net.mlp_skip_at')
        self.albedo_net = Network(dim_emb,3,W,depth,skip_at=[skip_at])
        if self.render_model == 'microfacet':
            self.rough_net = Network(dim_emb,1,W,depth,skip_at=[skip_at])
        elif self.render_model in ['sgbasis']:
            if self.specular_rgb: 
                nbasis *= 3
            self.rough_net = Normal_Network(dim_emb,nbasis,conf.get_int('brdf.sgnet.mlp_width', 128),
                                                conf.get_int('brdf.sgnet.mlp_depth',4),
                                                skip_at=[conf.get_int('brdf.sgnet.mlp_skip_at',2)])
            self.nbasis = nbasis
        self.light_int = conf.get_float('brdf.light_intensity', default=4.0)

        self.shape_pregen = conf.get_bool('train.shape_pregen',default=False)
        self.xyz_jitter_std = conf.get_float('brdf.net.xyz_jitter_std', default=0)

        self.normal_mlp = conf.get_bool('train.normal_mlp', default = False)
        if self.normal_mlp:
            self.embedder_n, dim_emb_n = get_embedder(conf.get_int('normal.net.n_freqs_xyz'))
            W_n = conf.get_int('normal.net.mlp_width')
            depth_n = conf.get_int('normal.net.mlp_depth')
            skip_at_n = conf.get_int('normal.net.mlp_skip_at')
            self.normal_net = Normal_Network(dim_emb_n,3,W_n,depth_n,skip_at=[skip_at_n])
            self.normal_joint = conf.get_bool('train.normal_joint', default=False)
            self.normal_jitter_std = conf.get_float('normal.net.xyz_jitter_std', default=0)
            if not self.normal_joint:
                self.normal_net = self.normal_net.eval().requires_grad_(False)
                self.normal_jitter_std = 0

        self.visibility = conf.get_bool('train.visibility', default=False)
        self.light_vis_detach = conf.get_bool('train.light_vis_detach', default=False)
        if self.visibility:
            self.visibility_net = Normal_Network(dim_emb*2,1,
                                                    conf.get_int('visibility.net.mlp_width'),
                                                    conf.get_int('visibility.net.mlp_depth'),
                                                    skip_at=[conf.get_int('visibility.net.mlp_skip_at')])

    def forward(self, input, albedo_new=None, basis_new=None):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"]
        device = uv.device

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        if self.shape_pregen:
            surface_mask = input['surface_mask']
            points = input['points']
            normals = input['normal']
            surface_points = points[surface_mask]
 
        if self.normal_mlp:
            normal_pred = torch.ones_like(points).float().to(device)
            if surface_points.shape[0] > 0:
                normal_out = self.normal_net(self.embedder_n(surface_points))
                normal_out = F.normalize(normal_out, dim=-1)
                normal_pred[surface_mask] = normal_out

                if self.normal_jitter_std > 0:
                    xyz_jitter = surface_points + torch.normal(0,torch.ones_like(surface_points)*self.normal_jitter_std)
                    normal_out_jitter = self.normal_net(self.embedder_n(xyz_jitter))
                    normal_out_jitter = F.normalize(normal_out_jitter, dim=-1)
                    normal_jitter = torch.ones_like(points).float().to(device)
                    normal_jitter[surface_mask] = normal_out_jitter
                    jitter_n = {'normal_jitter': normal_jitter}  
            normal_output = {'normal_pred': normal_pred}
            if self.normal_jitter_std > 0 and surface_points.shape[0] > 0:
                normal_output.update(jitter_n)       

        rgb_values = torch.ones_like(points).float().to(device)
        albedo_values = torch.ones_like(points).float().to(device)
        if self.render_model == 'microfacet':
            rough_values = torch.ones_like(points).float().to(device)
        elif self.render_model in ['sgbasis']: 
            rough_values = torch.ones_like(points).float().to(device)
            weight_values = torch.zeros(*points.shape[:-1],self.nbasis).float().to(device)
        vis_values = torch.ones_like(points).float().to(device)
        # multi-light rendering
        lnum = input['light_direction'].shape[0]
        if lnum>1:
            rgb_values = rgb_values.repeat(lnum,1,1)
            if self.render_model in ['sgbasis']:
                rough_values = rough_values.repeat(lnum,1,1)
            vis_values = vis_values.repeat(lnum,1,1)
        if surface_points.shape[0] > 0:
            normal = normals[surface_mask] if not self.normal_mlp else normal_pred[surface_mask]
            pts2c = - ray_dirs[surface_mask]   ### camera z axis points to surface
            pts2l = input['light_direction'][:,None].to(device).expand(rgb_values.shape)[surface_mask.expand(lnum,-1)]

            point_emb = self.embedder(surface_points)
            albedo = self.albedo_net(point_emb)
            if albedo_new is not None:
                albedo = torch.from_numpy(albedo_new).to(albedo.device)[None,].expand_as(albedo)
            rough = self.rough_net(point_emb)

            if self.render_model == 'microfacet':
                brdf = self.microfacet(pts2l.view(lnum,-1,3).permute(1,0,2), pts2c, normal, albedo=albedo, rough=rough).permute(1,0,2).reshape(-1,3)
            elif self.render_model in ['sgbasis']:
                weights = F.relu(rough)
                if basis_new is not None:
                    weight_new = torch.zeros_like(weights)
                    if self.specular_rgb:
                        weight_new.view(-1,3, self.nbasis//3)[:,:,basis_new]=2**basis_new/100
                    else:
                        weight_new.view(-1,1, self.nbasis)[:,:,basis_new]=2**basis_new/100
                    weights = weight_new.reshape(-1,self.nbasis)
                if lnum>1:
                    brdf, rough = self.sgbasis(l=pts2l, v=pts2c.tile(lnum,1), n=normal.tile(lnum,1),albedo=albedo.tile(lnum,1), weights=weights.tile(lnum,1))
                else:
                    brdf, rough = self.sgbasis(l=pts2l, v=pts2c, n=normal,albedo=albedo, weights=weights)
                weight_values[surface_mask] = weights
            cos = torch.einsum('lni,ni->ln', pts2l.view(lnum,-1,3), normal).reshape(-1,1)
            light_int = input.get('light_intensity', self.light_int)
            if torch.is_tensor(light_int) and light_int.shape[0]>1:
                light_int = light_int.repeat_interleave(surface_points.shape[0],dim=0)
            if self.visibility:
                if self.light_vis_detach:
                    vis = self.visibility_net(torch.cat([point_emb.tile(lnum,1), self.embedder(pts2l.detach())],-1))
                else:
                    vis = self.visibility_net(torch.cat([point_emb.tile(lnum,1), self.embedder(pts2l)],-1))
                if self.conf.get_bool('train.vis_rgb_detach',default=False):
                    rgb = (brdf*light_int*cos*vis.detach().clamp(0,1)).clamp(0,1)   ## vis detached
                else:
                    rgb = (brdf*light_int*cos*vis.clamp(0,1)).clamp(0,1)
                vis_values[surface_mask.expand(lnum,-1)] = vis.expand(rgb.shape)
            else:
                rgb = (brdf*light_int*cos).clamp(0,1)

            rgb_values[surface_mask.expand(lnum,-1)] = rgb
            albedo_values[surface_mask] = albedo
            if self.render_model in ['sgbasis']:
                rough_values[surface_mask.expand(lnum,-1)] = rough.expand(-1,3)
            else:
                rough_values[surface_mask] = rough.expand(-1,3)

            if self.xyz_jitter_std > 0:
                xyz_jitter = surface_points + torch.normal(0,torch.ones_like(surface_points)*self.xyz_jitter_std)
                point_emb_jitter = self.embedder(xyz_jitter)
                albedo_jitter = self.albedo_net(point_emb_jitter)
                rough_jitter = self.rough_net(point_emb_jitter)
                albedo_jitter_all = torch.ones_like(points).float().to(device)
                albedo_jitter_all[surface_mask] = albedo_jitter
                if self.render_model in ['sgbasis']:
                    rough_jitter_all = torch.ones_like(weight_values).float().to(device)
                    rough_jitter_all[surface_mask] = F.relu(rough_jitter)
                    rough_ori = weight_values
                elif self.render_model == 'microfacet':
                    rough_jitter_all = torch.ones_like(points).float().to(device)
                    rough_jitter_all[surface_mask] = rough_jitter.expand(-1,3)
                    rough_ori = rough_values
                jitter = {
                    'albedo_values': albedo_values,
                    'albedo_jitter': albedo_jitter_all,
                    'rough_values': rough_ori,
                    'rough_jitter': rough_jitter_all,
                }
            


        output = {
            'points': points,
            'object_mask': object_mask,
            'network_object_mask': surface_mask,
            'sg_rgb_values': rgb_values,
            'normal_values': normals,
            'sg_diffuse_albedo_values': albedo_values,
            'sg_specular_rgb_values': rough_values,
        }

        if self.xyz_jitter_std > 0 and surface_points.shape[0] > 0:
            output.update(jitter)
        if self.normal_mlp:
            output.update(normal_output)
        if self.visibility:
            output['visibility'] = vis_values
            if 'vis_train_gt' in input or 'light_vis_train' in input:
                lnum = input['light_vis_train'].shape[0]
                pts2l_train = input['light_vis_train'][:,None].to(device).expand(-1, rgb_values.shape[1],-1)[surface_mask.expand(lnum,-1)]
                vis_train = torch.ones_like(points).float().to(device)
                if lnum>1:
                    vis_train = vis_train.repeat(lnum,1,1)
                if self.light_vis_detach:
                    vis = self.visibility_net(torch.cat([point_emb.tile(lnum,1), self.embedder(pts2l_train.detach())],-1))
                else:
                    vis = self.visibility_net(torch.cat([point_emb.tile(lnum,1), self.embedder(pts2l_train)],-1))
                vis_train[surface_mask.expand(lnum,-1)] = vis.expand(-1,3)
                output['vis_train'] = vis_train
        if self.render_model in ['sgbasis']:
            output['sg_weight'] = weight_values

        return output
