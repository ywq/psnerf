import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import dataloading as dl
import model as mdl
import imageio, json

from model.common import arange_pixels

np.random.seed(42)
torch.manual_seed(42)
from utils.tools import set_debugger
set_debugger()

to_img = lambda x: (x.astype(np.float32).clip(0,1) * 255).round().astype(np.uint8)
to_numpy = lambda x: x.detach().cpu().numpy()
to_hw = lambda x, h, w: x.reshape(w,h,-1).permute(1,0,2)

# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument('--obj_name', type=str, default='bear',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--exp_folder', type=str, default='out',)
parser.add_argument('--test_out_dir', type=str, default='exps_shape',)
parser.add_argument('--visibility', action='store_true', default=False,)
parser.add_argument('--vis_plus', action='store_true', default=False,)
parser.add_argument('--vis_plus_num', type=int, default=256)
parser.add_argument('--semisphere', action='store_true', default=False,)
parser.add_argument('--load_iter', type=int, default=None)
parser.add_argument('--chunk', type=int, default=32000)
parser.add_argument('--visualize', action='store_true', default=False,)

args = parser.parse_args()
cfg = dl.load_config(os.path.join(args.exp_folder,args.obj_name, args.expname, 'config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# Fix seeds
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# init dataloader
test_loader = dl.get_dataloader(cfg, mode='all', shuffle=False)

# init network
model = mdl.NeuralNetwork(cfg)
# init renderer
renderer = mdl.Renderer(model, cfg, device=device, normal_loss=False)
# init checkpoints and load
out_dir = os.path.join(args.exp_folder, args.obj_name, args.expname)
checkpoint_io = mdl.CheckpointIO(os.path.join(out_dir,'models'), model=model)

try:
    load_dict = checkpoint_io.load(f'model_{args.load_iter}.pt' if args.load_iter else 'model.pt')
except FileExistsError:
    load_dict = dict()
it = load_dict.get('it', 100000)

test_out_path = os.path.join(args.test_out_dir, f"{args.obj_name}/{args.expname}_{args.load_iter if args.load_iter else it}")

for savedir in ['mask', 'points', 'normal']:
    os.makedirs(os.path.join(test_out_path, savedir), exist_ok=True)
if args.visibility:
    os.makedirs(os.path.join(test_out_path, 'visibility'), exist_ok=True)
if args.visualize:
    for savedir in ['vis_mask', 'vis_points', 'vis_normal',]:
        os.makedirs(os.path.join(test_out_path, savedir), exist_ok=True)

f = os.path.join(test_out_path, 'config.yaml')
with open(f, 'w') as file:
    file.write(open(os.path.join(out_dir, 'config.yaml'), 'r').read())

def process_data_dict(data):
    img = data.get('img').to(device)
    img_idx = data.get('img.idx')
    mask_img = data.get('img.mask').unsqueeze(1).to(device)
    world_mat = data.get('img.world_mat').to(device)
    camera_mat = data.get('img.camera_mat').to(device)
    scale_mat = data.get('img.scale_mat').to(device)

    return (img, mask_img, world_mat, camera_mat, scale_mat, img_idx)

if args.visibility:
    light_pred = np.load(os.path.join(test_loader.dataset.est_norm_dir,'light_direction_pred.npy'), allow_pickle=True)
    light_pred = light_pred[test_loader.dataset.train_slt]
    light_pred = [np.einsum('ij,kj->ki',test_loader.dataset.pose0[ln,:3,:3],li).astype(np.float32) for ln,li in enumerate(light_pred)]
    light_all = [torch.tensor(lli).to(device) for lli in light_pred]
    vis_vd = []
    if args.vis_plus:
        from torch_cluster import fps
        vis_plus_light = {}
        os.makedirs(os.path.join(test_out_path, 'vis_plus'), exist_ok=True)

for di, data in enumerate(tqdm(test_loader, ncols=120)):
    (img, mask, world_mat, camera_mat, scale_mat, img_idx) = \
        process_data_dict(data)
    vidx = img_idx.item()
    vidx_ori = test_loader.dataset.train_slt[vidx]
    h, w = img.shape[-2:] #resolution
    p_loc, pixels = arange_pixels(resolution=(h, w))
    pixels = pixels.to(device)
    p_loc = p_loc.to(device)

    light_dir = None
    if args.visibility:
        light_dir = light_all[vidx]
        ln_ori = light_dir.shape[0] 
        if args.vis_plus:
            vnum = args.vis_plus_num 
            rnum = 10000
            view_dir = world_mat[0,:3,2].detach().cpu().numpy()
            vector = np.random.normal(size=(rnum,3))
            unit_vec = vector / np.linalg.norm(vector,axis=-1,keepdims=True)
            if args.semisphere:
                unit_vec = unit_vec[(unit_vec*view_dir).sum(-1)<0]
            pos = torch.tensor(unit_vec).to(device).float()
            index = fps(pos, ratio=vnum/unit_vec.shape[0], random_start=True)
            assert index.shape[0]==vnum
            ln_plus = vnum
            light_dir = torch.cat([light_dir,pos[index]],dim=0)

    with torch.no_grad():
        mask_pred, normal_pred, points_pred, vis_pred, vis_plus = [],[],[],[],[]
        for ii, pixels_i in enumerate(torch.split(p_loc, args.chunk, dim=1)):
            mout = renderer(pixels_i, camera_mat, world_mat, scale_mat, 'shape_extract', 
                        add_noise=False, eval_=True, it=it, visibility=args.visibility, light_dir=light_dir)
            mask_pred.append(mout['mask'])
            normal_pred.append(mout['normal'])
            points_pred.append(mout['points'])
            if args.visibility:
                vis_pred.append(mout['visibility'][:ln_ori])
                if args.vis_plus:
                    vis_plus.append(mout['visibility'][ln_ori:])
    
        mask_all = to_numpy(to_hw(torch.cat(mask_pred, dim=1),h,w))[...,0]
        normal_all = to_numpy(to_hw(torch.cat(normal_pred, dim=1),h,w))
        points_all = to_numpy(to_hw(torch.cat(points_pred, dim=1),h,w))

        np.save(os.path.join(test_out_path,'points/view_{:02d}.npy'.format(vidx_ori+1)),points_all.astype(np.float32))
        np.save(os.path.join(test_out_path,'normal/view_{:02d}.npy'.format(vidx_ori+1)),normal_all.astype(np.float32))
        np.save(os.path.join(test_out_path,'mask/view_{:02d}.npy'.format(vidx_ori+1)),mask_all.astype(bool))
        if args.visualize:
            imageio.imwrite(os.path.join(test_out_path,'vis_points/view_{:02d}.png'.format(vidx_ori+1)),to_img(points_all/2.+0.5))
            imageio.imwrite(os.path.join(test_out_path,'vis_normal/view_{:02d}.png'.format(vidx_ori+1)),to_img(normal_all/2.+0.5))
            imageio.imwrite(os.path.join(test_out_path,'vis_mask/view_{:02d}.png'.format(vidx_ori+1)),to_img(mask_all))

        if args.visibility:
            vis_all = to_numpy(torch.cat(vis_pred, dim=1)).reshape(ln_ori,h,w,).transpose(0,2,1)
            np.save(os.path.join(test_out_path,'visibility/view_{:02d}.npy'.format(vidx_ori+1)),vis_all.astype(np.float32))
            if args.visualize:
                vis_vd.append((vis_all*255).round().astype(np.uint8))
            if args.vis_plus:
                vis_plus_all = to_numpy(torch.cat(vis_plus, dim=1)).reshape(ln_plus,h,w,).transpose(0,2,1)
                np.save(os.path.join(test_out_path,'vis_plus/view_{:02d}.npy'.format(vidx_ori+1)),vis_plus_all.astype(np.float32))
                vis_plus_light[f'view_{vidx_ori+1:02d}'] = to_numpy(pos[index]).astype(np.float32).tolist()
        torch.cuda.empty_cache()
if args.visibility and args.visualize:
    vis_vd = np.concatenate(vis_vd, axis=0)
    imageio.mimwrite(os.path.join(test_out_path,'light_visibility.mp4'), vis_vd, fps=10, quality=8)
if args.visibility and args.vis_plus:
    with open(os.path.join(test_out_path,'vis_plus/light_dir.json'),'w') as f0:
        json.dump(vis_plus_light,f0, indent=4)
