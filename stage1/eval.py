import os
import sys
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import dataloading as dl
import model as mdl

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
    description='Testing of Stage 1'
)
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--obj_name', type=str, default='bear',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--exp_folder', type=str, default='out',)
parser.add_argument('--test_out_dir', type=str, default='test_out',)
parser.add_argument('--load_iter', type=int, default=None)
parser.add_argument('--save_npy', action='store_true', default=False)


args = parser.parse_args()
cfg = dl.load_config(os.path.join(args.exp_folder,args.obj_name, args.expname, 'config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# Fix seeds
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

test_out_path = os.path.join(args.test_out_dir,args.obj_name, args.expname)

for savedir in ['rgb', 'normal', 'mask', 'acc']:
    os.makedirs(os.path.join(test_out_path, savedir, 'img'), exist_ok=True)
    if args.save_npy or savedir=='normal':
        os.makedirs(os.path.join(test_out_path, savedir, 'npy'), exist_ok=True)

# init dataloader
test_loader = dl.get_dataloader(cfg, mode='test',shuffle=False)
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
    
for di, data in enumerate(tqdm(test_loader, ncols=120)):
    (img, mask, world_mat, camera_mat, scale_mat, img_idx) = \
        process_data_dict(data)
    vidx = img_idx.item()
    vidx_ori = test_loader.dataset.train_slt[vidx]
    h, w = img.shape[-2:] #resolution
    p_loc, pixels = arange_pixels(resolution=(h, w))
    pixels = pixels.to(device)
    p_loc = p_loc.to(device)

    with torch.no_grad():
        rgb_pred, normal_pred, acc_pred, mask_pred = [],[],[],[]
        for ii, pixels_i in enumerate(torch.split(p_loc, 1024, dim=1)):
            mout = renderer(pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                        add_noise=False, eval_=True, it=it,)
            rgb_pred.append(mout['rgb'])
            normal_pred.append(mout['normal_pred'])
            acc_pred.append(mout['acc_map'])
            mask_pred.append(mout['mask_pred'])
    
        rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h, w)).astype(np.float32)
        normal_pred = to_numpy(to_hw(torch.cat(normal_pred, dim=1),h, w)).astype(np.float32)
        acc_pred = to_numpy(to_hw(torch.cat(acc_pred, dim=1),h, w))[...,0].astype(np.float32)
        mask_pred = to_numpy(to_hw(torch.cat(mask_pred, dim=0),h, w))[...,0]

        img = Image.fromarray(to_img(rgb_pred))
        img.save(os.path.join(test_out_path,'rgb/img/view_{:02d}.png'.format(vidx_ori+1)))
        img = Image.fromarray(to_img(acc_pred))
        img.save(os.path.join(test_out_path,'acc/img/view_{:02d}.png'.format(vidx_ori+1)))
        img = Image.fromarray(to_img(mask_pred))
        img.save(os.path.join(test_out_path,'mask/img/view_{:02d}.png'.format(vidx_ori+1)))
        img = Image.fromarray(to_img(normal_pred/2.+0.5))
        img.save(os.path.join(test_out_path,'normal/img/view_{:02d}.png'.format(vidx_ori+1)))
        np.save(os.path.join(test_out_path,'normal/npy/view_{:02d}.npy'.format(vidx_ori+1)), normal_pred)
        if args.save_npy:
            np.save(os.path.join(test_out_path,'rgb/npy/view_{:02d}.npy'.format(vidx_ori+1)), rgb_pred)
            np.save(os.path.join(test_out_path,'acc/npy/view_{:02d}.npy'.format(vidx_ori+1)), acc_pred)
            np.save(os.path.join(test_out_path,'mask/npy/view_{:02d}.npy'.format(vidx_ori+1)), mask_pred)
