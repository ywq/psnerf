import os, sys
import argparse
import numpy as np
import imageio, json
from pyhocon import ConfigFactory
from collections import defaultdict

import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')

from stage2.utils.metrics import MAE, PSNR, SSIM, LPIPS
metrics = {'psnr':PSNR, 'ssim':SSIM, 'lpips':LPIPS()}

# for training with input images normalized using light intensity predicted by sdps-net
def scale_img(img, gt, mask):
    opt_scale = []
    for i in range(3):
        x_hat = img[:, :, i][mask]
        x = gt[:, :, i][mask]
        scale = x_hat.dot(x) / x_hat.dot(x_hat)
        opt_scale.append(scale)
    opt_scale = np.array(opt_scale).mean()
    img = (img*opt_scale).clip(0,1)
    return img
  
bg = lambda x, mask: x*mask[...,None] + 1*~mask[...,None]   ## white background

# Arguments
parser = argparse.ArgumentParser(
    description='Evaluation'
)
parser.add_argument('--obj_name', type=str, default='bear',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--test_out_dir', type=str, default='stage2/test_out',)
args = parser.parse_args()

test_out_path = os.path.join(args.test_out_dir, args.obj_name, args.expname)
conf = ConfigFactory.parse_file(os.path.join(test_out_path,'runconf.conf'))
data_path = conf.get_string('dataset.data_dir')
train_all_view = conf.get_bool('dataset.all_view', default=False)
inten_normalize = conf.get_string('dataset.inten_normalize', default=None)
im_sub = 'img_intnorm_gt' if inten_normalize is not None else 'img'
if data_path.startswith('../'):
    data_path = data_path[3:]
para = json.load(open(os.path.join(data_path,'params.json')))

n_view = para['n_view']
if train_all_view:
    test_slt = np.arange(n_view)
else:
    test_slt = np.array(para['view_test'])
light_is_same = para['light_is_same']
gt_normal_world = para['gt_normal_world']
poses = np.array(para['pose_c2w']).astype(np.float32)
if light_is_same:
    light_direction = np.array(para['light_direction']).astype(np.float32)
    n_light = len(light_direction) if not train_all_view else conf.get_int('dataset.train_light', default=len(light_direction))
    light_slt = np.arange(n_light)
    light_slt = [light_slt] * len(test_slt)
    print(f"evaluation_view: {len(test_slt)} , light is same,  evaluation_light: {n_light}" )
else:
    light_direction = [np.array(ll).astype(np.float32) for li, ll in enumerate(para['light_direction']) if li in test_slt]
    light_slt = [np.arange(len(ll)) for ll in light_direction]
    print(f"evaluation_view: {len(test_slt)} , evaluation_light: {[len(li) for li in light_direction]}" )

img_data = defaultdict(list)
normal_data = []
for vidx, vi in enumerate(test_slt):
    mask_gt = np.array(imageio.imread(os.path.join(data_path,'norm_mask/view_{:02d}.png'.format(vi+1)))).astype(bool)
    mask_pred = np.array(imageio.imread(os.path.join(test_out_path,'mask/img/view_{:02d}.png'.format(vi+1)))).astype(bool)
    mask = mask_pred & mask_gt
    if os.path.exists(os.path.join(data_path,'normal')):
        normal_gt = np.load(os.path.join(data_path,'normal/npy/view_{:02d}.npy'.format(vi+1)))
        if not gt_normal_world:
            normal_gt = np.einsum('ij,hwj->hwi',poses[vi,:3,:3],normal_gt)
        normal_pred = np.load(os.path.join(test_out_path, 'normal/npy/view_{:02d}.npy'.format(vi+1)))
        normal_data.append(MAE(normal_pred,normal_gt, mask)[0])

    for lidx, li in enumerate(light_slt[vidx]):
        print(f'\rview: {vidx+1}/{len(test_slt)}, light: {lidx+1:02d}/{len(light_slt[vidx])}',end='')
        img_gt = imageio.imread(os.path.join(data_path,'{}/view_{:02d}/{:03d}.png'.format(im_sub,vi+1,li+1)))
        img_gt = np.array(img_gt).astype(np.float32)/255.
        img_gt = bg(img_gt, mask_gt)
        img_pred = imageio.imread(os.path.join(test_out_path,'rgb/img/view_{:02d}/{:03d}.png'.format(vi+1,li+1)))
        img_pred = np.array(img_pred).astype(np.float32)/255.
        if inten_normalize == 'sdps':
            img_pred = scale_img(img_pred, img_gt, mask)
        for mi, mtc in metrics.items():
            img_data[mi].append(mtc(bg(img_pred,mask),bg(img_gt,mask), mask))

print()
for mi in [*metrics]:
    err = np.array(img_data[mi]).mean()
    if mi=='lpips': err *= 100
    print(f'{mi.upper()} Error:  {err:.4f}' if mi in ['ssim'] else f'{mi.upper()} Error:  {err:.2f}')
if len(normal_data)>0:
    print(f'Normal MAE Error:  {np.array(normal_data).mean():.2f}')