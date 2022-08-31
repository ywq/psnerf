# save normalized/averaged lighting images
from PIL import Image
import numpy as np
import os, imageio, json
import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument("--obj", type=str, default=None, )
parser.add_argument("--path", type=str, default='dataset', )
parser.add_argument("--train_light", type=int, default=None, )
parser.add_argument("--light_intnorm", action="store_true", default=False, )
parser.add_argument("--sdps", action="store_true", default=False, )
args = parser.parse_args()

obj_name = args.obj
datadir = args.path
train_light = args.train_light
para = json.load(open(os.path.join(datadir,obj_name,'params.json')))
n_view = para['n_view']
light_is_same = para['light_is_same']
if train_light is None:
    if light_is_same:
        train_light = len(para['light_direction'])
        light_slt = [np.arange(train_light)] * n_view
        avgdir = f'avg_l{train_light}'
    else:
        light_slt = [np.arange(len(ll)) for ll in para['light_direction']]
        avgdir = 'avg'
else:
    assert light_is_same
    assert f'light_slt_{train_light}' in para, f"The light number {train_light} is not supported in `params.json`"
    light_slt = [para[f'light_slt_{train_light}']] * n_view
    avgdir = f'avg_l{train_light}'
if args.light_intnorm:
    if args.sdps:
        if light_is_same:
            light_int = np.load(os.path.join(datadir,obj_name,f'sdps_out_l{train_light}/light_intensity_pred.npy'), allow_pickle=True)
            normalizedir = os.path.join(datadir, obj_name, f'img_intnorm_sdps_l{train_light}')
        else:
            light_int = np.load(os.path.join(datadir,obj_name,'sdps_out/light_intensity_pred.npy'), allow_pickle=True)
            normalizedir = os.path.join(datadir, obj_name, 'img_intnorm_sdps')
        avgdir = 'avg'
    else:
        assert 'light_intensity' in para
        light_int = [np.array(para['light_intensity'])[light_slt[0]]]*n_view if light_is_same else [np.array(ll) for ll in para['light_intensity']]
        normalizedir = os.path.join(datadir, obj_name, f'img_intnorm_gt')
else:
    normalizedir = os.path.join(datadir, obj_name, f'img')

os.makedirs(os.path.join(normalizedir, avgdir), exist_ok=True)
for vi in range(n_view):
    print(f'\r processing: {obj_name}  view: {vi+1}/{n_view}', end='\t\t\t')
    mask = np.array(imageio.imread(os.path.join(datadir, obj_name, 'mask/view_{:02d}.png'.format(vi+1)))).astype(bool)
    if args.light_intnorm:
        relat_int = light_int[vi]/light_int[vi][3] if (light_is_same and args.train_light is None) else light_int[vi]/light_int[vi][0]
        os.makedirs(os.path.join(normalizedir, 'view_{:02d}'.format(vi+1)), exist_ok=True)

    light_img =[]
    for idx, li in enumerate(light_slt[vi]):
        limg = np.array(imageio.imread(os.path.join(datadir, obj_name,'img/view_{:02d}/{:03d}.png'.format(vi+1,li+1))))/255.
        limg = limg*mask[...,None]
        if args.light_intnorm:
            limg = limg/relat_int[idx]
            imageio.imwrite(os.path.join(normalizedir, 'view_{:02d}/{:03d}.png'.format(vi+1,li+1)), (limg.clip(0,1)*255).round().astype(np.uint8))
        light_img.append(limg)
    light_img_mean = np.mean(light_img,axis=0)
    imageio.imwrite(os.path.join(normalizedir, avgdir,'view_{:02d}.png'.format(vi+1)), (light_img_mean.clip(0,1)*255).round().astype(np.uint8))
print()