import os,sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse
import GPUtil
from pyhocon import ConfigFactory
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import utils.general as utils
import imageio, cv2, json
from tqdm import tqdm, trange
from utils.eval_utils import (load_light, vis_light, gen_light_xyz)
from scipy.spatial.transform import Rotation as Rotlib

to_img = lambda x: (x.astype(np.float32).clip(0,1) * 255).round().astype(np.uint8)
to_numpy = lambda x: x.detach().cpu().numpy()

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    exps_folder = kwargs['exps_folder']
    expname = kwargs['expname']
    obj_name = kwargs['obj_name']
    expdir = os.path.join(exps_folder, obj_name, expname)
    test_out_path = os.path.join(kwargs['test_out_dir'],obj_name, expname)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(expdir):
            timestamps = os.listdir(expdir)
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    kwargs['conf'] = os.path.join(expdir, timestamp, 'runconf.conf')
    conf = ConfigFactory.parse_file(kwargs['conf'])
    conf.put('brdf.net.xyz_jitter_std', 0)
    conf.put('normal.net.xyz_jitter_std', 0)
    if kwargs['render_envmap']:
        conf.put('train.light_train', False)

    light_train = conf.get_bool('train.light_train',default=False)
    light_inten_train = conf.get_bool('train.light_inten_train',default=False)

    normal_mlp = conf.get_bool('train.normal_mlp', default = False)
    visibility = conf.get_bool('train.visibility',default=False)
    shape_pregen = conf.get_bool('train.shape_pregen', default = False)
    stage1_shape_path = conf.get_string('train.stage1_shape_path', default=None)
    render_model = conf.get_string('train.render_model', default='sgbasis')

    basedir = conf.get_string('dataset.data_dir')
    para = json.load(open(os.path.join(basedir,'params.json')))
    n_view = para['n_view']
    train_all_view = conf.get_bool('dataset.all_view', default=False)
    if train_all_view:
        test_slt = np.arange(n_view)
    else:
        test_slt = np.array(para['view_test'])

    imh, imw = para['imhw']
    img_res = [imh,imw] 
    total_pixels = imh * imw 
    KK = torch.tensor(np.array(para['K']).astype(np.float32))
    # camera to world matrix, OpenGL ==> OpenCV
    poses_all = np.array(para['pose_c2w']).astype(np.float32) # all views
    pose0 = poses_all[test_slt].copy()   # openGL coordinates
    poses = poses_all[test_slt].copy()  
    poses[:,:3,1:3]*=-1.   # openCV coordinates

    light_is_same = para['light_is_same']
    if light_is_same:
        light_direction = np.array(para['light_direction']).astype(np.float32)
        n_light = len(light_direction) if not train_all_view else conf.get_int('dataset.train_light', default=len(light_direction))
        light_slt = np.arange(n_light)
        light_direction = np.einsum('bij,kj->bki',pose0[:,:3,:3],light_direction)
        light_direction = [torch.from_numpy(ll) for ll in light_direction]
        light_slt = [light_slt] * len(test_slt)
        print(f"test_view: {len(test_slt)} , light is same,  test_light: {n_light}" )
    else:
        light_direction = [np.array(ll).astype(np.float32) for li, ll in enumerate(para['light_direction']) if li in test_slt]
        light_direction = [np.einsum('ij,kj->ki',pose0[li,:3,:3],ll) for li,ll in enumerate(light_direction)]
        light_direction = [torch.from_numpy(ll) for ll in light_direction]
        light_slt = [np.arange(len(ll)) for ll in light_direction]
        print(f"test_view: {len(test_slt)} , test_light: {[len(li) for li in light_direction]}" )

    poses = torch.tensor(poses)
    pose0 = torch.tensor(pose0)

    dir_list = ['rgb', 'normal', 'albedo', 'rough',  'mask']
    if kwargs['render_envmap']:
        light_h = 16  
        kwargs['envmap_path'] = os.path.join(kwargs['envmap_path'],'indoor-{:02d}/indoor-{:02d}.exr'.format(kwargs['envmap_id'],kwargs['envmap_id']))
        env_light = load_light(kwargs['envmap_path'], light_h=light_h)
        env_light *= kwargs['envmap_scale']
        envmap_name = os.path.basename(kwargs['envmap_path'])[:-len('.hdr')]
        test_out_path = os.path.join(test_out_path, f'envmap/{envmap_name}')
        os.makedirs(test_out_path, exist_ok=True)
        os.system("""cp -r {0} "{1}" """.format(kwargs['envmap_path'], test_out_path))
        _ = vis_light(env_light, os.path.join(test_out_path,envmap_name+'.png'), h=light_h*8)
        lxyz, lareas = gen_light_xyz(light_h, 2*light_h, envmap_radius=1)
        lxyz = lxyz.reshape(-1,3)
        env_light = env_light.reshape(-1,3)
        dir_list = ['rgb',]
    if visibility:
        dir_list.append('visibility')
    if kwargs['edit_albedo'] or kwargs['edit_specular']:
        albedo_new,basis_new=None,None
        nexp = ''
        if kwargs['edit_albedo']:
            if kwargs['color'] is None:
                albedo_new = np.random.choice(range(128),size=3)
                nexp += '#{:02x}{:02x}{:02x}'.format(*list(albedo_new))
            else:
                albedo_new = np.array([int(kwargs['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
                albedo_new = albedo_new.astype(np.float32)/5.
                nexp = kwargs['color']
            albedo_new = (albedo_new/255.).astype(np.float32)
        if kwargs['edit_specular']:
            if kwargs['basis'] is None:
                basis_new = np.random.choice(range(9))
            else:
                basis_new = kwargs['basis']
            nexp = f'sg{basis_new+1}' if nexp=='' else nexp+f'_sg{basis_new+1}'
        test_out_path = os.path.join(test_out_path, f'edit_material/{nexp}')
        os.makedirs(test_out_path, exist_ok=True)
        dir_list = ['rgb', 'albedo', 'rough']
    for savedir in dir_list:
        os.makedirs(os.path.join(test_out_path, savedir, 'img'), exist_ok=True)
        if kwargs['save_npy'] or savedir=='normal':
            os.makedirs(os.path.join(test_out_path, savedir, 'npy'), exist_ok=True)


    print('Output directory is: ', test_out_path)
    os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], test_out_path))

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf).to(device)
    model.eval()

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])
    print('Loaded checkpoint: ', ckpt_path)
    if light_train and train_all_view:
        llen = sum([len(ln) for ln in light_slt])
        light_para = torch.nn.Embedding(llen, 3, sparse=True).eval().requires_grad_(False).to(device)
        data =torch.load(os.path.join(old_checkpnts_dir, 'LightParameters', str(kwargs['checkpoint']) + ".pth"))
        light_para.load_state_dict(data["light_state_dict"])
        print('Loaded light direction checkpoint: ', os.path.join(old_checkpnts_dir, 'LightParameters', str(kwargs['checkpoint']) + ".pth"))
        if light_inten_train:
            light_inten_para = torch.nn.Embedding(llen, 1, sparse=True).eval().requires_grad_(False).to(device)
            data =torch.load(os.path.join(old_checkpnts_dir, 'LightParameters', str(kwargs['checkpoint']) + ".pth"))
            light_inten_para.load_state_dict(data["light_inten_state_dict"])
            print('Loaded light intensity checkpoint: ', os.path.join(old_checkpnts_dir, 'LightParameters', str(kwargs['checkpoint']) + ".pth"))

    with open(os.path.join(test_out_path, 'ckpt_path.txt'), 'w') as fp:
        fp.write(ckpt_path + '\n')

    ####################################################################################################################
    print("evaluating...")
    tonemap_img = lambda x: np.power(x, 1./kwargs['gamma'])

    # render envmap only
    if kwargs['render_envmap']:
        for idx, pi in enumerate(tqdm(poses, ncols=120, desc='View')):
            vidx = idx
            vidx_ori = test_slt[idx]

            uv = np.mgrid[0:img_res[0], 0:img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0)

            model_input = {
                "object_mask": torch.ones(1,imh*imw,),
                "uv": uv[None,],
                "intrinsics": KK[None,],
                "pose": poses[idx:idx+1],
                "normal": torch.ones(1,imh*imw,3),
            }
            if shape_pregen:
                model_input['points'] = torch.tensor(np.load(os.path.join(stage1_shape_path,'points/view_{:02d}.npy'.format(vidx_ori+1)))).reshape(1,-1,3)
                model_input['surface_mask'] = torch.tensor(np.load(os.path.join(stage1_shape_path,'mask/view_{:02d}.npy'.format(vidx_ori+1)))).reshape(1,-1,)

            for sub_term in model_input.keys():
                model_input[sub_term] = model_input[sub_term].to(device)
            
            light_dim = light_h**2*2
            lbatch = kwargs['light_batch']
            rgb_all, vis_all = [], []
            for lstart in trange(0, light_dim, lbatch, ncols=120, leave=False, desc=f'Light (batch={lbatch})'):
                lend = min(light_dim, lstart+lbatch)
                model_input['light_direction'] = F.normalize(torch.tensor(lxyz[lstart:lend]).float().to(device),p=2,dim=-1)
                model_input['light_intensity'] = torch.tensor(env_light[lstart:lend]).float().to(device)

                split = utils.split_input(model_input, total_pixels)
                res = []
                for s in split:
                    out = model(s)
                    res_list = ['sg_rgb_values']
                    res_sub = {res_n:out[res_n].detach() for res_n in res_list}
                    if visibility:
                        res_sub['visibility'] = out.get('visibility',torch.ones_like(out['points'])).detach()
                    res.append(res_sub)
                model_outputs = utils.merge_output(res, total_pixels, 1)

                rgb_all.append(to_numpy(model_outputs['sg_rgb_values'].reshape(-1, *img_res, 3)))
                if visibility:
                    vis_all.append(to_numpy(model_outputs['visibility'].reshape(-1, *img_res, 3)))
            rgb_all = np.concatenate(rgb_all,0).sum(0).clip(0,1)

            img = Image.fromarray(to_img(tonemap_img(rgb_all)))
            img.save(os.path.join(test_out_path,'rgb/img/view_{:02d}.png'.format(vidx_ori+1)))
            # visibility
            if visibility:
                vis_all = np.concatenate(vis_all,0).mean(0)
                img = Image.fromarray(to_img(vis_all))
                img.save(os.path.join(test_out_path,'visibility/img/view_{:02d}.png'.format(vidx_ori+1)))
            if kwargs['save_npy']:
                np.save(os.path.join(test_out_path,'rgb/npy/view_{:02d}.npy'.format(vidx_ori+1)), rgb_all.astype(np.float32))
                if visibility:
                    np.save(os.path.join(test_out_path,'visibility/npy/view_{:02d}.npy'.format(vidx_ori+1)), vis_all.astype(np.float32))
        return
    
    if kwargs['edit_albedo'] or kwargs['edit_specular']:
        for idx, pi in enumerate(tqdm(poses, ncols=120, desc='View')):
            vidx = idx
            vidx_ori = test_slt[idx]
            lidx = torch.tensor(light_slt[idx]).long()
            lidx_ori = light_slt[idx]

            uv = np.mgrid[0:img_res[0], 0:img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0)

            model_input = {
                "object_mask": torch.ones(1,imh*imw,),
                "uv": uv[None,],
                "intrinsics": KK[None,],
                "lidx": lidx ,
                "pose": poses[idx:idx+1],
                "normal": torch.ones(1,imh*imw,3),
            }
            if shape_pregen:
                model_input['points'] = torch.tensor(np.load(os.path.join(stage1_shape_path,'points/view_{:02d}.npy'.format(vidx_ori+1)))).reshape(1,-1,3)
                model_input['surface_mask'] = torch.tensor(np.load(os.path.join(stage1_shape_path,'mask/view_{:02d}.npy'.format(vidx_ori+1)))).reshape(1,-1,)
      
            os.makedirs(os.path.join(test_out_path, 'rgb/img/view_{:02d}'.format(vidx_ori+1)), exist_ok=True)
            if render_model == 'sgbasis':
                os.makedirs(os.path.join(test_out_path, 'rough/img/view_{:02d}'.format(vidx_ori+1)), exist_ok=True) 

            for sub_term in model_input.keys():
                model_input[sub_term] = model_input[sub_term].to(device)
            
            lbatch = kwargs['light_batch']
            rgb_all, vis_all, rough_all = [], [], []
            for lstart in trange(0, len(light_slt[idx]), lbatch, ncols=120, leave=False, desc=f'Light (batch={lbatch})'):
                lend = min(len(light_slt[idx]), lstart+lbatch)
                model_input['light_direction'] = light_direction[vidx][lstart:lend].to(device)
                if light_train and train_all_view:
                    accu = [len(ln) for ln in light_slt]
                    l_slt = sum(accu[:vidx]) + lidx[lstart:lend]
                    model_input['light_direction'] = F.normalize(light_para.to(device)(l_slt.to(device)),p=2,dim=-1)
                    if light_inten_train:
                        model_input['light_intensity'] = light_inten_para.to(device)(l_slt.to(device))

                split = utils.split_input(model_input, total_pixels)
                res = []
                for s in split:
                    out = model(s,albedo_new=albedo_new,basis_new=basis_new)
                    res_sub = {res_n:out[res_n].detach() for res_n in out}
                    res.append(res_sub)
                model_outputs = utils.merge_output(res, total_pixels, 1)

                rgb_all.append(to_numpy(model_outputs['sg_rgb_values'].reshape(-1, *img_res, 3)))
                rough_all.append(to_numpy(model_outputs['sg_specular_rgb_values'].reshape(-1, *img_res, 3)))
            rgb_all = np.concatenate(rgb_all,0).clip(0,1)
            if render_model in ['sgbasis']:
                rough_all = np.concatenate(rough_all,0)
            else:
                rough_all = rough_all[-1][0]

            ## img
            for lli, lli_ori in enumerate(lidx_ori):
                img = Image.fromarray(to_img(rgb_all[lli]))
                img.save(os.path.join(test_out_path,'rgb/img/view_{:02d}/{:03d}.png'.format(vidx_ori+1,lli_ori+1)))
            ## rough
            if render_model in ['sgbasis']:
                for lli, lli_ori in enumerate(lidx_ori):
                    img = Image.fromarray(to_img(rough_all[lli]))
                    img.save(os.path.join(test_out_path,'rough/img/view_{:02d}/{:03d}.png'.format(vidx_ori+1,lli_ori+1)))
            else:
                img = Image.fromarray(to_img(rough_all))
                img.save(os.path.join(test_out_path,'rough/img/view_{:02d}.png'.format(vidx_ori+1)))
            ## albedo
            albedo_all = to_numpy(model_outputs['sg_diffuse_albedo_values'].reshape(*img_res, 3))
            img = Image.fromarray(to_img(albedo_all))
            img.save(os.path.join(test_out_path,'albedo/img/view_{:02d}.png'.format(vidx_ori+1)))
            
            if kwargs['save_npy']:
                np.save(os.path.join(test_out_path,'rgb/npy/view_{:02d}.npy'.format(vidx_ori+1)), rgb_all.astype(np.float32))
                np.save(os.path.join(test_out_path,'rough/npy/view_{:02d}.npy'.format(vidx_ori+1)), rough_all.astype(np.float32))
                np.save(os.path.join(test_out_path,'albedo/npy/view_{:02d}.npy'.format(vidx_ori+1)), albedo_all.astype(np.float32))
        return

    for idx, pi in enumerate(tqdm(poses, ncols=120, desc='View')):
        vidx = idx
        vidx_ori = test_slt[idx]
        lidx = torch.tensor(light_slt[idx]).long()
        lidx_ori = light_slt[idx]

        uv = np.mgrid[0:img_res[0], 0:img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        model_input = {
            "object_mask": torch.ones(1,imh*imw,),
            "uv": uv[None,],
            "intrinsics": KK[None,],
            "lidx": lidx ,
            "pose": poses[idx:idx+1],
            "normal": torch.ones(1,imh*imw,3),
        }
        if shape_pregen:
            model_input['points'] = torch.tensor(np.load(os.path.join(stage1_shape_path,'points/view_{:02d}.npy'.format(vidx_ori+1)))).reshape(1,-1,3)
            model_input['surface_mask'] = torch.tensor(np.load(os.path.join(stage1_shape_path,'mask/view_{:02d}.npy'.format(vidx_ori+1)))).reshape(1,-1,)
      
        os.makedirs(os.path.join(test_out_path, 'rgb/img/view_{:02d}'.format(vidx_ori+1)), exist_ok=True)
        if visibility:
            os.makedirs(os.path.join(test_out_path, 'visibility/img/view_{:02d}'.format(vidx_ori+1)), exist_ok=True)
        if render_model == 'sgbasis':
            os.makedirs(os.path.join(test_out_path, 'rough/img/view_{:02d}'.format(vidx_ori+1)), exist_ok=True)

        for sub_term in model_input.keys():
            model_input[sub_term] = model_input[sub_term].to(device)
        
        lbatch = kwargs['light_batch']
        rgb_all, vis_all, rough_all = [], [], []
        for lstart in trange(0, len(light_slt[idx]), lbatch, ncols=120, leave=False, desc=f'Light (batch={lbatch})'):
            lend = min(len(light_slt[idx]), lstart+lbatch)
            model_input['light_direction'] = light_direction[vidx][lstart:lend].to(device)
            # gt light directions
            if light_train and train_all_view:
                accu = [len(ln) for ln in light_slt]
                l_slt = sum(accu[:vidx]) + lidx[lstart:lend]
                model_input['light_direction'] = F.normalize(light_para.to(device)(l_slt.to(device)),p=2,dim=-1)
                if light_inten_train:
                    model_input['light_intensity'] = light_inten_para.to(device)(l_slt.to(device))


            split = utils.split_input(model_input, total_pixels)
            res = []
            for s in split:
                out = model(s)
                res_sub = {res_n:out[res_n].detach() for res_n in out}
                res.append(res_sub)
            model_outputs = utils.merge_output(res, total_pixels, 1)

            rgb_all.append(to_numpy(model_outputs['sg_rgb_values'].reshape(-1, *img_res, 3)))
            rough_all.append(to_numpy(model_outputs['sg_specular_rgb_values'].reshape(-1, *img_res, 3)))
            if visibility:
                vis_all.append(to_numpy(model_outputs['visibility'].reshape(-1, *img_res, 3)))
        rgb_all = np.concatenate(rgb_all,0).clip(0,1)
        if render_model in ['sgbasis']:
            rough_all = np.concatenate(rough_all,0)
        else:
            rough_all = rough_all[-1][0]

        # img
        for lli, lli_ori in enumerate(lidx_ori):
            img = Image.fromarray(to_img(rgb_all[lli]))
            img.save(os.path.join(test_out_path,'rgb/img/view_{:02d}/{:03d}.png'.format(vidx_ori+1,lli_ori+1)))
        # mask
        rmask = to_numpy(model_outputs['network_object_mask'].reshape(*img_res))
        img = Image.fromarray(to_img(rmask))
        img.save(os.path.join(test_out_path,'mask/img/view_{:02d}.png'.format(vidx_ori+1)))
        # specular
        if render_model in ['sgbasis']:
            for lli, lli_ori in enumerate(lidx_ori):
                img = Image.fromarray(to_img(rough_all[lli]))
                img.save(os.path.join(test_out_path,'rough/img/view_{:02d}/{:03d}.png'.format(vidx_ori+1,lli_ori+1)))
        else:
            img = Image.fromarray(to_img(rough_all))
            img.save(os.path.join(test_out_path,'rough/img/view_{:02d}.png'.format(vidx_ori+1)))
        # normal
        normal = model_outputs['normal_values'] if not normal_mlp else model_outputs['normal_pred']
        normal = to_numpy(normal.reshape(*img_res, 3)) * rmask[...,None]
        np.save(os.path.join(test_out_path,'normal/npy/view_{:02d}.npy'.format(vidx_ori+1)), normal.astype(np.float32))
        img = Image.fromarray(to_img(normal/2.+0.5))
        img.save(os.path.join(test_out_path,'normal/img/view_{:02d}.png'.format(vidx_ori+1)))
        # albedo
        rgb_eval = to_numpy(model_outputs['sg_diffuse_albedo_values'].reshape(*img_res, 3)).clip(0,1)
        img = Image.fromarray(to_img(rgb_eval))
        img.save(os.path.join(test_out_path,'albedo/img/view_{:02d}.png'.format(vidx_ori+1)))
        
        # visibility
        if visibility:
            vis_all = np.concatenate(vis_all,0).clip(0,1)
            for lli, lli_ori in enumerate(lidx_ori):
                img = Image.fromarray(to_img(vis_all[lli]))
                img.save(os.path.join(test_out_path,'visibility/img/view_{:02d}/{:03d}.png'.format(vidx_ori+1,lli_ori+1)))

        if kwargs['save_npy']:
            np.save(os.path.join(test_out_path,'rgb/npy/view_{:02d}.npy'.format(vidx_ori+1)), rgb_all.astype(np.float32))
            np.save(os.path.join(test_out_path,'mask/npy/view_{:02d}.npy'.format(vidx_ori+1)), rmask.astype(bool))
            np.save(os.path.join(test_out_path,'rough/npy/view_{:02d}.npy'.format(vidx_ori+1)), rough_all.astype(np.float32))
            np.save(os.path.join(test_out_path,'albedo/npy/view_{:02d}.npy'.format(vidx_ori+1)), rgb_eval.astype(np.float32))
            if visibility:
                np.save(os.path.join(test_out_path,'visibility/npy/view_{:02d}.npy'.format(vidx_ori+1)), vis_all.astype(np.float32))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--obj_name', type=str, default='', help='The object name to be evaluated.')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--test_out_dir', type=str, default='test_out')
    parser.add_argument('--exps_folder', type=str, default='out', help='The experiments folder name.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--gamma', type=float, default=1., help='gamma correction coefficient')

    parser.add_argument('--render_envmap', default=False, action="store_true", help='If set, render envmap')
    parser.add_argument('--envmap_path', type=str, default='envmap')
    parser.add_argument('--envmap_id', default=3, type=int,)
    parser.add_argument('--envmap_scale', default=1, type=float, help='scale of envmap intensity')
    parser.add_argument('--edit_albedo', default=False, action="store_true", help='If set, edit albedo')
    parser.add_argument('--edit_specular', default=False, action="store_true", help='If set, edit specular')
    parser.add_argument('--basis', default=None, type=int, help='specular basis')
    parser.add_argument('--color', default=None, type=str, help='albedo color')

    parser.add_argument('--save_npy', action='store_true', default=False)
    parser.add_argument('--light_batch', default=64, type=int,)
    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(**vars(opt))
