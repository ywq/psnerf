from __future__ import division
import os
import numpy as np
from imageio import imread
import torch
import torch.utils.data as data
from glob import glob
import json

from datasets import pms_transforms
np.random.seed(0)


class UPS_Custom_Dataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = args.bm_dir
        self.split  = split
        self.args   = args
        self.light_intnorm_gt = args.light_intnorm_gt
        self.para = json.load(open(os.path.join(self.root,'params.json')))
        self.poses = np.array(self.para['pose_c2w']).astype(np.float32)
        args.light_is_same = self.para['light_is_same']
        self.objs   = ['view_{:02d}'.format(vi) for vi in range(1,self.para['n_view']+1)]
        args.log.printWrite('[%s Data] \t%d views. Root: %s' % (split, len(self.objs), self.root))

    def __getitem__(self, index):
        obj   = self.objs[index]
        assert index == int(obj[-2:])-1
        if self.args.light_intnorm_gt:
            assert 'light_intensity' in self.para
            img_list = sorted(glob(os.path.join(self.root, f'img_intnorm_gt/{obj}/*.png')))
        else:
            img_list = sorted(glob(os.path.join(self.root, f'img/{obj}/*.png')))
        names = [os.path.basename(i) for i in img_list]
        assert [int(ni.split('.')[0]) for ni in names]==[ni+1 for ni in range(len(names))]
        poses = self.poses[index]

        if 'light_direction' in self.para:
            if self.para['light_is_same']:
                light_dir = np.array(self.para['light_direction']).astype(np.float32)
            else:
                light_dir = np.array(self.para['light_direction'][index]).astype(np.float32)
        else:
            light_dir = np.zeros((len(names), 3))
            light_dir[:,2] = 1

        if 'light_intensity' in self.para:
            if self.para['light_is_same']:
                light_int = np.array(self.para['light_intensity']).astype(np.float32)
            else:
                light_int = np.array(self.para['light_intensity'][index]).astype(np.float32)
        else:
            light_int = np.ones((len(names), 3))

        if self.para['light_is_same'] and self.args.train_light is not None:
            assert f'light_slt_{self.args.train_light}' in self.para, f"The light number {self.args.train_light} is not supported in `params.json`"
            lslt = np.array(self.para[f'light_slt_{self.args.train_light}'])
            light_dir = light_dir[lslt]
            light_int = light_int[lslt]
            img_list = [img_list[li] for li in lslt]
            names = [names[li] for li in lslt]

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            imgs.append(img)
        img = np.concatenate(imgs, 2)
        h, w, c = img.shape

        if os.path.exists(os.path.join(self.root, 'normal')):
            normal_path = os.path.join(self.root, f'normal/npy/{obj}.npy')
            normal = np.load(normal_path)
            if self.para['gt_normal_world']:
                normal = np.einsum('ij,hwj->hwi', poses[:3,:3].T,normal)
        else:
            normal = np.zeros((h, w, 3))

        mask = np.array(imread(os.path.join(self.root, f'norm_mask/{obj}.png')))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask[...,None]/255.
        img  = img * mask.repeat(img.shape[2], 2)

        mi, mj,_ = np.where(mask)
        pad = 15
        crop = (max(0,min(mi)-pad), max(0,min(mj)-pad), min(h,max(mi)+pad), min(w,max(mj)+pad), )
        indices = np.meshgrid(np.arange(crop[0], crop[2]),
                            np.arange(crop[1], crop[3]),
                            indexing='ij')
        img = img[tuple(indices)]
        mask = mask[tuple(indices)]
        normal = normal[tuple(indices)]

        item = {'normal': normal, 'img': img, 'mask': mask}

        downsample = 4 
        for k in item.keys():
            item[k] = pms_transforms.imgSizeToFactorOfK(item[k], downsample)

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item['dirs'] = torch.from_numpy(light_dir).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(light_int).view(-1, 1, 1).float()
        item['view'] = obj
        item['crop'] = crop
        item['imres'] = (h,w)
        return item

    def __len__(self):
        return len(self.objs)
