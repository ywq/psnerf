import os
import torch
import numpy as np
from utils import rend_util
import json, imageio
import random


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 conf=None,
                 split='train',
                 multi_light=True,
                 ):
        self.conf = conf
        self.split = split
        self.basedir = conf.get_string('dataset.data_dir')
        self.multi_light = multi_light and conf.get_bool('train.multi_light', default=False)
        self.light_bs = conf.get_int('train.light_bs', default=32)
        self.sample_in_mask = conf.get_bool('train.sample_in_mask',default=False)
        self.sampling_idx = None

        self.shape_pregen = conf.get_bool('train.shape_pregen', default=False)
        self.stage1_shape_path = conf.get_string('train.stage1_shape_path', default=None)
        assert self.shape_pregen, "currently only support pre-extracted shape"
        assert os.path.exists(self.stage1_shape_path)
        self.vis_loss = conf.get_bool('train.visibility', default=False) and conf.get_bool('train.vis_loss', default=False)
        self.inten_normalize = conf.get_string('dataset.inten_normalize', default=None)
        self.train_all_view = conf.get_bool('dataset.all_view', default=False)

        para = json.load(open(os.path.join(self.basedir,'params.json')))
        self.obj_name = conf.get_string('dataset.obj_name')
        assert os.path.exists(self.basedir), "Data directory is empty"
        print('Creating dataset from: ', self.basedir)
        
        n_view = para['n_view']
        self.train_view = conf.get_int('dataset.train_view', default=None)
        if self.train_all_view:
            train_slt = np.arange(n_view)
        elif split=='train':
            train_slt = np.array(para[f'view_slt_{self.train_view}']) if self.train_view is not None else np.array(para['view_train'])
        elif split=='test':
            train_slt = np.array(para['view_test'])
        else:
            raise ValueError
        self.KK = np.array(para['K']).astype(np.float32)
        # camera to world matrix, OpenGL ==> OpenCV
        poses = np.array(para['pose_c2w']).astype(np.float32) # all views
        self.pose0 = poses.copy()   # openGL coordinates
        self.poses = poses.copy()  
        self.poses[:,:3,1:3]*=-1.   # openCV coordinates

        self.poses = self.poses[train_slt]
        self.pose0 = self.pose0[train_slt]
        self.poses = torch.from_numpy(self.poses)
        self.KK = torch.from_numpy(self.KK)
        self.train_slt = train_slt   # original index of views

        self.light_is_same = para['light_is_same']
        if self.light_is_same:
            self.light_direction = np.array(para['light_direction']).astype(np.float32)
            n_light = len(self.light_direction)
            self.train_light = conf.get_int('dataset.train_light', default=n_light)
            self.light_slt = np.arange(n_light)
            if self.train_light< n_light:
                self.light_slt = np.array(para[f'light_slt_{self.train_light}'])
                self.light_direction = self.light_direction[self.light_slt]
            self.light_direction = np.einsum('bij,kj->bki',self.pose0[:,:3,:3],self.light_direction)
            self.light_direction = [torch.from_numpy(ll) for ll in self.light_direction]
            self.light_slt = [self.light_slt] * len(train_slt)
            print(f"{split}_view: {len(train_slt)} , light is same,  {split}_light: {self.train_light}" )
        else:
            self.light_direction = [np.array(ll).astype(np.float32) for li, ll in enumerate(para['light_direction']) if li in train_slt]
            self.light_direction = [np.einsum('ij,kj->ki',self.pose0[li,:3,:3],ll) for li,ll in enumerate(self.light_direction)]
            self.light_direction = [torch.from_numpy(ll) for ll in self.light_direction]
            self.light_slt = [np.arange(len(ll)) for ll in self.light_direction]
            print(f"{split}_view: {len(train_slt)} , {split}_light: {[len(li) for li in self.light_direction]}" )

        self.im_sub = 'img'
        if self.inten_normalize is not None:
            assert self.inten_normalize in ['gt','sdps']
            self.im_sub += '_intnorm_'+ self.inten_normalize
        if self.light_is_same and self.inten_normalize in ['sdps']:
            self.im_sub += f'_l{self.train_light}'

        masks, gt_normal, normal, points, surface_mask = [],[],[],[],[]
        for v0, vi in enumerate(train_slt):
            mask = np.array(imageio.imread(os.path.join(self.basedir,'mask/view_{:02d}.png'.format(vi+1))))
            masks.append(mask)
            if os.path.exists(os.path.join(self.basedir,'normal')):
                gt_nml = np.load(os.path.join(self.basedir,'normal/npy/view_{:02d}.npy'.format(vi+1)))
                if not para['gt_normal_world']:
                    gt_nml = np.einsum('ij,hwj->hwi', self.pose0[v0,:3,:3],gt_nml)
                gt_normal.append(gt_nml*mask[...,None].astype(bool))
            else:
                gt_normal.append(np.zeros((*mask.shape,3)))
            if self.shape_pregen:
                points.append(np.load(os.path.join(self.stage1_shape_path,'points/view_{:02d}.npy'.format(vi+1))))
                surface_mask.append(np.load(os.path.join(self.stage1_shape_path,'mask/view_{:02d}.npy'.format(vi+1))))
                normal.append(np.load(os.path.join(self.stage1_shape_path,'normal/view_{:02d}.npy'.format(vi+1))))
        masks = (np.array(masks) / 255.).astype(np.float32)
        imh, imw = masks.shape[-2:]
        self.img_res = [imh,imw] 
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.object_masks = torch.from_numpy(masks).bool().view(masks.shape[0],-1)
        self.gt_normal = torch.from_numpy(np.array(gt_normal).astype(np.float32)).view(masks.shape[0],-1,3)
        if self.shape_pregen:
            self.normal = torch.from_numpy(np.array(normal).astype(np.float32)).view(masks.shape[0],-1,3)
            self.points = torch.from_numpy(np.array(points).astype(np.float32)).view(masks.shape[0],-1,3)
            self.surface_mask = torch.from_numpy(np.array(surface_mask)).view(masks.shape[0],-1)
        if self.vis_loss:
            self.visibility = [torch.from_numpy(np.load(os.path.join(self.stage1_shape_path,f'visibility/view_{vi+1:02d}.npy'))).float() for vi in train_slt]
            self.visibility = [vi.reshape(vi.shape[0],-1) for vi in self.visibility]

        if self.multi_light:
            imgs = []
            for v0, vi in enumerate(train_slt):
                img_vi = []
                for li in self.light_slt[v0]:
                    img = imageio.imread(os.path.join(self.basedir,'{}/view_{:02d}/{:03d}.png'.format(self.im_sub,vi+1,li+1)))
                    img_vi.append(img)
                img_vi = np.array(img_vi).astype(np.float32) / 255.
                img_vi = torch.from_numpy(img_vi).view(img_vi.shape[0],-1,3)*self.object_masks[v0][...,None]
                imgs.append(img_vi)
            self.imgs = imgs
        
        if self.multi_light:
            self.view_idx = np.arange(len(train_slt))
        else:
            self.view_idx = np.arange(sum([len(ln) for ln in self.light_slt]))

    def __len__(self):
        return len(self.view_idx)

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        if not self.multi_light:
            llen = np.array([len(ln) for ln in self.light_slt])
            accu = np.cumsum(llen)
            vidx = np.searchsorted(accu,self.view_idx[idx],side='right')
            lidx = self.view_idx[idx] - accu[vidx-1] if vidx >0 else self.view_idx[idx]
        else:
            vidx = self.view_idx[idx]
            lidx = torch.tensor(np.random.choice(np.arange(len(self.light_slt[idx])), self.light_bs, replace=False)).long() \
                        if self.split=='train' and len(self.light_slt[idx])>=self.light_bs \
                            else torch.tensor(np.arange(len(self.light_slt[idx]))).long()
        vidx_ori = self.train_slt[vidx]

        sample = {
            "object_mask": self.object_masks[vidx],
            "uv": uv,
            "vidx": torch.tensor(vidx),
            "vidx_ori": torch.tensor(vidx_ori),
            "intrinsics": self.KK,
            "lidx": lidx if torch.is_tensor(lidx) else torch.tensor(lidx),
        }
        if self.shape_pregen:
            sample['normal'] = self.normal[vidx]
            sample['points'] = self.points[vidx]
            sample['surface_mask'] = self.surface_mask[vidx]
        sample["light_direction"] = self.light_direction[vidx][lidx]
        sample["gt_normal"] = self.gt_normal[vidx]
        if self.vis_loss:
            sample['visibility'] = self.visibility[vidx][lidx]


        if self.multi_light:
            img = self.imgs[vidx][lidx]*self.object_masks[vidx][None,...,None]
        else:
            img = imageio.imread(os.path.join(self.basedir,'{}/view_{:02d}/{:03d}.png'.format(self.im_sub,vidx_ori+1,self.light_slt[vidx][lidx]+1)))
            img = np.array(img).astype(np.float32) / 255.
            img = torch.from_numpy(img).view(-1,3)*self.object_masks[vidx][...,None]
        ground_truth = {
            "rgb": img
        }

        if self.sampling_idx is not None:
            if self.sample_in_mask:
                pick_list = np.arange(self.total_pixels)[sample['object_mask'].numpy()]
                self.sampling_idx = torch.tensor(np.random.choice(pick_list, min(self.sampling_idx.shape[0], pick_list.shape[0]), replace=False)).long()
            ground_truth["rgb"] = img[:,self.sampling_idx, :] if self.multi_light else img[self.sampling_idx, :] 
            sample["object_mask"] = self.object_masks[vidx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]
            if self.shape_pregen:
                sample['normal'] = self.normal[vidx][self.sampling_idx, :]
                sample['points'] = self.points[vidx][self.sampling_idx, :]
                sample['surface_mask'] = self.surface_mask[vidx][self.sampling_idx]
            if self.vis_loss:
                sample['visibility'] = sample['visibility'][:,self.sampling_idx] if self.multi_light else sample['visibility'][self.sampling_idx]
            sample['sampling_idx'] = self.sampling_idx.clone()

        sample["pose"] = self.poses[vidx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            # self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
            select_inds = random.sample(range(self.total_pixels), sampling_size)
            self.sampling_idx = torch.tensor(select_inds).long()

    def change_sampling_idx_patch(self, N_patch, r_patch=1):
        '''
        :param N_patch: number of patches to be sampled
        :param r_patch: patch size will be (2*r_patch)*(2*r_patch)
        :return:
        '''
        if N_patch == -1:
            self.sampling_idx = None
        else:
            # offsets to center pixels
            H, W = self.img_res
            u, v = np.meshgrid(np.arange(-r_patch, r_patch),
                               np.arange(-r_patch, r_patch))
            u = u.reshape(-1)
            v = v.reshape(-1)
            offsets = v * W + u
            # center pixel coordinates
            u, v = np.meshgrid(np.arange(r_patch, W - r_patch),
                               np.arange(r_patch, H - r_patch))
            u = u.reshape(-1)
            v = v.reshape(-1)
            select_inds = np.random.choice(u.shape[0], size=(N_patch,), replace=False)
            # convert back to original image
            select_inds = v[select_inds] * W + u[select_inds]
            # pick patches
            select_inds = np.stack([select_inds + shift for shift in offsets], axis=1)
            select_inds = select_inds.reshape(-1)
            self.sampling_idx = torch.from_numpy(select_inds).long()

