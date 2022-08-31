import os
import logging
import torch
from torch.utils import data
import numpy as np
import imageio, json
from scipy import ndimage

logger = logging.getLogger(__name__)


def get_dataloader(cfg, mode='train', shuffle=True,):
    
    batch_size = cfg['dataloading']['batchsize']
    n_workers = cfg['dataloading']['n_workers']

    dataset = Shapes3dDataset(split=mode, cfg=cfg,)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, 
        shuffle=shuffle, collate_fn=collate_remove_none,
    )

    return dataloader



class Shapes3dDataset(data.Dataset):
    def __init__(self, split='train', cfg=None,):
        # Attributes
        basedir = cfg['dataloading']['data_dir']
        white_background = cfg['rendering']['white_background']
        self.est_norm = cfg['training'].get('est_norm', False)
        self.normal_loss = cfg['training'].get('normal_loss', False)
        para = json.load(open(os.path.join(basedir,'params.json')))
        self.para = para
        n_view = para['n_view']
        self.train_view = cfg['dataloading'].get('train_view',None)
        if split=='train':
            train_slt = np.array(para[f'view_slt_{self.train_view}']) if self.train_view is not None else np.array(para['view_train'])
        elif split=='test':
            train_slt = np.array(para['view_test'])
        elif split=='all':
            train_slt = np.array(para[f'view_slt_{self.train_view}']) if self.train_view is not None else np.array(para['view_train'])
            train_slt = np.concatenate([train_slt,np.array(para['view_test'])], axis=0)
            train_slt.sort(axis=0)
        else:
            raise ValueError
        if cfg['dataloading'].get('all_view',False):
            train_slt = np.arange(n_view)
        self.KK = np.array(para['K']).astype(np.float32)
        # camera to world matrix, OpenGL ==> OpenCV
        poses = np.array(para['pose_c2w']).astype(np.float32) # all views
        self.pose0 = poses.copy()   # openGL coordinate system
        self.poses = poses.copy()  
        self.poses[:,:3,1:3]*=-1.   # openCV coordinate system

        self.poses = self.poses[train_slt]
        self.pose0 = self.pose0[train_slt]
        self.train_slt = train_slt   # original index of views

        im_sub = 'img'
        im_type = 'avg'
        self.est_norm_dir = os.path.join(basedir,'sdps_out')
        inten_normalize = cfg['dataloading'].get('inten_normalize',None)
        if para['light_is_same']:
            n_light = len(para['light_direction'])
            train_light = cfg['dataloading'].get('train_light',n_light)
            if inten_normalize in ['gt']:
                self.est_norm_dir += '_intnorm_gt'
            self.est_norm_dir += f"_l{train_light}"
        assert os.path.exists(self.est_norm_dir)
        if inten_normalize is not None:
            assert inten_normalize in ['gt','sdps']
            im_sub += '_intnorm_'+inten_normalize
        if para['light_is_same']:
            if inten_normalize in ['sdps']:
                im_sub += f'_l{train_light}'
            else:
                im_type += f'_l{train_light}'

        imgs, masks, normals, norm_mask, mask_valid = [],[],[],[],[]
        for vi in train_slt:
            imgs.append(imageio.imread(os.path.join(basedir,'{}/{}/view_{:02d}.png'.format(im_sub,im_type,vi+1))))
            mask = np.array(imageio.imread(os.path.join(basedir,'mask/view_{:02d}.png'.format(vi+1))))
            masks.append(mask)
            norm_mask.append(imageio.imread(os.path.join(basedir,'norm_mask/view_{:02d}.png'.format(vi+1))))
            if cfg['training'].get('mask_valid', False):
                maskd = ndimage.binary_dilation(mask,iterations=2)
                maske = ndimage.binary_erosion(mask,iterations=2)
                mask_valid.append(~np.logical_xor(maskd, maske))
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        masks = (np.array(masks) / 255.).astype(np.float32) 
        imh, imw = masks.shape[-2:]
        self.mask_valid = mask_valid

        if self.normal_loss:
            self.norm_mask = (np.array(norm_mask) / 255.).astype(np.float32) 
            for vi in train_slt:
                if self.est_norm:
                    norm = np.load(os.path.join(self.est_norm_dir,'outnpy/view_{:02d}.npy'.format(vi+1)))
                    normals.append(norm)
                else:
                    raise ValueError
            self.normals = np.array(normals).astype(np.float32).transpose(0,3,1,2)
            if cfg['training'].get('mask_black',False):
                self.norm_mask[(imgs<0.1).all(-1)]=False

        if white_background:
            imgs = imgs*masks[...,None] + (1.-masks[...,None])
        self.imgs = imgs.transpose(0,3,1,2)    ##  N,C,H,W
        self.masks = masks

        self.view_idx = np.arange(int(imgs.shape[0]))   # sorted index of the selected views
        print(f"{split}_view: {len(train_slt)}{' , light is same' if para['light_is_same'] else ''}" )

    def __len__(self):
        return len(self.view_idx)

    def __getitem__(self, idx):
        data = {}
        img_idx = self.view_idx[idx]
        data['img']=self.imgs[img_idx]
        data['img.idx']=img_idx
        data['img.world_mat'] = self.poses[img_idx]
        data['img.camera_mat'] = self.KK
        data['img.scale_mat'] = np.eye(4,dtype=np.float32)
        data['img.mask'] = self.masks[img_idx]
        if self.normal_loss:
            data['img.normal'] = self.normals[img_idx]
            data['img.norm_mask'] = self.norm_mask[img_idx]
        if len(self.mask_valid)>0:
            data['img.mask_valid'] = self.mask_valid[img_idx]
        return data


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

