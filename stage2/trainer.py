import os
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
import math, json
from glob import glob

# imageio.plugins.freeimage.download()


class TrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.max_niters = kwargs['max_niters']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.gamma = kwargs['gamma']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.multi_light = self.conf.get_bool('train.multi_light', default=False)
        self.light_bs = self.conf.get_int('train.light_bs', default=32)
        self.light_train = self.conf.get_bool('train.light_train',default=False)    ## whether use estimated lights or gt lights
        self.light_init = self.conf.get_string('train.light_init',default=None)
        self.light_inten_train = self.conf.get_bool('train.light_inten_train',default=False)
        self.light_inten_init = self.conf.get_string('train.light_inten_init',default=None)
        self.light_decay = self.conf.get_bool('train.light_decay', default=False)
        self.ana_fixlight = self.conf.get_bool('train.ana_fixlight',default=False)

        self.normal_mlp = self.conf.get_bool('train.normal_mlp', default = False)    ## whether use normal_net
        self.normal_joint = self.conf.get_bool('train.normal_joint', default=False)   ## whether fix normal_net or joint train
        self.normal_train = self.normal_mlp and self.normal_joint
        self.shape_pregen = self.conf.get_bool('train.shape_pregen', default=False)
        self.stage1_shape_path = self.conf.get_string('train.stage1_shape_path', default=None)
        
        self.visibility = self.conf.get_bool('train.visibility',default=False)
        self.vis_loss = self.visibility and self.conf.get_bool('train.vis_loss',default=False)

        self.train_order = self.conf.get_bool('train.train_order', default=False)

        self.expname = self.conf.get_string('train.expname')
        self.obj_name = self.conf.get_string('dataset.obj_name')
        self.expdir = os.path.join(self.exps_folder_name, self.obj_name,self.expname)
        
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(self.expdir):
                timestamps = os.listdir(self.expdir)
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.model_params_subdir = "ModelParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"
        os.makedirs(os.path.join(self.checkpoints_path, self.model_params_subdir), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir), exist_ok=True)

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,split='train', multi_light=self.multi_light)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,split='test', multi_light=False)
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf).to(self.device)

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        if self.normal_train:
            self.loss_n = utils.get_class('model.loss.NormalLoss')(**self.conf.get_config('normal.loss'))

        self.sg_optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.conf.get_float('train.sg_learning_rate'))
        sched_ms = self.conf.get_list('train.sg_sched_milestones', default=[])
        sched_ms = [si*len(self.train_dataset) for si in sched_ms]
        if self.multi_light:
            sched_ms = [si*self.light_bs for si in sched_ms]
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.sg_optimizer,
                                                              sched_ms,
                                                              gamma=self.conf.get_float('train.sg_sched_factor', default=0.0))

        if self.light_train:
            self.optimizer_light_params_subdir = "OptimizerLightParameters"
            self.light_params_subdir = "LightParameters"
            os.makedirs(os.path.join(self.checkpoints_path, self.optimizer_light_params_subdir),exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints_path, self.light_params_subdir),exist_ok=True)

            llen = sum([len(ln) for ln in self.train_dataset.light_slt])
            if self.light_init == 'gt':
                self.light_para = torch.nn.Embedding(llen, 3, sparse=True)
                self.light_para.weight.data.copy_(torch.cat(self.train_dataset.light_direction,dim=0))
            elif self.light_init == 'pred':
                estdir = 'sdps_out'
                if self.conf.get_string('dataset.inten_normalize', default=None)=='gt':
                    estdir += '_intnorm_gt'
                if self.train_dataset.light_is_same:
                    estdir += f'_l{self.train_dataset.train_light}'
                self.light_pred_path = os.path.join(self.train_dataset.basedir,estdir,'light_direction_pred.npy')
                assert os.path.exists(self.light_pred_path), "light directions predicted by SDPS-Net not found."
                light_pred = np.load(self.light_pred_path, allow_pickle=True)
                light_pred = [light_pred[vi] for vi in self.train_dataset.train_slt]
                light_pred = [torch.tensor(np.einsum('ij,kj->ki',self.train_dataset.pose0[ln,:3,:3],li).astype(np.float32)) for ln,li in enumerate(light_pred)]
                self.light_para = torch.nn.Embedding(llen, 3, sparse=True)
                self.light_para.weight.data.copy_(torch.cat(light_pred,dim=0))
                self.light_vis_train = [li.clone() for li in light_pred]
            else:
                raise ValueError
            light_para_list = [{'params':list(self.light_para.parameters())}]

            if self.light_inten_train:
                if self.light_inten_init == 'pred':
                    light_inten_pred = np.load(self.light_pred_path.replace('light_direction_pred.npy','light_intensity_pred.npy'),allow_pickle=True)
                    light_inten_pred = [light_inten_pred[vi] for vi in self.train_dataset.train_slt]
                    self.light_inten_para = torch.nn.Embedding(llen, 1, sparse=True)
                    self.light_inten_para.weight.data.copy_(torch.tensor(np.concatenate(light_inten_pred,axis=0)).reshape(-1,1))
                elif self.light_inten_init == 'same':
                    self.light_inten_para = torch.nn.Embedding(llen, 1, sparse=True)
                    torch.nn.init.constant_(self.light_inten_para.weight, self.model.light_int)
                light_para_list += [{'params':list(self.light_inten_para.parameters()), 'lr':self.conf.get_float('train.light_inten_lr', self.conf.get_float('train.light_learning_rate'))}]

            self.light_optimizer = torch.optim.SparseAdam(light_para_list, lr=self.conf.get_float('train.light_learning_rate'))
            self.light_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.light_optimizer, sched_ms,
                                                                gamma=self.conf.get_float('train.sg_sched_factor', default=0.0)) \
                                            if self.light_decay else None

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.light_train:
                data = torch.load(os.path.join(old_checkpnts_dir, self.optimizer_light_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.light_optimizer.load_state_dict(data["optimizer_light_state_dict"])
                if self. light_decay:
                    self.light_scheduler.load_state_dict(data["scheduler_light_state_dict"])
                data = torch.load(os.path.join(old_checkpnts_dir, self.light_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.light_para.load_state_dict(data["light_state_dict"])
                if self.light_inten_train:
                    self.light_inten_para.load_state_dict(data["light_inten_state_dict"])


        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')
        if self.conf.get_bool('train.train_all_pixels', False):
            self.num_pixels = self.total_pixels
        
        self.vis_plus = self.conf.get_bool('train.vis_plus',default=False)
        if self.vis_loss and self.vis_plus:
            self.vis_plus_light = json.load(open(os.path.join(self.stage1_shape_path,'vis_plus/light_dir.json')))
            self.vis_plus_all = {}
            for vi in self.train_dataset.train_slt:
                self.vis_plus_all[f'view_{vi+1:02d}'] = np.load(os.path.join(self.stage1_shape_path,'vis_plus/view_{:02d}.npy'.format(vi+1)), allow_pickle=True)

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

        if self.light_train:
            torch.save(
                {"epoch": epoch, "optimizer_light_state_dict": self.light_optimizer.state_dict(),
                 "scheduler_light_state_dict": self.light_scheduler.state_dict() if self.light_decay else None},
                os.path.join(self.checkpoints_path, self.optimizer_light_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_light_state_dict": self.light_optimizer.state_dict(),
                 "scheduler_light_state_dict": self.light_scheduler.state_dict() if self.light_decay else None},
                os.path.join(self.checkpoints_path, self.optimizer_light_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "light_state_dict": self.light_para.state_dict(), 
                "light_inten_state_dict": self.light_inten_para.state_dict() if self.light_inten_train else self.model.light_int},
                os.path.join(self.checkpoints_path, self.light_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "light_state_dict": self.light_para.state_dict(), 
                "light_inten_state_dict": self.light_inten_para.state_dict() if self.light_inten_train else self.model.light_int},
                os.path.join(self.checkpoints_path, self.light_params_subdir, "latest.pth"))

    def mae_error(self, img1, img2, mask=None, normalize=True):
        if mask is not None:
            img1, img2 = torch.masked_select(img1,mask.to(torch.bool)[:,None]).view(-1,3), torch.masked_select(img2,mask.to(torch.bool)[:,None]).view(-1,3)
        if normalize:
            img1 = F.normalize(img1, dim=-1)
            img2 = F.normalize(img2, dim=-1)
        dot_product = (img1 * img2).sum(-1).clamp(-1, 1)
        angular_err = torch.acos(dot_product) * 180.0 / math.pi
        l_err_mean  = angular_err.mean()
        return l_err_mean, angular_err

    def psnr_error(self, img1, img2, mask=None):
        if mask is not None:
            img1, img2 = torch.masked_select(img1,mask.to(torch.bool)[:,None]).view(-1,3), torch.masked_select(img2,mask.to(torch.bool)[:,None]).view(-1,3)
        mse = ((img1 - img2)**2).mean()
        if mse == 0:
            psnr = 100
        else:
            psnr = - 10.0 * torch.log10(mse)
        return psnr        

    def plot_to_disk(self):
        self.model.eval()
        if self.light_train:
            self.light_para.eval()
        sampling_idx = self.train_dataset.sampling_idx
        self.train_dataset.change_sampling_idx(-1)
        model_out_all, plot_pose, plot_gt, model_input_all = [],[],[],[]
        ## train view + test view
        for pi, (indices, model_input, ground_truth) in enumerate([next(iter(self.train_dataloader)), next(iter(self.plot_dataloader))]):
            ## test view
            if pi==1 and self.light_train and self.conf.get_bool('train.train_all_view',default=False):
                model_input['light_direction'] = F.normalize(self.light_para(torch.tensor(self.plot_dataset.view_idx[indices])),p=2,dim=-1)[None,...].to(self.device)
                if self.light_inten_train:
                    model_input['light_intensity'] = self.light_inten_para(torch.tensor(self.plot_dataset.view_idx[indices])).to(self.device)

            ## train view
            if pi==0 and self.multi_light:
                lslt = torch.randint(0,ground_truth['rgb'].shape[1],(1,))[0]
                model_input['visibility'] = model_input['visibility'][:,lslt]
                ground_truth['rgb'] = ground_truth['rgb'][:,lslt]
                vidx = self.train_dataset.view_idx[indices]
                lidx = model_input['lidx'][:,lslt]
                accu = [len(ln) for ln in self.train_dataset.light_slt]
                l_slt = (sum(accu[:vidx]) + lidx)
                model_input['light_direction'] = F.normalize(self.light_para(l_slt),p=2,dim=-1).to(self.device)
                if self.light_inten_train:
                    model_input['light_intensity'] = self.light_inten_para(l_slt).to(self.device)
            
            for sub_term in model_input.keys():
                model_input[sub_term] = model_input[sub_term].to(self.device)

            split = utils.split_input(model_input, self.total_pixels)
            res = []
            for s in split:
                out = self.model(s)
                res_sub = {k:v.detach() for k,v in out.items()}
                res.append(res_sub)

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            # val
            tsplit = 'train' if pi == 0 else 'test'
            mask = torch.logical_and(model_outputs['network_object_mask'], model_outputs['object_mask'])
            psnr = self.psnr_error(ground_truth['rgb'][0].to(self.device),model_outputs['sg_rgb_values'],mask)
            self.writer.add_scalar(f'{tsplit}.psnr',psnr.item(), self.cur_iter)
            if self.normal_mlp:
                normal_mae, normal_error = self.mae_error(model_input['gt_normal'][0],model_outputs['normal_pred'],mask)
                self.writer.add_scalar(f'{tsplit}.normal_MAE',normal_mae.item(), self.cur_iter)
                normal_error_map = torch.zeros_like(model_outputs['normal_pred'][...,0])
                normal_error_map[mask]=normal_error
                model_outputs['normal_mae'] = normal_error_map

            model_out_all.append(model_outputs)
            plot_gt.append(ground_truth['rgb']) 
            model_input_all.append(model_input)

        plt.plot_micro(self.gamma, 
                model_out_all,
                plot_gt,
                self.plots_dir,
                self.cur_iter,
                self.img_res,
                model_input_all=model_input_all,
                )

        self.model.train()
        if self.light_train:
            self.light_para.train()
        self.train_dataset.sampling_idx = sampling_idx

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.train_dataset.change_sampling_idx(self.num_pixels)
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)
                if self.cur_iter % self.plot_freq == 0:
                    with torch.no_grad():
                        self.plot_to_disk()
                if self.train_order:
                    self.train_fix()

                if self.multi_light:
                    ground_truth['rgb'] = ground_truth['rgb'][0]
                    model_input['light_direction'] = model_input['light_direction'][0]
                    model_input['visibility'] = model_input['visibility'][0]
                if self.light_train:
                    if self.multi_light:
                        vidx = self.train_dataset.view_idx[indices]
                        lidx = model_input['lidx'][0]
                        accu = [len(ln) for ln in self.train_dataset.light_slt]
                        l_slt = sum(accu[:vidx]) + lidx
                    else:
                        l_slt = torch.tensor([self.train_dataset.view_idx[indices]])
                    model_input['light_direction'] = F.normalize(self.light_para(l_slt),p=2,dim=-1).to(self.device)
                    model_input['light_vis_train'] = F.normalize(torch.cat(self.light_vis_train,dim=0)[l_slt],p=2,dim=-1).to(self.device)
                    if self.light_inten_train:
                        model_input['light_intensity'] = self.light_inten_para(l_slt).to(self.device)

                for sub_term in model_input.keys():
                    model_input[sub_term] = model_input[sub_term].to(self.device)

                if self.vis_loss and self.vis_plus:
                    vnum = self.conf.get_int('train.vis_train_num', 16)
                    light_plus = np.array(self.vis_plus_light[f'view_{model_input["vidx_ori"][0]+1:02d}']).astype(np.float32)
                    vis_plus_v = torch.cat([torch.tensor(self.vis_plus_all[f'view_{model_input["vidx_ori"][0]+1:02d}']).reshape(len(light_plus),-1).to(self.device), self.train_dataset.visibility[model_input["vidx"][0]].to(self.device)], dim=0)
                    light_plus = torch.cat([torch.tensor(light_plus).to(self.device),self.light_vis_train[model_input["vidx"][0]].to(self.device)],dim=0)
                    sidx = torch.tensor(np.random.choice(np.arange(len(light_plus)), vnum, replace=False)).long().to(self.device)
                    model_input['light_vis_train'] = light_plus[sidx].to(self.device)
                    assert light_plus.shape[0] == vis_plus_v.shape[0]
                    model_input['vis_train_gt'] = vis_plus_v[sidx].to(self.device)[:,model_input['sampling_idx'][0]]
                        
                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth, model_input)

                loss = loss_output['loss']
                if self.normal_train:
                    loss_normal = self.loss_n(model_outputs)
                    loss += loss_normal['loss']

                self.sg_optimizer.zero_grad()
                if self.light_train and self.light_para.weight.requires_grad:
                    self.light_optimizer.zero_grad()

                loss.backward()

                self.sg_optimizer.step()
                if self.light_train and self.light_para.weight.requires_grad:
                    self.light_optimizer.step()

                if self.cur_iter % 100 == 0:
                    if self.light_train:
                        dot_product = (F.normalize(self.light_para.weight.data.detach().to(self.device),p=2,dim=-1) \
                                        * torch.cat(self.train_dataset.light_direction,dim=0).to(self.device)).sum(-1).clamp(-1, 1)
                        light_direction_error = (torch.acos(dot_product) * 180.0 / math.pi).mean()
                    print(
                        '[{}][{}] ({}/{}): loss = {:.6f}, sg_rgb_loss = {:.6f}, '
                        'sg_lr = {:.6f}, sg_psnr = {:.6f}, '
                        'sg_rgb_weight = {:.2f}, '
                            .format(self.obj_name,self.expname, epoch, self.cur_iter, loss.item(),
                                    loss_output['sg_rgb_loss'].item(),
                                    self.sg_scheduler.get_last_lr()[0],
                                    mse2psnr(loss_output['sg_rgb_loss'].item()),
                                    self.loss.sg_rgb_weight),
                        'albedo_smooth_loss = {:.6f}, '.format(loss_output['albedo_smooth_loss']) if loss_output['albedo_smooth_loss'] is not None else '',
                        'rough_smooth_loss = {:.6f}, '.format(loss_output['rough_smooth_loss']) if loss_output['rough_smooth_loss'] is not None else '',
                        'normal_loss = {:.6f}, '.format(loss_normal['normal_loss']) if self.normal_train else '',
                        'normal_smooth_loss = {:.6f}, '.format(loss_normal['normal_smooth_loss']) if (self.normal_train and loss_normal['normal_smooth_loss'] is not None) else '',
                        'vis_loss = {:.6f}, '.format(loss_output['vis_loss']) if self.vis_loss and 'vis_loss' in [*loss_output] else '',
                        'light_direction_error = {:.6f}, '.format(light_direction_error.item()) if self.light_train else '',
                        'light_inten_mean = {:.6f}, '.format(self.light_inten_para.weight.data.detach().mean()) if self.light_inten_train else '',
                        'light_inten_iter = {:.6f}, '.format(self.light_inten_para(l_slt).mean().item()) if self.light_inten_train else '',
                        'light_lr = {:.6f}, '.format(self.light_scheduler.get_last_lr()[0]) if self.light_train and self.light_decay else '',
                        )

                    self.writer.add_scalar('sg_rgb_loss', loss_output['sg_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('sg_psnr', mse2psnr(loss_output['sg_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('sg_rgb_weight', self.loss.sg_rgb_weight, self.cur_iter)
                    self.writer.add_scalar('sg_lrate', self.sg_scheduler.get_last_lr()[0], self.cur_iter)
                    if loss_output['albedo_smooth_loss'] is not None:
                        self.writer.add_scalar('albedo_smooth_loss',loss_output['albedo_smooth_loss'].item(), self.cur_iter)
                    if loss_output['rough_smooth_loss'] is not None:
                        self.writer.add_scalar('rough_smooth_loss',loss_output['rough_smooth_loss'].item(), self.cur_iter)
                    if self.light_train:
                        self.writer.add_scalar('light_direction_error',light_direction_error.item(), self.cur_iter)
                        if self.light_decay:
                            self.writer.add_scalar('light_lr', self.light_scheduler.get_last_lr()[0], self.cur_iter)
                    if self.normal_train:
                        self.writer.add_scalar('normal_loss', loss_normal['normal_loss'].item(), self.cur_iter)
                        if loss_normal['normal_smooth_loss'] is not None:
                            self.writer.add_scalar('normal_smooth_loss',loss_normal['normal_smooth_loss'].item(), self.cur_iter)
                    if self.vis_loss and 'vis_loss' in [*loss_output]:
                        self.writer.add_scalar('visibility_loss',loss_output['vis_loss'].item(), self.cur_iter)
                    if self.light_inten_train:
                        self.writer.add_scalar('light_inten_mean',self.light_inten_para.weight.data.detach().mean().item(), self.cur_iter)
                        self.writer.add_scalar('light_inten',self.light_inten_para(l_slt).mean().item(), self.cur_iter)


                self.cur_iter += 1

                self.sg_scheduler.step()
                if self.light_decay and self.light_para.weight.requires_grad:
                    self.light_scheduler.step()

            # clear plot/save files
            freq_list = [xi*5 for xi in range(10)]
            for ckpt_sub in os.listdir(self.checkpoints_path):
                ckpt_sub_dir = os.path.join(self.checkpoints_path, ckpt_sub)
                fall = sorted(os.listdir(ckpt_sub_dir),key=lambda x:int(x.split('.')[0]) if x.split('.')[0]!='latest' else 0)
                for fsub in fall[:-2]:
                    if fsub.split('.')[0] != 'latest':
                        fidx = np.ceil(int(fsub.split('.')[0])*len(self.train_dataset)/self.ckpt_freq) 
                        if fidx not in freq_list and fidx % 10 !=0:
                            os.remove(os.path.join(ckpt_sub_dir, fsub))
            freq_list = [xi for xi in range(10)] + [xi*5 for xi in range(10)]
            fall = sorted(os.listdir(self.plots_dir),key=lambda x:int(x.split('.')[0].split('_')[-1]))
            for fsub in fall[:-5]:
                pidx = int(fsub.split('.')[0].split('_')[-1])//self.plot_freq 
                if pidx not in freq_list and pidx%10!=0:
                    os.remove(os.path.join(self.plots_dir, fsub))


    
    def train_fix(self,):
        if self.cur_iter == 0:
            self.ori_sg_rgb_weight = self.loss.sg_rgb_weight
            self.ori_albedo_smooth_weight = self.loss.albedo_smooth_weight
            self.ori_rough_smooth_weight = self.loss.rough_smooth_weight
            self.ori_vis_weight = self.loss.vis_weight
            # print(self.ori_sg_rgb_weight, self.ori_albedo_smooth_weight, self.ori_rough_smooth_weight, self.ori_vis_weight )
            self.loss.sg_rgb_weight = 0
            self.loss.albedo_smooth_weight = 0
            self.loss.rough_smooth_weight = 0
            self.loss.vis_weight = 10
            self.model.albedo_net = self.model.albedo_net.eval().requires_grad_(False)
            self.model.rough_net = self.model.rough_net.eval().requires_grad_(False)
            if self.visibility and not self.vis_loss:
                self.model.visibility_net = self.model.visibility_net.eval().requires_grad_(False)
            if self.light_train:
                self.light_para = self.light_para.requires_grad_(False)
                if self.light_inten_train:
                    self.light_inten_para = self.light_inten_para.requires_grad_(False)
        elif self.cur_iter == 5000:
            self.loss.sg_rgb_weight = self.ori_sg_rgb_weight
            self.loss.albedo_smooth_weight = self.ori_albedo_smooth_weight
            self.loss.rough_smooth_weight = self.ori_rough_smooth_weight
            self.loss.vis_weight = self.ori_vis_weight
            self.model.albedo_net = self.model.albedo_net.train().requires_grad_(True)
            self.model.rough_net = self.model.rough_net.train().requires_grad_(True)
            if not self.ana_fixlight and self.light_train:
                self.light_para = self.light_para.requires_grad_(True)
                if self.light_inten_train:
                    self.light_inten_para = self.light_inten_para.requires_grad_(True)