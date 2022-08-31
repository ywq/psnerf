import os
import torch
from models import model_utils
from utils import eval_utils, time_utils 
import numpy as np
from PIL import Image
import scipy

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, split, loader, models, log, epoch, recorder):
    models[0].eval()
    models[1].eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);
    os.makedirs(os.path.join(log.args.log_dir, 'outimg'),exist_ok=True)
    os.makedirs(os.path.join(log.args.log_dir, 'outnpy'),exist_ok=True)

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    light_dirs, light_ints = [],[]
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            pred_c = models[0](input); timer.updateTime('Forward')
            input.append(pred_c)
            pred = models[1](input); timer.updateTime('Forward')

            recoder, iter_res, error = prepareRes(args, data, pred_c, pred, recorder, log, split)

            res.append(iter_res)
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                results, nrow = prepareSave(args, data, pred_c, pred)
                light_dirs.append(pred_c['dirs'])
                light_ints.append(pred_c['intens'][0,::3])
                out_iter = int(sample['view'][0].split('_')[-1])
                log.saveImgResults(results, split, epoch, out_iter, nrow=nrow, error='')
                log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)
                rstdir = os.path.join(log.args.log_dir, 'outimg/view_%02d.png' % out_iter)
                rst = results[-2][0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                rst0 = np.zeros((sample['imres'][0],sample['imres'][1],3),np.uint8)
                th, tw = sample['crop'][0], sample['crop'][1]
                rst0[th:th+rst.shape[0], tw:tw+rst.shape[1]] = rst
                rst = Image.fromarray(rst0)
                rst.save(rstdir)
                rstdir = os.path.join(log.args.log_dir, 'outnpy/view_%02d.npy' % out_iter)
                norm = pred['n'].data * data['m'].data.expand_as(pred['n'].data)
                norm = norm[0].cpu().permute(1, 2, 0).numpy()
                norm0 = np.zeros((sample['imres'][0],sample['imres'][1],3), np.float32)
                norm0[th:th+norm.shape[0], tw:tw+norm.shape[1]] = norm
                np.save(rstdir, norm0)

            if stop_iters > 0 and iters >= stop_iters: break
    res = np.vstack([np.array(res), np.array(res).mean(0)])
    save_name = '%s_res.txt' % (args.suffix)
    np.savetxt(os.path.join(args.log_dir, split, save_name), res, fmt='%.2f')
    if res.ndim > 1:
        for i in range(res.shape[1]):
            save_name = '%s_%d_res.txt' % (args.suffix, i)
            np.savetxt(os.path.join(args.log_dir, split, save_name), res[:,i], fmt='%.3f')

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

    if args.light_is_same:
        light_dirs = torch.stack(light_dirs, dim=0).detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'light_direction_pred.npy'), light_dirs)
    else:
        light_dirs = [i.detach().cpu().numpy() for i in light_dirs]
        np.save(os.path.join(args.log_dir, 'light_direction_pred.npy'), light_dirs, allow_pickle=True)

    if args.light_is_same:
        light_ints = torch.stack(light_ints, dim=0).detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'light_intensity_pred.npy'), light_ints)
    else:
        light_ints = [i.detach().cpu().numpy() for i in light_ints]
        np.save(os.path.join(args.log_dir, 'light_intensity_pred.npy'), light_ints, allow_pickle=True)


def prepareRes(args, data, pred_c, pred, recorder, log, split):
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred_c['dirs'].data, data_batch)
        recorder.updateIter(split, l_acc.keys(), l_acc.values())
        iter_res.append(l_acc['l_err_mean'])
        error += 'D_%.3f-' % (l_acc['l_err_mean']) 
    if args.s1_est_i:
        int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred_c['intens'].data, data_batch)
        recorder.updateIter(split, int_acc.keys(), int_acc.values())
        iter_res.append(int_acc['ints_ratio'])
        error += 'I_%.3f-' % (int_acc['ints_ratio'])

    if args.s2_est_n:
        acc, error_map = eval_utils.calNormalAcc(data['n'].data, pred['n'].data, data['m'].data)
        recorder.updateIter(split, acc.keys(), acc.values())
        iter_res.append(acc['n_err_mean'])
        error += 'N_%.3f-' % (acc['n_err_mean'])
        data['error_map'] = error_map['angular_map']
    return recorder, iter_res, error

def prepareSave(args, data, pred_c, pred):
    results = [data['img'].data, data['m'].data, (data['n'].data+1) / 2]
    if args.s2_est_n:
        pred_n = (pred['n'].data + 1) / 2
        masked_pred = pred_n * data['m'].data.expand_as(pred['n'].data)
        res_n = [masked_pred, data['error_map']]
        results += res_n

    nrow = data['img'].shape[0]
    return results, nrow
