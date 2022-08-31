import sys,os
import argparse
import GPUtil

from trainer import TrainRunner

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--exps_folder_name', type=str, default='out')
    parser.add_argument('--gamma', type=float, default=1., help='inverse gamma correction coefficient')

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--max_niter', type=int, default=200001, help='max number of iterations to train for')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')


    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    runner = TrainRunner
    trainrunner = runner(conf=opt.conf,
                        gamma=opt.gamma,
                        batch_size=opt.batch_size,
                        nepochs=opt.nepoch,
                        max_niters=opt.max_niter,
                        gpu_index=gpu,
                        exps_folder_name=opt.exps_folder_name,
                        is_continue=opt.is_continue,
                        timestamp=opt.timestamp,
                        checkpoint=opt.checkpoint,
                        )

    trainrunner.run()
