import os
import argparse
import time
import torch
from dataloading import load_config
from model.checkpoints import CheckpointIO
from model.network import NeuralNetwork
from model.extracting import Extractor3D

torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--upsampling-steps', type=int, default=-1,
                    help='Overrites the default upsampling steps in config')
parser.add_argument('--refinement-step', type=int, default=-1,
                    help='Overrites the default refinement steps in config')
parser.add_argument('--obj_name', type=str, default='bunny',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--exp_folder', type=str, default='out',)
parser.add_argument('--test_out_dir', type=str, default='test_out',)
parser.add_argument('--load_iter', type=int, default=None)
parser.add_argument('--mesh_extension', type=str, default='obj')
parser.add_argument('--clip', action='store_true', default=False,
                    help='clip the bottom area')

args = parser.parse_args()
cfg = load_config(os.path.join(args.exp_folder,args.obj_name, args.expname, 'config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

if args.upsampling_steps != -1:
    cfg['extraction']['upsampling_steps'] = args.upsampling_steps
if args.refinement_step != -1:
    cfg['extraction']['refinement_step'] = args.refinement_step

test_out_path = os.path.join(args.test_out_dir,args.obj_name, args.expname)
os.makedirs(test_out_path,exist_ok=True)

# Model
model = NeuralNetwork(cfg)
out_dir = os.path.join(args.exp_folder, args.obj_name, args.expname)
checkpoint_io = CheckpointIO(os.path.join(out_dir,'models'), model=model)
checkpoint_io.load(f'model_{args.load_iter}.pt' if args.load_iter else 'model.pt')
 
# Generator
generator = Extractor3D(
    model, resolution0=cfg['extraction']['resolution'], 
    upsampling_steps=cfg['extraction']['upsampling_steps'], 
    device=device
)
# Generate
model.eval()

try:
    t0 = time.time()
    out = generator.generate_mesh(mask_loader=None,clip=args.clip)

    try:
        mesh, stats_dict = out
    except TypeError:
        mesh, stats_dict = out, {}

    mesh_out_file = os.path.join(
        test_out_path, f'mesh.{args.mesh_extension}')
    mesh.export(mesh_out_file)

except RuntimeError:
    print("Error generating mesh")

