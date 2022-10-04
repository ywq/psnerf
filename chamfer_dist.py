import os
import numpy as np
import json, argparse
import trimesh
import trimesh.proximity
import trimesh.sample


# Arguments
parser = argparse.ArgumentParser(
    description='Evaluation'
)
parser.add_argument('--mesh_gt', type=str, required=True,)
parser.add_argument('--mesh_pred', type=str, required=True,)
parser.add_argument('--num_samples', type=int, default=10000,)
args = parser.parse_args()


def get_chamfer_dist(src_mesh, tgt_mesh, num_samples=10000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist_mean = src_tgt_dist.mean()
    tgt_src_dist_mean = tgt_src_dist.mean()

    chamfer_dist = (src_tgt_dist_mean + tgt_src_dist_mean) / 2

    return chamfer_dist


mesh_gt = trimesh.load(args.mesh_gt)
mesh_pred = trimesh.load(args.mesh_pred)
chamfer = get_chamfer_dist(mesh_pred, mesh_gt, num_samples=args.num_samples)
print(f'Chamfer Distance (mm):  {chamfer*1000:.2f}')
