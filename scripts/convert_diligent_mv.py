import os
import json
import argparse
from shutil import copy2
import numpy as np
from PIL import Image
from scipy.io import loadmat


def load_calibration(calib_dir):
    """Load intrinsics, extrinsics and light directions from DiLiGenT-MV calib files."""
    K_path = os.path.join(calib_dir, 'intrinsics.txt')
    if os.path.exists(K_path):
        K = np.loadtxt(K_path).reshape(3, 3)
    else:
        # fallback to .mat file
        mat_path = os.path.join(calib_dir, 'intrinsics.mat')
        if os.path.exists(mat_path):
            K = loadmat(mat_path)['K']
        else:
            raise FileNotFoundError('Cannot find camera intrinsics')

    pose_path = os.path.join(calib_dir, 'extrinsics.txt')
    if os.path.exists(pose_path):
        poses = np.loadtxt(pose_path).reshape(-1, 4, 4)
    else:
        mat_path = os.path.join(calib_dir, 'extrinsics.mat')
        if os.path.exists(mat_path):
            poses = loadmat(mat_path)['pose_c2w']
        else:
            raise FileNotFoundError('Cannot find camera extrinsics')

    light_path = os.path.join(calib_dir, 'light_directions.txt')
    if os.path.exists(light_path):
        light_dirs = np.loadtxt(light_path)
    else:
        mat_path = os.path.join(calib_dir, 'light_directions.mat')
        if os.path.exists(mat_path):
            light_dirs = loadmat(mat_path)['light_direction']
        else:
            raise FileNotFoundError('Cannot find light directions')
    return K, poses, light_dirs


def copy_images(src_dir, dst_dir):
    view_dirs = sorted([d for d in os.listdir(src_dir) if d.lower().startswith('view')])
    if not view_dirs:
        raise RuntimeError('No view directories found')
    os.makedirs(os.path.join(dst_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'mask'), exist_ok=True)

    im_hw = None
    for v in view_dirs:
        src_view = os.path.join(src_dir, v)
        dst_view = os.path.join(dst_dir, 'img', v)
        os.makedirs(dst_view, exist_ok=True)
        for img_name in sorted(os.listdir(src_view)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            copy2(os.path.join(src_view, img_name), os.path.join(dst_view, img_name))
            if im_hw is None:
                with Image.open(os.path.join(src_view, img_name)) as img:
                    im_hw = img.size[::-1]

        mask_path = os.path.join(src_dir, 'mask', f'{v}.png')
        if os.path.exists(mask_path):
            copy2(mask_path, os.path.join(dst_dir, 'mask', f'{v}.png'))
    return len(view_dirs), im_hw


def main():
    parser = argparse.ArgumentParser(description='Convert DiLiGenT-MV data to PS-NeRF format')
    parser.add_argument('src', type=str, help='Path to DiLiGenT-MV object directory')
    parser.add_argument('dst', type=str, help='Destination directory')
    parser.add_argument('--name', type=str, default=None, help='Object name')
    args = parser.parse_args()

    obj_name = args.name or os.path.basename(os.path.normpath(args.dst))

    K, poses, light_dirs = load_calibration(os.path.join(args.src, 'calib'))
    n_light = light_dirs.shape[0]

    n_view, imhw = copy_images(os.path.join(args.src, 'images'), args.dst)

    para = {
        'obj_name': obj_name,
        'n_view': int(n_view),
        'imhw': list(map(int, imhw)),
        'gt_normal_world': False,
        'view_train': list(range(n_view)),
        'view_test': [],
        'K': K.tolist(),
        'pose_c2w': poses.tolist(),
        'light_is_same': True,
        'light_direction': light_dirs.tolist(),
    }

    with open(os.path.join(args.dst, 'params.json'), 'w') as f:
        json.dump(para, f, indent=2)
    print('Converted object saved to', args.dst)


if __name__ == '__main__':
    main()
