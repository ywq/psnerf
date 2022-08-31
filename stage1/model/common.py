import torch
import numpy as np
import logging
from copy import deepcopy


logger_py = logging.getLogger(__name__)

def sample_patch_points(batch_size, n_points, patch_size=1,
                        image_resolution=(128, 128), 
                        sensor_size=np.array([[-1, 1],[-1, 1]]), 
                        continuous=True):
    ''' Returns sampled points in the range of sensorsize.

    Args:
        batch_size (int): required batch size
        n_points (int): number of points to sample
        patch_size (int): size of patch; if > 1, patches of size patch_size
            are sampled instead of individual points
        image_resolution (tuple): image resolution (required for calculating
            the pixel distances)
        continuous (bool): whether to sample continuously or only on pixel
            locations
    '''
    assert(patch_size > 0)

    n_patches = int(n_points)

    if continuous:
        p = torch.rand(batch_size, n_patches, 2)  # [0, 1]
    else:
        px = torch.randint(0, image_resolution[1], size=(
            batch_size, n_patches, 1)).float() 
        py = torch.randint(0, image_resolution[0], size=(
            batch_size, n_patches, 1)).float() 
        p = torch.cat([px, py], dim=-1)

    p = p.view(batch_size, n_patches, 1, 2) 
    p = p.view(batch_size, -1, 2)
    pix = p.clone()

    p[:,:,0] *= (sensor_size[1,1] - sensor_size[1,0]) / (image_resolution[1] - 1)
    p[:,:,1] *= (sensor_size[0,1] - sensor_size[0,0])  / (image_resolution[0] - 1)
    p[:,:,0] += sensor_size[1,0] 
    p[:,:,1] += sensor_size[0,0]

    assert(p.max() <= sensor_size.max())
    assert(p.min() >= sensor_size.min())
    assert(pix.max() < max(image_resolution))
    assert(pix.min() >= 0)

    return p, pix


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0])/ 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()

    return mask


def transform_mesh(mesh, transform):
    ''' Transforms a mesh with given transformation.

    Args:
        mesh (trimesh mesh): mesh
        transform (tensor): transformation matrix of size 4 x 4
    '''
    mesh = deepcopy(mesh)
    v = np.asarray(mesh.vertices).astype(np.float32)
    v_transformed = transform_pointcloud(v, transform)
    mesh.vertices = v_transformed
    return mesh


def transform_pointcloud(pointcloud, transform):
    ''' Transforms a point cloud with given transformation.

    Args:
        pointcloud (tensor): tensor of size N x 3
        transform (tensor): transformation of size 4 x 4
    '''

    assert(transform.shape == (4, 4) and pointcloud.shape[-1] == 3)

    pcl, is_numpy = to_pytorch(pointcloud, True)
    transform = to_pytorch(transform)

    # Transform point cloud to homogen coordinate system
    pcl_hom = torch.cat([
        pcl, torch.ones(pcl.shape[0], 1)
    ], dim=-1).transpose(1, 0)

    # Apply transformation to point cloud
    pcl_hom_transformed = transform @ pcl_hom

    # Transform back to 3D coordinates
    pcl_out = pcl_hom_transformed[:3].transpose(1, 0)
    if is_numpy:
        pcl_out = pcl_out.numpy()

    return pcl_out


def get_tensor_values(tensor, pe, grid_sample=True, mode='nearest',
                      with_mask=False, squeeze_channel_dim=False):
    '''
    Returns values from tensor at given location p.

    Args:
        tensor (tensor): tensor of size B x C x H x W
        p (tensor): position values scaled between [-1, 1] and
            of size B x N x 2
        grid_sample (boolean): whether to use grid sampling
        mode (string): what mode to perform grid sampling in
        with_mask (bool): whether to return the mask for invalid values
        squeeze_channel_dim (bool): whether to squeeze the channel dimension
            (only applicable to 1D data)
    '''

    batch_size, _, h, w = tensor.shape

    p = pe.clone().detach()
    p[:, :, 0] = 2.*p[:, :, 0]/w - 1
    p[:, :, 1] = 2.*p[:, :, 1]/h - 1
    p = p.unsqueeze(1)
    # tensor = tensor.permute(0, 1, 3, 2)
    values = torch.nn.functional.grid_sample(tensor, p, mode=mode,align_corners=True)
    values = values.squeeze(2).detach()
    values = values.permute(0, 2, 1)

    if squeeze_channel_dim:
        values = values.squeeze(-1)

    return values


def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=False):
    p_world = world_mat[:,:3,-1].unsqueeze(1).repeat(1,n_points,1)
    return p_world


def image_points_to_ray(image_points, camera_mat, world_mat):
    # image_points (tensor):  B x N x 2
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    depth = torch.ones(batch_size, n_pts, 1).to(device)
    pixels, is_numpy = to_pytorch(image_points, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    p_trans = (pixels-camera_mat[0,:2,2])/camera_mat[0,0,0]
    p_trans = torch.cat([p_trans, torch.ones_like(p_trans[...,:1])], dim=2) 
    p_trans[:,:,:3] *= depth
    ray_dirs = torch.einsum('bij,bnj->bni',world_mat[:,:3,:3], p_trans)
    if is_numpy:
        ray_dirs = ray_dirs.numpy()
    return ray_dirs
 

def check_weights(params):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in params.items():
        if torch.isnan(v).any():
            logger_py.warn('NaN Values detected in model weight %s.' % k)


def check_tensor(tensor, tensorname='', input_tensor=None):
    ''' Checks tensor for illegal values.

    Args:
        tensor (tensor): tensor
        tensorname (string): name of tensor
        input_tensor (tensor): previous input
    '''
    if torch.isnan(tensor).any():
        logger_py.warn('Tensor %s contains nan values.' % tensorname)
        if input_tensor is not None:
            logger_py.warn('Input was:', input_tensor)

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
    ''' Normalizes the tensor.

    Args:
        tensor (tensor): tensor
        min_norm (float): minimum norm for numerical stability
        feat_dim (int): feature dimension in tensor (default: -1)
    '''
    norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True),
                              min=min_norm)
    normed_tensor = tensor / norm_tensor
    return normed_tensor
