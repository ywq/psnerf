import os, sys
import torch
import numpy as np
from PIL import Image
import math
import imageio
import cv2


# envmap
def load_light(path, light_h=None):
    ext = os.path.basename(path).split('.')[-1]
    if ext == 'exr':
        arr = read_exr(path)
    elif ext == 'hdr':
        arr = read_hdr(path)
    else:
        raise NotImplementedError(ext)
    if light_h:
        arr = cv2.resize(arr, (2*light_h, light_h), interpolation=cv2.INTER_LINEAR)
    return arr

def read_exr(path):
    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise RuntimeError(f"Failed to read\n\t{path}")
    # RGB
    if arr.ndim == 3 or arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb
    raise NotImplementedError(arr.shape)

def read_hdr(path):
    with open(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def write_hdr(rgb, outpath):
    # Writes a ``float32`` RGB array as an HDR map to disk.
    assert rgb.dtype == np.float32, "Input must be float32"
    os.makedirs(os.path.dirname(outpath),exist_ok=True)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(outpath, bgr)
    assert success, "Writing HDR failed"

def vis_light(light_probe, outpath=None, h=None):
    # In case we are predicting too low of a resolution
    if h is not None:
        light_probe = cv2.resize(light_probe, (2*h, h), interpolation=cv2.INTER_NEAREST)
    # Tonemap
    tonemap = lambda hdr, gamma: (hdr / hdr.max()) ** (1 / gamma)
    img = tonemap(light_probe, gamma=4) # [0, 1]
    img_uint = (img*255).astype(np.uint8)
    # Optionally, write to disk
    if outpath is not None:
        Image.fromarray(img_uint).save(outpath)
    return img_uint


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas

def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        print(
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians")

def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def uniform_sample_sph(n, r=1, convention='lat-lng'):
    r"""Uniformly samples points on the sphere
    [`source <https://mathworld.wolfram.com/SpherePointPicking.html>`_].

    Args:
        n (int): Total number of points to sample. Must be a square number.
        r (float, optional): Radius of the sphere. Defaults to :math:`1`.
        convention (str, optional): Convention for spherical coordinates.
            See :func:`cart2sph` for conventions.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians. The points are ordered such that all azimuths are looped
        through first at each elevation.
    """
    n_ = np.sqrt(n)
    if n_ != int(n_):
        raise ValueError("%d is not perfect square" % n)
    n_ = int(n_)

    pts_r_theta_phi = []
    for u in np.linspace(0, 1, n_):
        for v in np.linspace(0, 1, n_):
            theta = np.arccos(2 * u - 1) # [0, pi]
            phi = 2 * np.pi * v # [0, 2pi]
            pts_r_theta_phi.append((r, theta, phi))
    pts_r_theta_phi = np.vstack(pts_r_theta_phi)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = _convert_sph_conventions(
            pts_r_theta_phi, 'theta-phi_to_lat-lng')
    elif convention == 'theta-phi':
        pts_sph = pts_r_theta_phi
    else:
        raise NotImplementedError(convention)

    return pts_sph


def cart2sph(pts_cart, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    pts_cart = np.array(pts_cart)

    # Validate inputs
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cart), axis=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = np.arctan2(y, x) # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_sph = pts_sph.reshape(3)

    return pts_sph


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(
            pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart

