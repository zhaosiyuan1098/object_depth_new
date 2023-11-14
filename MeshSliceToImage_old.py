# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

import numpy as np
import open3d as o3d
from pytransform3d import transformations as pt
from skspatial.objects import Plane
import sigpy.plot as pl
import trimesh
import numbers
import SimpleITK as sitk
from os import path, listdir


def getFiles(folder):
    """Get all files in the input folder.

    Args:
        folder (str): Path to target folder.

    Returns:
        list: files (folder+/+filename) list.
        
        
    Example:
        ```python
        print(getFiles('.')) # ['./.gitignore', './BaseRecon.py']
        ```
    """
    onlyfiles = [(folder + '/' + f) for f in listdir(folder)
                 if path.isfile(path.join(folder + '/' + f))]
    return onlyfiles


def o3d_create_mesh():
    pcd = o3d.io.read_point_cloud('data/EaglePointCloud.ply')
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    print('mesh', mesh)
    if True:
        o3d.visualization.draw_geometries([mesh],
                                          zoom=0.664,
                                          front=[-0.4761, -0.4698, -0.7434],
                                          lookat=[1.8900, 3.2596, 0.9284],
                                          up=[0.2304, -0.8825, 0.4101])


def trimesh_example():
    mesh = trimesh.load_mesh('data/BunnyMesh.ply')
    # [-0.02679343  0.09413622  0.00829883]
    print(mesh, mesh.centroid)
    # plane is a infinite plane, translation inside plane would
    # not change the final section result.
    slice = mesh.section(plane_origin=mesh.centroid, plane_normal=[0, 1, 0])
    print(slice)
    # slice.show()

    slice_2D, to_3D = slice.to_planar()

    slice_2D.simplify_spline()
    slice_2D.fill_gaps()
    slice_2D.remove_duplicate_entities()
    slice_2D.remove_invalid()
    slice_2D.remove_unreferenced_vertices()

    print(slice_2D, slice_2D.area, slice_2D.is_closed, slice_2D.length,
          slice_2D.extents, slice_2D.bounds)
    print(to_3D)
    slice_2D.show()

    origin = slice_2D.bounds[0]
    pitch = slice_2D.extents.max() / 100
    resolution = [100, 100]

    # https://github.com/mikedh/trimesh/issues/1569
    # pitch: float or (2,) float, length(s) in model space of pixel edges
    im = slice_2D.rasterize(pitch, origin, resolution, fill=True)
    pl.ImagePlot(np.asarray(im))

    return im, to_3D


def slice_mesh_to_image(mesh, mat, fovLen, spacing, show=False):
    """Slice mesh to 2d image based on certain plane (mat) function.

    Args:
        mesh (_type_): _description_
        mat ((4, 4)): plane transformation matrix.
        fovLen (float or (2,) float):  length(s) in model space of pixel edges
        spacing (float or (2,) float): spacing(s) in model space
        show (bool, optional): show sliced mesh path and image. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: sliced image (numpy array).
    """
    if isinstance(fovLen, numbers.Number):
        fovLen = [fovLen, fovLen]
    fovLen = np.asanyarray(fovLen, dtype=np.float64)

    if isinstance(spacing, numbers.Number):
        spacing = [spacing, spacing]
    spacing = np.asanyarray(spacing, dtype=np.float64)

    # physical metric should be consistent among points, origin, fov, spacing...
    p_normal = pt.transform(mat, pt.vector_to_direction([0, 0, 1]))[:3]
    print('Plane normal:', p_normal)

    # plane is a infinite plane, translation inside plane would
    # not change the final section result.
    slice = mesh.section(plane_origin=mat[:3, 3], plane_normal=p_normal)

    if slice is None:
        raise ValueError('Invalid slicing: ', mat[:3, 3], p_normal)
    print('Slice: centroid', slice.centroid, '. bounds', slice.bounds,
          '. extents', slice.extents, '. length', slice.length)

    # to_2D is the matrix that transforms mesh points from original coordinate
    # onto the target coordinate, so it is reversed to the input mat.
    slice_2D, _ = slice.to_planar(to_2D=pt.invert_transform(mat))
    slice_2D.simplify_spline()
    slice_2D.fill_gaps()
    slice_2D.remove_duplicate_entities()
    slice_2D.remove_invalid()
    slice_2D.remove_unreferenced_vertices()
    if show:
        slice_2D.show()

    resolution = (fovLen / spacing).astype(int)
    pitch = spacing

    # warning: section plane is originated at mat[:3, 3], so points coordinates
    # in slice_2d are based on mat[:3, 3], which means that the final crop
    # image's origin should always be [0, 0].
    # when translation of matrix is movement of center instead of origin, then
    # we should use picOrigin = -fovLen / 2
    picOrigin = [0, 0]

    # https://github.com/mikedh/trimesh/issues/1569
    im = slice_2D.rasterize(pitch, picOrigin, resolution, fill=True)
    print('final image:', np.shape(np.asarray(im)))

    if show:
        pl.ImagePlot(np.asarray(im))
    return im, resolution


def mesh_to_dicom(mesh, dcmFiles):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dcmFiles)
    image = reader.Execute()
    size = image.GetSize()
    fov = np.array(image.GetSize()) * np.array(image.GetSpacing())
    spacing = image.GetSpacing()
    mat = np.reshape(image.GetDirection(), (3, 3))
    print("Image size:", size)
    print("Image spacing:", spacing)
    print("Image origin:", image.GetOrigin()) 
    print("Image mat:", mat)
    print("Image fov:", fov)

    tf = pt.transform_from(mat, image.GetOrigin())
    mask, _ = slice_mesh_to_image(mesh, tf, fov[:2], spacing[:2], True)
    maskedImage = mask * sitk.GetArrayFromImage(image)
    pl.ImagePlot(maskedImage)


if __name__ == "__main__":
    if False:
        mesh = trimesh.load_mesh('data/BunnyMesh.ply')
        # unit in bunny mesh file is meter, we need to convert it to millimeter,
        # which is the default unit for all dicom files.
        mesh.apply_scale(1000)
        print('mesh: centroid', mesh.centroid, '. bounds', mesh.bounds,
              '. extents', mesh.extents)
        # mesh.show()

    folder = 'data/20221222/'
    head_full_in_display = trimesh.load_mesh(folder +
                                             '/head_full_in_display2.ply')
    dcm_files = getFiles(folder + '/T1SE')
    mesh_to_dicom(head_full_in_display, dcm_files)

    if False:
        matrix = np.load('data/regtest/0711/lineK/matrix.npy')
        # warning: translation of tf should be movement of origin
        # instead of center, so be careful when we passing center
        # as translation, we should also change picture origin inside
        # slice_mesh_to_image from [0, 0] to -fovLen/2.
        tf = pt.transform_from(matrix[3], mesh.centroid)
        im, _ = slice_mesh_to_image(mesh, tf, [0.1, 0.3], [0.0001], True)
