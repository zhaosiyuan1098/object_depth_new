# Copyright (C) USTC BMEC RFLab - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import open3d as o3d
import utils


def get_coil_object_from_bag(bag_file, show_plt=True):
    depth_image, color_image, intrinsics, depth_scale = utils.read_bag(
        bag_file)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy)

    coil_pcd, object_pcd = utils.get_coil_face(depth_image,
                                               color_image,
                                               pinhole_camera_intrinsic,
                                               depth_scale,
                                               object_max_n=3,
                                               show_plt=show_plt)
    return coil_pcd, object_pcd


coil_pcds = []
object_pcd = []

rebuild = False

if rebuild:
    bags = []
    bags.append('./data/JTH/2021_09_14_21_07_22_aligned_frames2624.bag')
    bags.append('./data/JTH/2021_09_14_21_07_41_aligned_frames399.bag')
    bags.append('./data/JTH/2021_09_14_21_07_59_aligned_frames265.bag')
    bags.append('./data/JTH/2021_09_14_21_08_09_aligned_frames120.bag')

    for idx, bag in enumerate(bags):
        coil, obj = get_coil_object_from_bag(bag, show_plt=True)
        o3d.io.write_point_cloud('./temp/coil_pcds_' + str(idx) + '.pcd', coil)
        o3d.io.write_point_cloud('./temp/obj_pcds_' + str(idx) + '.pcd', obj)
        coil_pcds.append(coil)
        object_pcd.append(obj)
else:
    count = 4
    for idx in range(count):
        coil = o3d.io.read_point_cloud('./temp/coil_pcds_' + str(idx) + '.pcd')

        coil, ind = coil.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1)
        coil_pcds.append(coil)
        obj = o3d.io.read_point_cloud('./temp/obj_pcds_' + str(idx) + '.pcd')
        object_pcd.append(obj)

source = object_pcd[3]
target = object_pcd[0]

# BGR to RGB
source.colors = o3d.utility.Vector3dVector(np.asarray(source.colors)[:, ::-1])
# target.colors = o3d.utility.Vector3dVector(np.asarray(target.colors)[:, ::-1])

o3d.visualization.draw_geometries([source, target])

tf_param = utils.reg_pcd(source, target, voxel_size=2, tol=0.01, show_cb=True)
tt = utils.transform_matrix(tf_param.rot, tf_param.t, tf_param.scale)

print('rigid transformation', tt)

utils.colored_pcd_reg(source, target, current_transformation=tt, debug=False)
