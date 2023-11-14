# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
import numpy as np
import open3d as o3d
import sigpy.plot as pl
import trimesh
import copy
from pytransform3d import transformations as pytr
from pytransform3d.transform_manager import TransformManager
import transforms3d as t3d
# from probreg import cpd, callbacks
import time


def reg_pcd_cpd(source, target):
    use_cuda = True
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    source = cp.asarray(source.points, dtype=cp.float32)
    target = cp.asarray(target.points, dtype=cp.float32)

    rcpd = cpd.RigidCPD(source, use_cuda=use_cuda)
    start = time.time()
    tf_param, _, _ = rcpd.registration(target)
    elapsed = time.time() - start
    print("time: ", elapsed)
    print("result: ", np.rad2deg(t3d.euler.mat2euler(to_cpu(tf_param.rot))),
          tf_param.scale, to_cpu(tf_param.t))
    return tf_param


def reg_full_head(folder, head_piece_path):
    head_piece = o3d.io.read_point_cloud(head_piece_path)
    scene_all = o3d.io.read_point_cloud(folder + "/face/scene.pcd")
    if False:
        head_full_cam_init = o3d.io.read_triangle_mesh(folder +
                                                       '/head_in_cam.ply')
        # head_full_cam_init_pcd = head_full_cam_init.sample_points_uniformly(50000)
        head_full_cam_init_pcd = head_full_cam_init.sample_points_poisson_disk(
            10000)
        o3d.io.write_point_cloud(folder + '/head_full_cam_init_pcd.pcd',
                                 head_full_cam_init_pcd)
        o3d.visualization.draw_geometries(
            [head_full_cam_init_pcd, head_piece, scene_all])
    head_full_cam_init_pcd = o3d.io.read_point_cloud(
        folder + "/head_full_cam_init_pcd.pcd")

    source = head_full_cam_init_pcd.voxel_down_sample(voxel_size=1)
    target = head_piece.voxel_down_sample(voxel_size=1)
    o3d.visualization.draw_geometries([source, target, scene_all])

    # # compute cpd registration
    # cbs = [callbacks.Open3dVisualizerCallback(source, target)]
    # tf_param, _, _ = cpd.registration_cpd(source, target, callbacks=cbs)

    tf_param = reg_pcd_cpd(source, target)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    # draw result
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result, scene_all])


def get_head_10():
    # 全局到局部点云配准的两个变换矩阵
    init_mat = [[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0, 0],
                [np.sin(np.pi / 2),
                 np.cos(np.pi / 2), 0, -20], [0, 0, 1, 100], [0, 0, 0, 1]]
    tm_p = TransformManager()
    tm_p.add_transform("cam", "head_full_1", init_mat)
    reg_mat = [
        [-8.46220366e-01, 7.60963461e-02, -5.27371253e-01, -2.83009119e+02],
        [-1.46101629e-01, 9.18678141e-01, 3.66994261e-01, 1.19695479e+02],
        [5.12411365e-01, 3.87607817e-01, -7.66286352e-01, -8.64201244e+01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]
    tm_p.add_transform("head_full_1", "head_full", reg_mat)
    folder = 'data/20221208/'
    return get_head(folder, 'data/xiaohan_head.ply', folder + '/piece.ply',
                    tm_p.get_transform('head_full', 'cam'))


def get_head(folder,folderface, full_head_path, head_piece_path, head_to_cam):
    head_full = trimesh.load_mesh(full_head_path)
    head_full.apply_transform(head_to_cam)
    head_piece = o3d.io.read_point_cloud(head_piece_path)
    scene_all = o3d.io.read_point_cloud(folder + "/face/scene.pcd")

    if False:
        # FIXME: 'open3d.cpu.pybind.geometry.TriangleMesh' object is not callable
        head_full_pcd = head_full.as_open3d()
    else:
        head_full.export(folderface + '/head_in_cam.ply')
        head_full_pcd = o3d.io.read_triangle_mesh(folderface + '/head_in_cam.ply')

    print('head center in cam', head_full_pcd.get_center())
    print('piece center in cam', head_piece.get_center())

    # o3d.visualization.draw_geometries([head_full_pcd, head_piece, scene_all])
    return head_full_pcd, head_piece, scene_all


def get_cam_to_display_mat(folder,folderface):
    cam_to_ref_p = np.load(folderface + '/cam_to_ref_passive.npy')
    ref_to_mr_active = np.load(folder + '/ref_to_mr_active.npy')
    print('ref_to_mr_active', ref_to_mr_active)
    display_to_mr = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    print('display_to_mr det:', np.linalg.det(display_to_mr))
    tm_a = TransformManager()
    # active mode
    tm_a.add_transform("cam", "ref", pytr.invert_transform(cam_to_ref_p))
    tm_a.add_transform("ref", "mr", ref_to_mr_active)
    tm_a.add_transform("display", "mr", display_to_mr)
    print('cam to display', tm_a.get_transform('cam', 'display'))
    return tm_a.get_transform('cam', 'display')


def transform_pcd(pcd, tf):
    tf_points = pytr.transform(tf,
                               pytr.vectors_to_points(np.asarray(
                                   pcd.points)))[:, :3]

    pcd.points = o3d.utility.Vector3dVector(tf_points)
    return pcd


def  reg_head(folder,folderface):
    cam_to_display = get_cam_to_display_mat(folder,folderface)
    head_to_piece_tf = np.load(folderface + '/head_to_piece.npy')
    head_full_in_cam, _, scene_all = get_head(folder,folderface,
                                              'data/wangjiaen_head.ply',
                                              folderface + '/head_piece.ply',
                                              head_to_piece_tf)
    np.save(folderface + '/cam_to_display.npy', cam_to_display)

    head_full_in_display = copy.deepcopy(head_full_in_cam)
    head_full_in_display.transform(cam_to_display)
    print('head full in display center', head_full_in_display.get_center())
    o3d.io.write_triangle_mesh(folderface + '/head_full_in_display.ply',
                               head_full_in_display)

    scene_in_display = copy.deepcopy(scene_all)
    scene_in_display = transform_pcd(scene_in_display, cam_to_display)
    scene_in_display.estimate_normals()

    radii = [2, 2, 2, 4]
    scene_mesh_in_display = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        scene_in_display, o3d.utility.DoubleVector(radii)) 
    o3d.io.write_triangle_mesh(folderface + '/scene_mesh_in_display.ply',
                               scene_mesh_in_display)

    o3d.visualization.draw_geometries([head_full_in_display, scene_in_display])


if __name__ == "__main__":
    folder = 'data/regtest/0731'
    folderface = 'data/regtest/0731/face'
    reg_head(folder,folderface)
    # reg_full_head('data/20221209', 'data/20221209/head_piece_mm.ply')
