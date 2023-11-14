# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
import pyrealsense2 as rs
import numpy as np
from rsworker import *
import cv2
import open3d as o3d
import utils as utils
from tqdm import tqdm
import time
import copy
import os


def frame_to_np_array(frame, colorize_depth=False):
    # Create colorized depth frame
    if colorize_depth:
        colorizer = rs.colorizer()
        frame_as_image = np.asanyarray(colorizer.colorize(frame).get_data())
        return frame_as_image
    else:
        return np.asanyarray(frame.get_data())


def readByRSW(bag_file, usePlayBack=True, showPlt=True):
    rsw = RSWorker(
        enable_RGB=True,
        enable_infrared=False,
        ros_bag=bag_file,
    )
    rsw.start()
    playback = rsw.profile.get_device().as_playback()
    depthFrameList = []
    colorFrameList = []

    # FIXME: how to acquire global depth_scale?
    # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_scale = 0.001
    try:
        while True:
            frames = rsw.get_data(aligned_to_color=True)
            if usePlayBack:
                playback.pause()

            if not frames:
                break
            depth_frame = frames.get_depth_frame()

            # FIXME: how to acquire global intrinsics?
            intrinsics = frames.profile.as_video_stream_profile().intrinsics

            depth_np = rsw.frame_to_np_array(depth_frame)

            # NOTE: only first 32 frames could be sustained if copy=False
            depthFrameList.append(np.array(depth_np, copy=True))

            color_np = rsw.frame_to_np_array(frames.get_color_frame())
            color_np = color_np[:, :, ::-1].copy()
            colorFrameList.append(np.array(color_np, copy=True))

            if showPlt:
                depth_img = rsw.frame_to_np_array(depth_frame,
                                                  colorize_depth=True)
                images = np.hstack((depth_img, color_np))
                cv2.namedWindow('Example', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Example', images)
                key = cv2.waitKey(1)

            if usePlayBack:
                playback.resume()
        rsw.stop()
    except RuntimeError:
        print("There are no more frames left in the .bag file!")
    return depthFrameList, colorFrameList, intrinsics, depth_scale


def readWithoutRSW(bag_file, usePlayBack=True):
    # A simple example for reading bag file
    frameList = []
    try:
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=False)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        while True:
            frames = pipeline.wait_for_frames()
            if usePlayBack:
                playback.pause()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            frameList.append(
                np.array(frame_to_np_array(depth_frame), copy=True))
            depth_image = frame_to_np_array(depth_frame, colorize_depth=True)

            cv2.namedWindow('Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Example', depth_image)
            key = cv2.waitKey(1)

            if usePlayBack:
                playback.resume()

    except RuntimeError:
        print("There are no more frames left in the .bag file!")
    return frameList


def extractPcds(bag_file, folder, save_file=True):
    depthFrameList, colorFrameList, intrinsics, depth_scale = readByRSW(
        bag_file=bag_file,
        usePlayBack=False,
        showPlt=True,
    )
    print('frameList', np.shape(np.asarray(depthFrameList)))

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy)

    idx = 1
    pcds = []
    for depthF, colorF in zip(depthFrameList, colorFrameList):
        pcdC = utils.rgbdToPcd(
            depthF,
            colorF[:, :, ::-1].copy(),
            pinhole_camera_intrinsic,
            depth_scale,
        )
        distance = np.linalg.norm(np.asarray(pcdC.points), axis=1)

        # reserve points inside 360mm from camera
        pcdC = pcdC.select_by_index(np.where(distance < 360)[0])

        pcdC, ind = pcdC.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=20)
        if save_file:
            o3d.io.write_point_cloud(
                folder + str(idx) + '.pcd',
                pcdC,
            )
        idx = idx + 1
        pcds.append(pcdC)
    return pcds


def extractFace(pcd, coil):
    big_clusters, _ = utils.max_n_cluster(
        pcd,
        max_n=4,
        eps=4,
        min_points=20,
        show_plt=False,
    )
    dists = big_clusters.compute_point_cloud_distance(coil)
    dists = np.asarray(dists)
    ind = np.where(dists > 10)[0]
    pcd_without_coil = big_clusters.select_by_index(ind)
    return pcd_without_coil


def playFaces(faces, range=[0, -1]):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    faceA = faces[0]
    vis.add_geometry(faceA)

    idx = range[0]
    while True:
        face = faces[idx]
        faceA.points = face.points
        faceA.colors = face.colors
        vis.update_geometry(faceA)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        idx = idx + 1
        if idx >= len(faces):
            idx = range[0]
        if range[1] > 0 and idx >= range[1]:
            print('restting to range[1]', range[1])
            idx = range[0]


def mergeFaces(faces, startIdx, num=20, step=1, showPlt=True):
    target = faces[startIdx]
    targetA, _ = utils.max_n_cluster(
        target,
        1,
        show_plt=False,
        eps=4,
        min_points=20,
    )
    faceA = targetA

    tfList = []
    vis = o3d.visualization.Visualizer()
    if showPlt:
        vis.create_window()
        vis.add_geometry(target)

    # registration between consecutive frames (more robust)
    # instead of current frame to a global ref
    incremented = True

    for i in tqdm(range(num), position=0, leave=False):
        source = faces[step * i + startIdx + 1]

        if False:
            sourceA, _ = utils.max_n_cluster(
                source,
                1,
                show_plt=False,
                eps=4,
                min_points=20,
            )
        else:
            clusters, _ = utils.max_n_cluster(source,
                                              3,
                                              show_plt=False,
                                              eps=4,
                                              min_points=20,
                                              combined=False)
            sourceA, _, _ = utils.closest_pcd(clusters, faceA)
        # o3d.visualization.draw_geometries([sourceA])

        # FIXME: reg_pcd could also cause crash
        tf_param = utils.reg_pcd(sourceA, targetA, show_cb=False)
        trans_init = utils.transform_matrix(tf_param.rot, tf_param.t,
                                            tf_param.scale)
        try:
            # FIXME: open3d reg could cause crash
            tf = utils.colored_pcd_reg(sourceA,
                                       targetA,
                                       current_transformation=trans_init,
                                       debug=False,
                                       show_plt=False)
        except:
            print('Open3d errors')
            tf = trans_init
        if incremented and i > 0:
            tf = tfList[i - 1] @ tf
        tfList.append(tf)

        temp = copy.deepcopy(source)
        temp.transform(tf)

        merge = utils.combinePcds([temp, target])
        merge = merge.voxel_down_sample(voxel_size=1)
        target.points = merge.points
        target.colors = merge.colors

        # in case clusters are connected when points are merged
        if not incremented:
            tempA = copy.deepcopy(sourceA)
            tempA.transform(tf)
            targetA = utils.combinePcds([tempA, targetA])
            targetA = targetA.voxel_down_sample(voxel_size=1)
        else:
            targetA = copy.deepcopy(sourceA)

        # gc.collect()

        if showPlt:
            vis.update_geometry(target)
            vis.poll_events()
            vis.update_renderer()

    if showPlt:
        o3d.visualization.draw_geometries([target])
    return tfList, target, targetA


# folder = 'data/035T/fang_pcd/'
folder = 'data/035T/fang_cycle_0415/'
if not os.path.exists(folder):
    os.mkdir(folder)

if False:
    extractPcds(
        'data/035T/fang_cycle_0415.bag',
        folder=folder,
    )

pcd = o3d.io.read_point_cloud(folder + str(1) + '.pcd')

size = 157
faces = []

reload_faces = False
if reload_faces:
    coil, _ = utils.max_n_cluster(
        pcd,
        max_n=1,
        eps=4,
        min_points=20,
        show_plt=False,
    )
    # o3d.visualization.draw_geometries([coil])

    if not os.path.exists(folder + 'faces/'):
        os.mkdir(folder + 'faces/')

    faceA = extractFace(pcd, coil)
    for i in tqdm(range(size), position=0, leave=False):
        pcdA = o3d.io.read_point_cloud(folder + str(i + 1) + '.pcd')
        face = extractFace(pcdA, coil)
        o3d.io.write_point_cloud(
            folder + 'faces/' + str(i + 1) + '.pcd',
            face,
        )
        faces.append(face)
else:
    for i in range(size):
        face = o3d.io.read_point_cloud(folder + '/faces/' + str(i + 1) +
                                       '.pcd')
        faces.append(face)

# o3d.visualization.draw_geometries(faces)
# o3d.visualization.draw_geometries([faces[20], faces[53]])

if False:
    start = 0
    num = 70
    tfList, target, _ = mergeFaces(faces,
                                   startIdx=start,
                                   num=num,
                                   showPlt=False)
    o3d.visualization.draw_geometries([target])
    if not os.path.exists(folder + 'transforms/'):
        os.mkdir(folder + 'transforms/')
    np.save(folder + '/transforms/' + str(start) + '_' + str(num) + '_1.npy',
            np.asarray(tfList))

show_all_faces = True
if show_all_faces:
    playFaces(faces)
