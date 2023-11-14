# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
import pyrealsense2 as rs
import numpy as np
from rsworker import *
import cv2
from PIL import Image, ImageDraw
import sigpy.plot as pl
import json


def poly_mask(shape, *vertices, value=np.nan):
    """
    Create a mask array filled with 1s inside the polygon and 0s outside.
    The polygon is a list of vertices defined as a sequence of (column, line) number, where the start values (0, 0) are in the
    upper left corner. Multiple polygon lists can be passed in input to have multiple,eventually not connected, ROIs.
        column, line   # x, y
        vertices = [(x0, y0), (x1, y1), ..., (xn, yn), (x0, y0)] or [x0, y0, x1, y1, ..., xn, yn, x0, y0]
    Note: the polygon can be open, that is it doesn't have to have x0,y0 as last element.

    adapted from: https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/64876117#64876117
    :param shape:    (tuple) shape of the output array (height, width)
    :param vertices: (list of tuples of int): sequence of vertices defined as
                                            [(x0, y0), (x1, y1), ..., (xn, yn), (x0, y0)] or
                                            [x0, y0, x1, y1, ..., xn, yn, x0, y0]
                                            Multiple lists (for multiple polygons) can be passed in input
    :param value:    (float or NAN)      The masking value to use (e.g. a very small number). Default: np.nan
    :return:         (ndarray) the mask array
    """
    width, height = shape[::-1]
    # create a binary image
    img = Image.new(mode='L', size=(width, height),
                    color=0)  # mode L = 8-bit pixels, black and white
    draw = ImageDraw.Draw(img)
    # draw polygons
    for polygon in vertices:
        draw.polygon(polygon, outline=1, fill=1)
    # replace 0 with 'value'
    mask = np.array(img).astype('float32')
    mask[np.where(mask == 0)] = value
    return mask


def frame_to_np_array(frame, colorize_depth=False):
    # Create colorized depth frame
    if colorize_depth:
        colorizer = rs.colorizer()
        frame_as_image = np.asanyarray(colorizer.colorize(frame).get_data())
        return frame_as_image
    else:
        return np.asanyarray(frame.get_data())


def detectAruco(frameList):
    arucoList = []
    markers = []
    for idx, frame in enumerate(frameList):
        if frame.ndim == 2:
            frame3d = np.dstack((frame, frame, frame))
        elif frame.ndim == 3:
            frame3d = frame
        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame3d, arucoDict, parameters=arucoParams)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame3d, corners, ids)
            if np.shape(ids)[0] == 2:
                marker = {}
                marker['idx'] = idx
                marker['corners'] = corners
                marker['ids'] = ids
                markers.append(marker)

        arucoList.append(frame3d)
    pl.ImagePlot(np.asarray(arucoList), z=0, y=1, x=2, c=3, showPlt=False)


arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
arucoParams = cv2.aruco.DetectorParameters_create()


def online_acq(proc_func=None):
    rsw = RSWorker(fps=30, enable_RGB=False)
    rsw.start()

    while True:
        frames = rsw.get_data()
        depth_frame = rsw.depth_frame
        if rsw.emitter_on or depth_frame is None:
            continue

        infrared_frame = frames.first(rs.stream.infrared)
        depth_img = np.array(rsw.frame_to_np_array(depth_frame), copy=True)
        depth_color_img = rsw.frame_to_np_array(depth_frame,
                                                colorize_depth=True)
        infrared_img = np.array(rsw.frame_to_np_array(infrared_frame),
                                copy=True)

        if proc_func is not None:
            go_on = proc_func(depth_img, infrared_img, rsw.intrinsics,
                              depth_color_img)
            if not go_on:
                break
        else:
            infrared_img_3d = np.dstack(
                (infrared_img, infrared_img, infrared_img))

            corners, ids, rejected = cv2.aruco.detectMarkers(
                infrared_img_3d, arucoDict, parameters=arucoParams)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(infrared_img_3d, corners, ids)

            images = np.hstack((depth_color_img, infrared_img_3d))
            cv2.namedWindow('Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Example', images)
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    rsw.stop()

    if proc_func is not None:
        import sigpy.plot as pl
        pl.ImagePlot(infrared_img)


def offline_acq(folder,
                bag_file,
                proc_func=None,
                save_frames=True,
                show_plt=False,
                first_n=-1):
    infra_available = True
    color_available = False
    rsw = RSWorker(
        enable_RGB=color_available,
        enable_infrared=infra_available,
        ros_bag=bag_file,
    )
    rsw.start()
    playback = rsw.profile.get_device().as_playback()
    depth_frame_list = []
    depth_color_list = []
    infra_frame_list = []
    color_frame_list = []
    depth_scale = 0.001
    usePlayBack = False
    frame_count = 0
    # try:
    while True:
        if infra_available:
            frames = rsw.get_data(aligned_to_infrared=True,
                                    only_emitter_on=True)
        else:
            frames = rsw.get_data(aligned_to_depth=True,
                                    only_emitter_on=True)

        if usePlayBack:
            playback.pause()

        if not frames:
            break
        # depth_frame = frames.get_depth_frame()
        # NOTE: rsw.depth_frame is captured when laser is on
        depth_frame = rsw.depth_frame
        
        print('depth_frame:', infra_available, depth_frame, rsw.emitter_on)
        # if (infra_available and rsw.emitter_on) or depth_frame is None:
        #     continue

        # FIXME: how to acquire global intrinsics?
        intrinsics = frames.profile.as_video_stream_profile().intrinsics
        depth_np = np.array(rsw.frame_to_np_array(depth_frame), copy=True)
        depth_img = np.array(rsw.frame_to_np_array(depth_frame,
                                                    colorize_depth=True),
                                copy=True)
        depth_frame_list.append(depth_np)
        depth_color_list.append(depth_img)
        
        if infra_available:
            infrared_frame = frames.first(rs.stream.infrared)
            infrared_img = np.array(rsw.frame_to_np_array(infrared_frame),
                                    copy=True)
            infra_frame_list.append(infrared_img)

        if color_available:
            color_frame = np.array(rsw.frame_to_np_array(rsw.color_frame),
                                    copy=True)
            color_frame_list.append(color_frame)

        if proc_func is not None:
            proc_func(depth_np, infrared_img, intrinsics, depth_img)
        else:
            depth_np_3d = np.dstack((depth_np, depth_np, depth_np))
            if infra_available:
                infrared_img_3d = np.dstack(
                    (infrared_img, infrared_img, infrared_img))

            if show_plt:
                if infra_available:
                    images = np.hstack((depth_np_3d, infrared_img_3d))
                elif color_available:
                    images = np.hstack((depth_np_3d, color_frame))
                else:
                    images = depth_np_3d
                cv2.namedWindow('Example', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Example', images)
                key = cv2.waitKey(1)

        if usePlayBack:
            playback.resume()

        # only acquire first n frames
        frame_count = frame_count + 1
        if first_n > 0 and frame_count > first_n:
            break
    rsw.stop()

    # except RuntimeError as e:
    #     print("Acquisition error:", e)

    print('depthColorFrameList', np.shape(np.asarray(depth_color_list)))

    ist = {}
    if save_frames:
        np.save(folder + '/depthFrameList.npy', np.asarray(depth_frame_list))
        np.save(folder + '/depthColorFrameList.npy',
                np.asarray(depth_color_list))

        if infra_available:
            np.save(folder + '/infraFrameList.npy',
                    np.asarray(infra_frame_list))
        if color_available:
            np.save(folder + '/colorFrameList.npy',
                    np.asarray(color_frame_list))

        with open(folder + '/intrinsics.json', 'w') as ff:
            ist['width'] = intrinsics.width
            ist['height'] = intrinsics.height
            ist['fx'] = intrinsics.fx
            ist['fy'] = intrinsics.fy
            ist['ppx'] = intrinsics.ppx
            ist['ppy'] = intrinsics.ppy
            ff.write(json.dumps(ist))

    if show_plt:
        pl.ImagePlot(np.asarray(depth_color_list),
                     z=0,
                     y=1,
                     x=2,
                     c=3,
                     showPlt=False)
        if infra_available:
            detectAruco(infra_frame_list)
            pl.ImagePlot(np.asarray(infra_frame_list), z=0, showPlt=True)
        elif color_available:
            detectAruco(color_frame_list)
            pl.ImagePlot(np.asarray(color_frame_list),
                         z=0,
                         y=1,
                         x=2,
                         c=3,
                         showPlt=True)

    return depth_frame_list, depth_color_list, infra_frame_list, ist


if __name__ == '__main__':
    if True:
        folder = 'data/hxh/20221219/'
        bag_file = folder + '/20221219_161513.bag'
        offline_acq(folder,
                    bag_file,
                    show_plt=True,
                    save_frames=True,
                    first_n=30)
    else:
        online_acq()