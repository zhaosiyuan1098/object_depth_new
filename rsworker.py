# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

from copy import copy
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class RSWorker:
    '''
    https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.option.html
    https://dev.intelrealsense.com/docs/post-processing-filters
    https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
    '''
    def __init__(self,
                 width=848,
                 height=480,
                 fps=30,
                 laser_power=60,
                 ros_bag=None,
                 save_file=None,
                 enable_infrared=True,
                 enable_RGB=True,
                 enable_decimation=False,
                 enable_spatial=True,
                 enable_temporal=True,
                 enable_hole_filling=False):
        # Used if openings stream from prerecorded ros .bag file
        # holds the path to the .bag file
        self.ros_bag = ros_bag

        self.width = width
        self.height = height
        self.fps = fps
        self.save_file = save_file
        self.laser_power = laser_power
        self.enable_infrared = enable_infrared
        self.enable_RGB = enable_RGB
        self.enable_decimation = enable_decimation
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal
        self.enable_hole_filling = enable_hole_filling

        # Data variables that will be set with get_data()
        self.depth_frame = None
        self.color_frame = None
        self.infrared_frame = None
        self.intrinsics = None
        self.depth_scale = None
        self.processed_frames = None
        self.emitter_on = False

        # Post Processing Filter variables with default values

        # Decimation filter variable
        self.decimation_magnitude = 2
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude,
                                   self.decimation_magnitude)

        # Spatial filter variables
        self.spatial_magnitude = 2
        self.spatial_smooth_alpha = 0.5
        self.spatial_smooth_delta = 20
        self.spatial_holes_fill = 0
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude,
                                self.spatial_magnitude)
        self.spatial.set_option(rs.option.filter_smooth_alpha,
                                self.spatial_smooth_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta,
                                self.spatial_smooth_delta)
        self.spatial.set_option(rs.option.holes_fill, self.spatial_holes_fill)

        # Temporal filter variables
        self.temporal_smooth_alpha = 0.4
        self.temporal_smooth_delta = 20
        self.persistency_index = 3
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.holes_fill, self.persistency_index)
        self.spatial.set_option(rs.option.filter_smooth_alpha,
                                self.temporal_smooth_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta,
                                self.temporal_smooth_delta)

        # Holes Filling filter variable
        self.hole_filling_option = 1
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill,
                                     self.hole_filling_option)

        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        # Holds the data frame after it has undergone filtering
        self.processed_depth_frame = None

        # Configure and start streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        if ros_bag is None:
            try:
                # Get device product line for setting a supporting resolution
                pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
                pipeline_profile = config.resolve(pipeline_wrapper)
                device = pipeline_profile.get_device()
                self.device_product_line = str(
                    device.get_info(rs.camera_info.product_line))
                self.found_rgb = False
                for s in device.sensors:
                    if s.get_info(rs.camera_info.name) == 'RGB Camera':
                        self.found_rgb = True
                        break
            except:
                print('Maybe no device is connected?')

        if ros_bag:
            # config = rs.config()
            rs.config.enable_device_from_file(config,
                                              self.ros_bag,
                                              repeat_playback=False)

        else:
            config.enable_stream(rs.stream.depth, self.width, self.height,
                                 rs.format.z16, self.fps)

            self.enable_RGB = self.enable_RGB and self.found_rgb
            if self.enable_RGB:
                if self.device_product_line == 'L500':
                    config.enable_stream(rs.stream.color, 960, 540,
                                         rs.format.bgr8, self.fps)
                else:
                    config.enable_stream(rs.stream.color, self.width,
                                         self.height, rs.format.bgr8, self.fps)
            if enable_infrared:
                config.enable_stream(rs.stream.infrared, 1, self.width,
                                     self.height, rs.format.y8, self.fps)
                config.enable_stream(rs.stream.infrared, 2, self.width,
                                     self.height, rs.format.y8, self.fps)

            if save_file:
                config.enable_record_to_file(save_file)

        self.config = config

    def start(self):
        self.align_to_color = rs.align(rs.stream.color)
        self.align_to_depth = rs.align(rs.stream.depth)
        self.align_to_infrared = rs.align(rs.stream.infrared)

        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()
        depth_sensor = self.profile.get_device().first_depth_sensor()
        # Get depth scale
        self.depth_scale = depth_sensor.get_depth_scale()

        if self.ros_bag:
            playback = self.profile.get_device().as_playback(
            )  # get playback device
            playback.set_real_time(False)  # disable real-time playback
            return

        for sensor in device.query_sensors():
            sensor.set_option(rs.option.frames_queue_size, 0)

        if self.laser_power <= 0:
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            print('Laser emitter off.')
        else:
            laser_pwr = depth_sensor.get_option(rs.option.laser_power)
            laser_range = depth_sensor.get_option_range(rs.option.laser_power)
            print("laser power = {}, laser power range = {}~{}".format(
                laser_pwr, laser_range.min, laser_range.max))
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
            if self.laser_power <= 1:
                depth_sensor.set_option(rs.option.laser_power,
                                        self.laser_power * laser_range.max)
            else:
                depth_sensor.set_option(rs.option.laser_power,
                                        self.laser_power)
            # https://github.com/IntelRealSense/librealsense/issues/8978
            # Manually control laser status would drop fps, and its status
            # won't take effect immediately on frames
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
            print('Laser emitter_on_off')

        sensor.set_option(rs.option.enable_auto_exposure, 1)
        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for x in range(5):
            self.pipeline.wait_for_frames()

    def get_data(self,
                 aligned_to_color=False,
                 aligned_to_depth=False,
                 aligned_to_infrared=False,
                 only_emitter_on=False):

        depth_sensor = self.profile.get_device().first_depth_sensor()
        if False:
            # FPS would drop dramatically
            self.emitter_enabled = not self.emitter_enabled
            depth_sensor.set_option(rs.option.emitter_enabled,
                                    self.emitter_enabled)

        # frames = self.pipeline.wait_for_frames()
        is_frame, frames = self.pipeline.try_wait_for_frames()
        if not is_frame:
            return None

        if True:
            self.processed_frames = self._filter_frames(frames)
        else:
            self.processed_frames = frames

        if aligned_to_color:
            self.processed_frames = self.align_to_color.process(
                self.processed_frames)
        elif aligned_to_depth:
            self.processed_frames = self.align_to_depth.process(
                self.processed_frames)
        elif aligned_to_infrared:
            self.processed_frames = self.align_to_infrared.process(
                self.processed_frames)
        else:
            self.processed_frames = self.processed_frames

        if self.enable_RGB:
            self.color_frame = self.processed_frames.get_color_frame()

        if self.enable_infrared:
            self.infrared_frame = self.processed_frames.first(
                rs.stream.infrared)
        else:
            self.infrared_frame = None

        self.emitter_on = rs.depth_frame.get_frame_metadata(
            frames.get_depth_frame(),
            rs.frame_metadata_value.frame_laser_power_mode)
        #     rs.frame_metadata_value.frame_emitter_mode)
        if self.emitter_on or not only_emitter_on:
            self.depth_frame = self.processed_frames.get_depth_frame()
        # print('self.depth_frame', self.depth_frame.get_timestamp())
        # Validate that both frames are valid

        self.intrinsics = self.processed_frames.profile \
                                .as_video_stream_profile() \
                                .intrinsics
        return self.processed_frames

    def stop(self):
        self.pipeline.stop()

    def _filter_frames(self, frames):
        '''
        Apply a cascade of filters on the depth frame
        '''
        filter_frames = frames

        # DECIMATION FILTER
        if self.enable_decimation and True:
            filter_frames = self.decimation.process(
                filter_frames).as_frameset()

        filter_frames = self.depth_to_disparity.process(
            filter_frames).as_frameset()

        # SPATIAL FILTER
        if self.enable_spatial:
            filter_frames = self.spatial.process(filter_frames).as_frameset()

        # TEMPORAL FILTER
        if self.enable_temporal:
            try:
                filter_frames = self.temporal.process(
                    filter_frames).as_frameset()
            except:
                print('Temporal filter errors')

        try:
            filter_frames = self.disparity_to_depth.process(
                filter_frames).as_frameset()
        except:
            print('Disparity filter errors')

        # HOLE FILLING
        if self.enable_hole_filling:
            filter_frames = self.hole_filling.process(
                filter_frames).as_frameset()

        return filter_frames

    def frame_to_np_array(self, frame, colorize_depth=False):
        # Create colorized depth frame
        if colorize_depth:
            colorizer = rs.colorizer()
            frame_as_image = np.asanyarray(
                colorizer.colorize(frame).get_data())
            return frame_as_image
        else:
            return np.asanyarray(frame.get_data())

    def DbScanClustering(folderface,pcd, numface, eps, min_points):
        # path = './temp/obj_pcds_0.pcd'
        # print(path)

        # pcd = o3d.io.read_point_cloud("data/20221222/scene_all.pcd")
        print(pcd)

        # 定义以选中的点开始蔓延，邻居点距离eps的，最小有min_points个点，可以构成一个簇；适用于点云分隔的比较开的，一块一块的点云。
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(
                eps, min_points, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(
            labels / (max_label if max_label > 0 else 1))

        colors[labels < 0] = 0
        #pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # o3d.visualization.draw_geometries([pcd], "o3d dbscanclusting origin", width=400, height=400)
        # o3d.io.write_point_cloud(folder + '/head_scene' + ".ply", pcd)
        print(labels, len(labels))
        min = labels.min()
        max = labels.max()
        # print('min: ', min, " max: ", max)

        # 打印聚类非0的点云下标，点云数
        print(np.nonzero(labels), '  type: ', type(np.nonzero(labels)),
            ' size: ', len(np.array(np.nonzero(labels)[0])))

        zero_index = np.where(labels == 0)  # 提取分类为0的聚类点云下标
        zero_pcd = pcd.select_by_index(np.array(zero_index)[0])  # 根据下标提取点云点
        print(zero_pcd)
        print('zero_index: ', zero_index, " size: ", len(np.array(zero_index)[0]))
        numface = 0
        face_pcd = o3d.geometry.PointCloud()
        for label in range(min, max+1):
            label_index = np.where(labels == label)  # 提取分类为label的聚类点云下标
            label_pcd = pcd.select_by_index(np.array(label_index)[0])  # 根据下标提取点云点
            print('label: ', str(label), '点云数：', len(label_pcd.points))
            if (len(label_pcd.points) == 93348 or len(label_pcd.points) == 23314):
            # if label == 1 or label == 2:
                # 可视化
                o3d.visualization.draw_geometries(
                    [label_pcd], "o3d dbscanclusting " + str(label) + " results", width=400, height=400)
                # 分别按分类写入文件
                face_pcd = face_pcd + label_pcd
        # o3d.io.write_point_cloud("F:/MRI/data/20221208/" +
        #                          str(numface+1) + ".ply", face_pcd)
        o3d.io.write_point_cloud(folderface + '/head_piece' + ".ply", face_pcd)
        o3d.visualization.draw_geometries([face_pcd])
