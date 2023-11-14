# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

import numpy as np


def read_bag(bag_file, enable_align=False):
    try:
        import pyrealsense2 as rs
    except (RuntimeError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError('pyrealsense2 is not found: ', e)
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)

    align = rs.align(rs.stream.color)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Calculate clipping distance
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 1  # unit meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    frames = pipeline.wait_for_frames()
    pipeline.stop()

    if enable_align:
        aligned_frames = align.process(frames)
        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        print('Using aligned')
    else:
        aligned_depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    intrinsics = frames.profile.as_video_stream_profile().intrinsics

    return depth_image, color_image, intrinsics, depth_scale


def draw_registration_result_original_color(source, target, transformation):
    try:
        import open3d as o3d
        import copy
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d or copy module is not found: ', e)
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])


def depthToPcd(depth_img, pinhole_camera_intrinsic, depth_scale=1e-3):
    try:
        import open3d as o3d
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d module is not found: ', e)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_img), pinhole_camera_intrinsic)

    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # convert unit from m to mm
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) / depth_scale)

    return pcd


def rgbdToPcd(depth_image,
              color_image,
              pinhole_camera_intrinsic,
              depth_scale=1e-3):
    try:
        import open3d as o3d
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d module is not found: ', e)
    img_color = o3d.geometry.Image(color_image)
    img_depth = o3d.geometry.Image(depth_image)
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        img_color, img_depth, convert_rgb_to_intensity=False)

    if len(np.shape(np.asarray(rgbd_image))) < 2 and False:
        print('rgbd_image', np.shape(np.asarray(rgbd_image)),
            'pinhole_camera_intrinsic', pinhole_camera_intrinsic)
        print('Invalid rgbd image: ', np.shape(np.asarray(rgbd_image)))
        print('color_image', np.shape(np.asarray(color_image)))
        print('depth_image', np.shape(np.asarray(depth_image)))
        try:
            import matplotlib.pyplot as plt
        except (RuntimeError, ModuleNotFoundError) as e:
            print('matplotlib module is not found: ', e)
        plt.subplot(2, 1, 1)
        plt.imshow(np.asarray(color_image))
        plt.subplot(2, 1, 2)
        plt.imshow(np.asarray(depth_image))
        plt.show()
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, pinhole_camera_intrinsic)

    # Flip it, otherwise the pointcloud will be upside down
    # temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # convert unit from m to mm
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(temp.points) / depth_scale)
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(temp.colors))
    return pcd


def seg_color(img,
              lower_range,
              upper_range,
              kernel=np.ones((3, 3)),
              show_plt=False):
    try:
        import cv2
    except (RuntimeError, ModuleNotFoundError) as e:
        print('OpenCV module is not found: ', e)
    img = img[:, :, ::-1]  # Convert BGR to RGB
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_range, upper_range)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)

    if show_plt:
        try:
            import matplotlib.pyplot as plt
        except (RuntimeError, ModuleNotFoundError) as e:
            print('matplotlib module is not found: ', e)
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        plt.imshow(hsv_img)
        plt.subplot(2, 2, 3)
        plt.imshow(mask, cmap="gray")
        plt.subplot(2, 2, 4)
        plt.imshow(result)
        plt.show()

    return result, mask


def watershed(image, showPlt=True):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    if showPlt:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=2,
                                 figsize=(9, 3),
                                 sharex=True,
                                 sharey=True)
        ax = axes.ravel()
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Overlapping objects')
        ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
        ax[1].set_title('Separated objects')
        for a in ax:
            a.set_axis_off()
        fig.tight_layout()
        plt.show()

    return labels


def get_clusters(pcd, show_plt=True, eps=8, min_points=50, debug=False):
    try:
        import open3d as o3d
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d module is not found: ', e)
    labels = np.array(
        pcd.cluster_dbscan(eps=eps,
                           min_points=min_points,
                           print_progress=False))
    max_label = labels.max()
    if debug:
        print(f"point cloud has {max_label+1} clusters. {labels.min()}")
    if show_plt:
        try:
            import matplotlib.pyplot as plt
        except (RuntimeError, ModuleNotFoundError) as e:
            print('matplotlib module is not found: ', e)
        colors = plt.get_cmap("tab20")(labels /
                                       (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        c_pcd = o3d.geometry.PointCloud()
        c_pcd.points = pcd.points
        c_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([c_pcd])

    clusters_size = []
    clusters = []
    for i in range(max_label + 1):
        cluster_id = np.where(labels == i)[0]
        clusters_size.append(np.shape(cluster_id)[0])
        clusters.append(cluster_id)

    return clusters, clusters_size


def max_n_cluster(pcd,
                  max_n=1,
                  show_plt=True,
                  eps=8,
                  min_points=50,
                  combined=True,
                  debug=False):
    try:
        import open3d as o3d
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d or matplotlib module is not found: ', e)

    clusters, clusters_size = get_clusters(pcd,
                                           show_plt=show_plt,
                                           eps=eps,
                                           min_points=min_points)
    max_n_ind = np.array(clusters_size).argsort()[-max_n:][::-1]
    if debug:
        print('max_n_ind', max_n_ind)

    if combined:
        max_idx = None
        for max_ind in max_n_ind:
            if max_idx is None:
                max_idx = clusters[max_ind]
            else:
                max_idx = np.concatenate((max_idx, clusters[max_ind]))
        return pcd.select_by_index(max_idx), max_idx

    else:
        max_clusters = []
        max_idx = []
        for max_ind in max_n_ind:
            max_idx.append(clusters[max_ind])
            max_clusters.append(pcd.select_by_index(clusters[max_ind]))
        return max_clusters, max_idx


def closest_pcd(clusters, target):
    min_idx = -1
    min_dist = 9999
    for cluster, idx in zip(clusters, range(len(clusters))):
        dists = np.asarray(cluster.compute_point_cloud_distance(target))
        if min_dist > np.average(dists):
            min_idx = idx
            min_dist = np.average(dists)
    return clusters[min_idx], min_idx, min_dist


def get_coil_face(depth_image,
                  color_image,
                  pinhole_camera_intrinsic,
                  depth_scale,
                  coil_color_low=(20, 0, 90),
                  coil_color_upper=(110, 40, 230),
                  close_kernel=15,
                  coil_max_n=1,
                  object_max_n=2,
                  show_plt=True):
    try:
        import open3d as o3d
        from skimage import morphology
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d or skimage module is not found: ', e)

    _, mask = seg_color(color_image,
                        coil_color_low,
                        coil_color_upper,
                        show_plt=show_plt)

    if np.amax(mask) > 0:
        # Extract coil that should be of certain color range
        depth_coil_image = (mask > 0) * depth_image
        coil_pcd = rgbdToPcd(depth_coil_image, color_image[:, ::-1],
                             pinhole_camera_intrinsic, depth_scale)
        coil_pcd_max, _ = max_n_cluster(coil_pcd,
                                        max_n=coil_max_n,
                                        show_plt=show_plt)
        if show_plt:
            o3d.visualization.draw_geometries([coil_pcd_max])

        # Extract object that should be inside of coil
        mask_fill = morphology.binary_opening(mask,
                                              morphology.disk(close_kernel))
        mask_fill = morphology.convex_hull_image(mask_fill)
        if show_plt:
            try:
                import matplotlib.pyplot as plt
            except (RuntimeError, ModuleNotFoundError) as e:
                print('matplotlib module is not found: ', e)
            plt.subplot(1, 2, 1)
            plt.imshow(mask)
            plt.subplot(1, 2, 2)
            plt.imshow(mask_fill)
            plt.show()
        depth_object_image = (mask <= 0) * mask_fill * depth_image
        object_pcd = rgbdToPcd(depth_object_image, color_image[:, ::-1],
                               pinhole_camera_intrinsic, depth_scale)
        object_pcd_max, _ = max_n_cluster(object_pcd,
                                          max_n=object_max_n,
                                          show_plt=show_plt)
        if show_plt:
            o3d.visualization.draw_geometries([object_pcd_max])

        return coil_pcd_max, object_pcd_max


def colored_pcd_reg(source,
                    target,
                    voxel_radius=[4, 2, 1],
                    max_iter=[200, 100, 60],
                    current_transformation=np.identity(4),
                    debug=True,
                    show_plt=True):
    try:
        import open3d as o3d
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d module is not found: ', e)

    if debug:
        print("3. Colored point cloud registration")
    for scale in range(len(voxel_radius)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        if debug:
            print([iter, radius, scale])
            print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        if debug:
            print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # o3d.visualization.draw_geometries([source_down, target_down])

        if debug:
            print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down,
            target_down,
            radius,
            current_transformation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
        current_transformation = result_icp.transformation
        if debug:
            print(result_icp)
            print('current_transformation', current_transformation)
    if show_plt:
        draw_registration_result_original_color(source, target,
                                                result_icp.transformation)
    return result_icp.transformation


def reg_pcd(source, target, voxel_size=1, tol=0.0001, show_cb=False):
    try:
        import open3d as o3d
        from probreg import callbacks
        import probreg as preg
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d or probreg module is not found: ', e)
    cp = np
    radius_normal = voxel_size * 2
    source = source.voxel_down_sample(voxel_size=voxel_size)
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60))

    target = target.voxel_down_sample(voxel_size=voxel_size)
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60))
    if show_cb:
        cbs = [callbacks.Open3dVisualizerCallback(source, target, save=False)]
    else:
        cbs = []

    objective_type = 'pt2pt'
    tf_param, _, _ = preg.filterreg.registration_filterreg(
        cp.asarray(source.points),
        cp.asarray(target.points),
        objective_type=objective_type,
        tol=tol,
        w=0.5,
        min_sigma2=1e-5,
        sigma2=None,
        update_sigma2=True,
        callbacks=cbs)
    return tf_param


def transform_matrix(rotation, translation, scale):
    # FIXME: use scale
    tt = np.zeros((4, 4))
    tt[:3, :3] = rotation[:3, :3]
    tt[:3, 3] = np.transpose(translation)[:3]
    tt[3, 3] = 1
    return tt


def combinePcds(pcds):
    try:
        import open3d as o3d
    except (RuntimeError, ModuleNotFoundError) as e:
        print('open3d module is not found: ', e)
        return pcds
    pcdAll = o3d.geometry.PointCloud()
    for pcd in pcds:
        pp = np.concatenate(
            (np.asarray(pcdAll.points), np.asarray(pcd.points)), axis=0)
        pc = np.concatenate(
            (np.asarray(pcdAll.colors), np.asarray(pcd.colors)), axis=0)
        pcdAll.points = o3d.utility.Vector3dVector(pp)
        pcdAll.colors = o3d.utility.Vector3dVector(pc)
    return pcdAll


def dist(x, y):
    return np.sqrt(np.sum((x - y)**2))


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    """
	Convert the depth map to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the point cloud in meters
	y : array
		The y values of the point cloud in meters
	z : array
		The z values of the point cloud in meters
	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
    """
	Convert the world coordinate to a 2D image coordinate
	Parameters:
	-----------
	pointcloud 	 	 : numpy array with shape 3xN
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordinate in image
	"""

    assert (pointcloud.shape[0] == 3)
    x_ = pointcloud[0, :]
    y_ = pointcloud[1, :]
    z_ = pointcloud[2, :]

    m = x_[np.nonzero(z_)] / z_[np.nonzero(z_)]
    n = y_[np.nonzero(z_)] / z_[np.nonzero(z_)]

    x = m * camera_intrinsics.fx + camera_intrinsics.ppx
    y = n * camera_intrinsics.fy + camera_intrinsics.ppy

    return x, y
