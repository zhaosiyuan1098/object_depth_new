from timeit import default_timer as timer
import copy
import numpy as np
import open3d as o3d
import time
# from skimage.transform import SimilarityTransform


def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    pcd_data = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(pcd_data.paths[0])
    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration(source, target):
    print("Demo for manual ICP")
    # pcd_data = o3d.data.DemoICPPointClouds()

    # target.transform(headtopiece)
    # source = o3d.io.read_point_cloud(pcd_data.paths[0])
    # target = o3d.io.read_point_cloud(pcd_data.paths[2])
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")
    print("reg_p2p", reg_p2p.transformation)
    teanform0 = reg_p2p.transformation
    np.save(folder + "/teanform0.npy", teanform0)
    return(teanform0)



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    # target_temp.paint_uniform_color([0.7451, 0.7451, 0.7451])
    source_temp.transform(transformation)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    o3d.visualization.draw_geometries([source_temp, target_temp,mesh],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])



def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source,target,voxel_size, teanform0):
    loc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    # o3d.visualization.draw_geometries([source,loc])
                                       # source.remove_non_finite_points()
    teanform1 = np.asarray([[ 9.99969896e-01  ,1.14254025e-03 ,-7.67479393e-03 ,-2.80705292e+00],
                            [-1.18128918e-03 , 9.99986570e-01, -5.04622118e-03  ,2.01346936e+00],
                            [ 7.66892535e-03 , 5.05513541e-03 , 9.99957816e-01  ,8.64843645e-01],
                            [ 0.00000000e+00  ,0.00000000e+00  ,0.00000000e+00  ,1.00000000e+00]])
    teanform2 = np.asarray([[ 1.00000000e+00 , 9.84838355e-07 ,-6.29598412e-06,-9.11726482e-05],
                            [-9.84869333e-07 , 1.00000000e+00 ,-4.92028569e-06 ,-8.52427323e-04],
                            [ 6.29597927e-06,  4.92029189e-06 , 1.00000000e+00 ,-6.44694216e-04],
                            [ 0.00000000e+00  ,0.00000000e+00  ,0.00000000e+00 , 1.00000000e+00]])
    # piece_to_head = teanform0
    # piece_to_head =teanform1@teanform0                                       
    piece_to_head =teanform2@teanform1@teanform0

    np.save(folder+"/piece_to_head.npy", piece_to_head)
    head_to_piece = np.linalg.inv(piece_to_head)
    np.save(folder+"/head_to_piece.npy", head_to_piece)

    # fortrans=temp
    # temp = np.dot(fortrans0,fortrans1)
    # fortrans = np.dot(temp,fortrans2)
    # print(head_to_piece)
    source.transform(piece_to_head)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

folder = 'data/regtest/0804_1/face'
source = o3d.io.read_point_cloud(folder + '/head_piece.ply')
target = o3d.io.read_point_cloud('data/wangjiaen_head.ply')
#可视化source和target
o3d.visualization.draw_geometries([source, target])
# teanform0 = demo_manual_registration(source, target)
teanform0 = np.load(folder + '/teanform0.npy')
voxel_size = 5  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target,
    voxel_size,teanform0)

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = 3
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.98),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))
    return result

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
# print(result_ransac)
# draw_registration_result(source_down, target_down, result_ransac.transformation)


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    # distance_threshold = voxel_size * 0.4
    distance_threshold = 5
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                 voxel_size)
print(result_icp)
while result_icp.fitness < 0.94:
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size)
    print(result_icp.fitness)
print("result_icp.transformation",result_icp.transformation)

draw_registration_result(source, target, result_icp.transformation)

# voxel_size = 0.05  # means 5cm for the dataset
# source, target, source_down, target_down, source_fpfh, target_fpfh = \
#         prepare_dataset(voxel_size)

