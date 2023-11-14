import numpy as np
import open3d as o3d
import copy
folder = "data/regtest/0802"
folderface = folder + '/face'
# source = o3d.io.read_point_cloud(folderface + '/head_piece.ply')
# target = o3d.io.read_point_cloud(folderface + '/scene.pcd')
# o3d.visualization.draw_geometries([source,  target])


# head_to_piece = np.load(folder + 'head_to_piece.npy')
ref_to_mr_active = np.load(folder + '/ref_to_mr_active.npy')
print('ref_to_mr_active', ref_to_mr_active)

# target1 = copy.deepcopy(target)
# target2 = copy.deepcopy(target)

# show1 = o3d.io.read_point_cloud(folder + 'show1.ply')

# o3d.visualization.draw_geometries([source2,  show1])
# o3d.visualization.draw_geometries([source1,  target1.transform(head_to_piece2)])
# o3d.visualization.draw_geometries([temp1, target2.transform(head_to_piece)])


# head_scene.estimate_normals()

# radii = [2, 2, 2, 4]
# scene_mesh_in_display = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     head_scene, o3d.utility.DoubleVector(radii))
# o3d.io.write_triangle_mesh(folder + '/head_mesh_scene.ply', head_scene)

#下面我给出两个三维空间的坐标点信息，计算两个点之间的距离

# def plantmodel(plant, x, y, z):
#     a, b, c, d = plant
#     t = (a*x + b*y + c*z + d) / (a*a + b*b + c*c)
#     return (x - a*t, y - b*t, z - c*t)
# cen1 = [ 25.36941356, 58.59951671, -2.25194508]
# cen2 = [9.67560649,  -4.6875    , -37.83313037]
# distance = np.sqrt(np.sum(np.square(np.array(cen1) - np.array(cen2))))
# print(distance)

# print('vvv', np.linalg.norm([-100, 0, 8.3]))
# print('222', np.linalg.norm([88,2,7]))
