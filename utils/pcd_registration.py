import numpy as np
import open3d as o3d
from sklearn import cluster
from copy import deepcopy




# Post-processing for 3D point cloud file


# Load point cloud data from PLY file
pcd = o3d.io.read_point_cloud("H:/Ubuntu/Final/01/01_mixed_onlydice_edit.ply")
o3d.visualization.draw_geometries_with_editing([pcd], window_name="1")

# Create the coordinate frame
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=16, origin=[0, -566.75256348, 179.75511169])




###############  Cluster all points  ##################

# Cluster the point cloud with sklearn DBSCAN
points = np.array(pcd.points)
clustering = cluster.DBSCAN(eps=0.8, min_samples=16)        # key configuration, generalizable
clustering.fit(points)
labels = clustering.labels_

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
# print(clustering.components_)                                        # not the coordinates of centroids


# 显示颜色设置
colors = np.random.randint(0, 255, size=(max(labels) + 1, 3)) / 255    # 需要设置为n+1类，否则会数据越界造成报错
colors = colors[labels]                                                # 为每个label直接分配一个颜色
colors[labels < 0] = 0                                                 # 噪点直接设置为0，用黑色显示
pcd_cluster = deepcopy(pcd)
pcd_cluster.colors = o3d.utility.Vector3dVector(colors)


# 点云可视化
o3d.visualization.draw_geometries([pcd_cluster, mesh_frame], window_name="3")

# Get the center of all initial point clouds
centroid_rib = pcd.get_center()         # [   4.73144334 -626.26757434  183.83373802]


###############  Remove all small fragments  ##################

# Define the threshold for the minimum number of points in a cluster
min_cluster_size_1 = 3000           # 5000
min_cluster_size_2 = 40000          # 5000 40000 (Nr.3 file) 70000 (Nr.1.2 file)

# Loop through each cluster and check if it is too small and if the center of this cluster is also right
for i in range(max(clustering.labels_)+1):

    cluster_indices = np.where(clustering.labels_ == i)[0]
    
    if len(cluster_indices) > 0:
        # Get the coordinates of the points in the cluster
        cluster_points = np.asarray(pcd.points)[cluster_indices]

        # Calculate the centroid of the cluster
        cluster_center = np.mean(cluster_points, axis=0)

    if (len(cluster_indices) < min_cluster_size_2 and cluster_center[0] > centroid_rib[0] and cluster_center[1] < centroid_rib[1]) \
        or len(cluster_indices) < min_cluster_size_1 or (cluster_center[0] > 50):

        # If the cluster is too small, delete it
        clustering.labels_[cluster_indices] = -1

# [  33.64257287 -592.24281944  186.10662022]

# Remove deleted points from the point cloud and save the result to a new file
pcd = pcd.select_by_index(np.where(clustering.labels_ != -1)[0])

o3d.visualization.draw_geometries_with_editing([pcd], window_name="4")



############  Visualization of the filtered point cloud cluster   ##############


points = np.array(pcd.points)
clustering = cluster.DBSCAN(eps=0.8, min_samples=16)        
clustering.fit(points)
labels = clustering.labels_

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")


colors = np.random.randint(0, 255, size=(max(labels) + 1, 3)) / 255    
colors = colors[labels]    
colors[labels < 0] = 0     
pcd_cluster = deepcopy(pcd)
pcd_cluster.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_cluster, mesh_frame], window_name="3")



'''
###################  Only for Nr.2 bone file, to duplicate for the misssing bone  ######################
for i in range(max(clustering.labels_)+1):     # i=3, total 7clusters

    cluster_indices = np.where(clustering.labels_ == i)[0]
    
    if len(cluster_indices) > 0:
        # Get the coordinates of the points in the cluster
        cluster_points = np.asarray(pcd.points)[cluster_indices]

        # Calculate the centroid of the cluster
        cluster_center = np.mean(cluster_points, axis=0)
        print(cluster_center)

    if i == 4:
        # If the cluster is too small, delete it
        clustering.labels_[cluster_indices] = 666


# Remove deleted points from the point cloud and save the result to a new file
mirror_pcd = pcd.select_by_index(np.where(clustering.labels_ == 666)[0])

# 定义y=centroid_rib[1]平面上的镜像变换
mirror_transform = [[1, 0, 0, 0],
                    [0, -1, 0, 2*centroid_rib[1]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]

# 对点云进行变换
mirror_pcd.transform(mirror_transform)

# 可视化结果
pcd.points = o3d.utility.Vector3dVector(np.append(np.asarray(pcd.points), np.asarray(mirror_pcd.points), axis=0))
o3d.visualization.draw_geometries([pcd])

points = np.array(pcd.points)
clustering = cluster.DBSCAN(eps=0.8, min_samples=16)        # eps=0.8, min_samples=16
clustering.fit(points)
labels = clustering.labels_

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
# print(clustering.components_)


# 显示颜色设置
colors = np.random.randint(0, 255, size=(max(labels) + 1, 3)) / 255    # 需要设置为n+1类，否则会数据越界造成报错
# print(colors, labels)
colors = colors[labels]    # 很巧妙，为每个label直接分配一个颜色
colors[labels < 0] = 0     # 噪点直接设置为0，用黑色显示
pcd_cluster = deepcopy(pcd)
pcd_cluster.colors = o3d.utility.Vector3dVector(colors)

'''
#################


# max_xyzs = []
# min_xyzs = []

# for i in range(max(clustering.labels_)+1):

#     cluster_indices = pcd.select_by_index(np.where(clustering.labels_ == i)[0])
    
#     max_xyz = cluster_indices.get_max_bound()
#     min_xyz = cluster_indices.get_min_bound()

#     max_xyzs.append(max_xyz)
#     min_xyzs.append(min_xyz)


# max_xyzs = np.asarray(max_xyzs)
# min_xyzs = np.asarray(min_xyzs)


# 点云可视化
o3d.visualization.draw_geometries([pcd_cluster], window_name="5")


# 以8根软骨的重心来画长方体，效果不好，需要额外设定参数
centroid_rib = pcd.get_center()     # [  -3.2841786  -628.52021931  181.3868074 ]



########################################
# 构建肋骨的平面并找到该平面的法向量


# 点云聚类
n_clusters = 4    # 聚类簇数
points = np.array(pcd.points)
kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(points)
labels = kmeans.labels_
# print(kmeans.cluster_centers_)   # 聚类中心点的顺序不固定

# 显示颜色设置
colors = np.random.randint(0, 255, size=(n_clusters, 3)) / 255
colors = colors[labels]

pcd_cluster = deepcopy(pcd)
pcd_cluster.colors = o3d.utility.Vector3dVector(colors)


p1, p2, p3 = kmeans.cluster_centers_[0], kmeans.cluster_centers_[3], centroid_rib


# 将三个点放在一个 numpy 数组中, 这三个点可以一直构造稳定的平面
points = np.array([p1, p2, p3])

centers_points = np.array(points)
points_pcd = o3d.geometry.PointCloud()
points_pcd.points = o3d.utility.Vector3dVector(centers_points)
points_pcd.paint_uniform_color([0, 0.3, 0])


# 创建三角形面片对象
triangle_mesh = o3d.geometry.TriangleMesh()

# 添加三角形面片的顶点
triangle_mesh.vertices = o3d.utility.Vector3dVector([p1, p2, p3])

# 添加三角形面片的三角形
triangle_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])

triangle_mesh.paint_uniform_color([1, 0.706, 0])


# 计算三角面片的法向量
triangle_mesh.compute_triangle_normals()

# 获取平面的法向量
normal = np.asarray(triangle_mesh.triangle_normals)[0]
normal /= np.linalg.norm(normal)
# print(normal)                                   # [-1.84938344e-01  7.35184713e-04  9.82749850e-01]

# 选择平面上的一个点，例如三角形的重心
center = np.mean([p1, p2, p3], axis=0)          
                                                
# 计算法向量的末端点坐标
arrow_end = center + normal * 50


# 创建法向量线段
arrow = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([center, arrow_end]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)

# 绘制三角形网格和法向量箭头
o3d.visualization.draw_geometries([triangle_mesh, arrow, pcd_cluster, points_pcd])


'''

# # 创建一个3x4x5的长方体网格
# box = o3d.geometry.create_box_geometry(30, 40, 5)
box_cloud = o3d.geometry.TriangleMesh.create_box(width=100.0, height=20.0, depth=3.0)
box_cloud.paint_uniform_color([0.9, 0.1, 0.1])

# # 计算点云的重心
centroid_cuboid = box_cloud.get_center()

# # 将点云平移到指定位置
box_cloud.translate(centroid_rib - centroid_cuboid)

num_points = 10000
box_cloud = box_cloud.sample_points_uniformly(num_points)
points = np.asarray(box_cloud.points)

o3d.visualization.draw_geometries([pcd, box_cloud], window_name="6")

'''

#######################   Create rectangular between each pair of bones   #######################

# 先根据所有的cluster的y max/min来判断这些cluster各自在centroid的上面还是下面
# 然后寻找上面的所有cluster中y min中的min值
# 寻找那个cluster中那个min值对应的xyz值
# 下面的cluster亦然
# 构造一个3个点所构成的平面，然后以centroid为中心，平面为平面生成一个长方体


possible_above_points = None
possible_below_points = None
min_x = np.inf
max_x = -np.inf
above_lowest_y = []
below_higest_y = []

# 创建点云对象
box_pcd1 = o3d.geometry.PointCloud()



# 遍历每个聚类
for i in range(max(clustering.labels_)+1):
    # 获取当前聚类的所有点
    indices = np.where(clustering.labels_ == i)[0]
    points = np.asarray(pcd.points)[indices, :]
    
    # 计算y方向上的最小值
    min_y = np.min(points[:, 1])
    idx = np.argmin(points[:, 1])

    x = points[idx, 0]
    y = points[idx, 1]
    z = points[idx, 2]

    if min_y > centroid_rib[1]:

        above_lowest_y.append(min_y)

        if x < min_x:
    
            min_x = x
            possible_above_points = [x,y,z]

        if np.abs(min_y-centroid_rib[1]) < 25:      # 30

            # 每次寻找到一个cluster，直接做出相应的补足四棱柱
            # 直接根据一边的四个点寻找按照centroid rib对称的四个点，直接做出四棱柱，即使同一行肋骨补了2次四棱柱也无所谓
            # 或者可以直接根据x min 和 x max的点位置，将他们的z各增加减少同样的值，构成长方体，但是要先尽量小一些，保证能相连就好
            # 先使用8个顶点的方法来构造长方体
            indice_surround = np.where((points[:, 1] >= min_y+2.8) & (points[:, 1] <= min_y+3.4))[0]      # 3 3.5
            surrounding_points = points[indice_surround, :]

            if surrounding_points.size != 0:

                idx_x_min = np.argmin(surrounding_points[:, 0])
                x_min = surrounding_points[idx_x_min, 0]
                z_x_min = surrounding_points[idx_x_min, 2]

                idx_x_max = np.argmax(surrounding_points[:, 0])
                x_max = surrounding_points[idx_x_max, 0]
                z_x_max = surrounding_points[idx_x_max, 2]

                idx_z_min = np.argmin(surrounding_points[:, 2])
                z_min = surrounding_points[idx_z_min, 2]
                x_z_min = surrounding_points[idx_z_min, 0]

                idx_z_max = np.argmax(surrounding_points[:, 2])
                z_max = surrounding_points[idx_z_max, 2]
                x_z_max = surrounding_points[idx_z_max, 0]
                
                above_4_points = [[x_min, min_y+3, z_x_min], [x_max, min_y+3, z_x_max], [x_z_min, min_y+3, z_min], [x_z_max, min_y+3, z_max]]
                
                # 需要按照一定的顺序：点和面
                vertices = np.array([
                    [x_min+1, min_y+1, z_x_min+3],
                    [x_min+1, min_y+1, z_x_min-3],
                    [x_max-1, min_y+1, z_x_max-3],
                    [x_max-1, min_y+1, z_x_max+3],
                    [x_min+1, centroid_rib[1]-(min_y+1-centroid_rib[1]), z_x_min+3],
                    [x_min+1, centroid_rib[1]-(min_y+1-centroid_rib[1]), z_x_min-3],
                    [x_max-1, centroid_rib[1]-(min_y+1-centroid_rib[1]), z_x_max-3],
                    [x_max-1, centroid_rib[1]-(min_y+1-centroid_rib[1]), z_x_max+3]
                ])

                # 定义长方体中的12个三角形面
                triangles = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [1, 5, 6],
                    [1, 6, 2],
                    [5, 4, 7],
                    [5, 7, 6],
                    [4, 0, 3],
                    [4, 3, 7],
                    [3, 2, 6],
                    [3, 6, 7],
                    [1, 0, 4],
                    [1, 4, 5]
                ])

                # 构造长方体的网格
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)

                pcd_mesh = mesh.sample_points_uniformly(10000)      # otherwise too sparse

                box_pcd1.points = o3d.utility.Vector3dVector(np.append(np.asarray(box_pcd1.points), np.asarray(pcd_mesh.points), axis=0))
                
                # 显示长方体
                o3d.visualization.draw_geometries([pcd, box_pcd1], window_name="above")
            else:
                pass

        else:
            pass


    if min_y < centroid_rib[1]:

        max_y = np.max(points[:, 1])
        idx = np.argmax(points[:, 1])

        below_higest_y.append(max_y)

        x = points[idx, 0]
        y = points[idx, 1]
        z = points[idx, 2]

        if np.abs(max_y-centroid_rib[1]) < 30:

            indice_surround = np.where((points[:, 1] >= max_y-3.5) & (points[:, 1] <= max_y-3))[0]
            surrounding_points = points[indice_surround, :]

            if surrounding_points.size != 0:

                idx_x_min = np.argmin(surrounding_points[:, 0])
                x_min = surrounding_points[idx_x_min, 0]
                z_x_min = surrounding_points[idx_x_min, 2]

                idx_x_max = np.argmax(surrounding_points[:, 0])
                x_max = surrounding_points[idx_x_max, 0]
                z_x_max = surrounding_points[idx_x_max, 2]

                idx_z_min = np.argmin(surrounding_points[:, 2])
                z_min = surrounding_points[idx_z_min, 2]
                x_z_min = surrounding_points[idx_z_min, 0]

                idx_z_max = np.argmax(surrounding_points[:, 2])
                z_max = surrounding_points[idx_z_max, 2]
                x_z_max = surrounding_points[idx_z_max, 0]
                
                below_4_points = [[x_min, max_y-3, z_x_min], [x_max, max_y-3, z_x_max], [x_z_min, max_y-3, z_min], [x_z_max, max_y-3, z_max]]
            
                # 需要按照一定的顺序：点和面
                vertices = np.array([
                    [x_min+1, max_y-1, z_x_min+3],
                    [x_min+1, max_y-1, z_x_min-3],
                    [x_max-1, max_y-1, z_x_max-3],
                    [x_max-1, max_y-1, z_x_max+3],
                    [x_min+1, centroid_rib[1]+(centroid_rib[1]-(max_y-1)), z_x_min+3],
                    [x_min+1, centroid_rib[1]+(centroid_rib[1]-(max_y-1)), z_x_min-3],
                    [x_max-1, centroid_rib[1]+(centroid_rib[1]-(max_y-1)), z_x_max-3],
                    [x_max-1, centroid_rib[1]+(centroid_rib[1]-(max_y-1)), z_x_max+3]
                ])

                # 定义12个三角形面
                triangles = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [1, 5, 6],
                    [1, 6, 2],
                    [5, 4, 7],
                    [5, 7, 6],
                    [4, 0, 3],
                    [4, 3, 7],
                    [3, 2, 6],
                    [3, 6, 7],
                    [1, 0, 4],
                    [1, 4, 5]
                ])

                # 构造长方体的网格
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)

                pcd_mesh = mesh.sample_points_uniformly(10000)

                box_pcd1.points = o3d.utility.Vector3dVector(np.append(np.asarray(box_pcd1.points), np.asarray(pcd_mesh.points), axis=0))

                # 显示长方体
                o3d.visualization.draw_geometries([pcd, box_pcd1], window_name="below")        
            else:
                pass
        
        else:
            pass


        if x > max_x:

            max_x = x
            possible_below_points = [x,y,z]


'''
# 使用 create_box() 函数创建一个默认方向的四棱柱：
quadrangular = o3d.geometry.TriangleMesh.create_box()
# 将四棱柱的顶点坐标设置为给定的8个顶点：
vertices = np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,1], [0,1,1], [1,1,1], [1,0,1]])
quadrangular.vertices = o3d.utility.Vector3dVector(vertices)

# 将四棱柱的方向进行调整，使其与给定的顶点匹配：
# 首先找到底面的中心点坐标
v = vertices[:4]
bottom_center = np.mean(v, axis=0)     # bottom_center = [(v[0]+v[1]+v[2]+v[3])/4 for v in vertices[:4]]
# 然后计算底面法向量
bottom_normal = o3d.geometry.TriangleMesh.compute_vertex_normals(quadrangular)[0][0:4]
# 计算底面法向量和 $y$ 轴的夹角，调整四棱柱的旋转角度
angle = np.arccos(np.dot(bottom_normal, [0,1,0]) / (np.linalg.norm(bottom_normal)*np.linalg.norm([0,1,0])))
axis = np.cross(bottom_normal, [0,1,0])
axis /= np.linalg.norm(axis)

R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis, angle)
quadrangular.rotate(R, center=bottom_center)

pcd.points = o3d.utility.Vector3dVector(np.append(np.asarray(pcd.points), np.asarray(quadrangular.points), axis=0))
o3d.visualization.draw_geometries([pcd])
'''

####################   Create the middle rectangular   #####################

min_bound = []
max_bound = []

# # 定义长方体区域的最小坐标和最大坐标
min_bound.append(min(possible_above_points[0],possible_below_points[0]))
min_bound.append(centroid_rib[1]-14)         # max(below_higest_y)-6
min_bound.append(centroid_rib[2]-4)          # min(possible_above_points[2],possible_below_points[2])

max_bound.append(max(possible_above_points[0],possible_below_points[0]))
max_bound.append(centroid_rib[1]+14)         # min(above_lowest_y)+6
max_bound.append(centroid_rib[2]+4)          # max(possible_above_points[2],possible_below_points[2])


# # 在长方体区域内均匀采样生成点云
new_points = np.random.uniform(min_bound, max_bound, size=(150000, 3))


# 创建点云对象
box_pcd2 = o3d.geometry.PointCloud()
box_pcd2.points = o3d.utility.Vector3dVector(new_points)


# 定义裁剪框, 实现中空的长方体
min_bound_crop = [x + 1.5 for x in min_bound] 
max_bound_crop = [x - 1.5 for x in max_bound]  


new_points = np.random.uniform(min_bound_crop, max_bound_crop, size=(10000, 3))
cropped_pcd = o3d.geometry.PointCloud()
cropped_pcd.points = o3d.utility.Vector3dVector(new_points)


# 使用点云2创建裁剪框
bbox = cropped_pcd.get_axis_aligned_bounding_box()

# 获取边界框内的点的索引
indices = np.where(np.logical_and.reduce((
        np.asarray(box_pcd2.points)[:, 0] >= bbox.min_bound[0],
        np.asarray(box_pcd2.points)[:, 1] >= bbox.min_bound[1],
        np.asarray(box_pcd2.points)[:, 2] >= bbox.min_bound[2],
        np.asarray(box_pcd2.points)[:, 0] <= bbox.max_bound[0],
        np.asarray(box_pcd2.points)[:, 1] <= bbox.max_bound[1],
        np.asarray(box_pcd2.points)[:, 2] <= bbox.max_bound[2]
    )))[0]

# 获取保留的点的索引
remain_indices = set(range(np.asarray(box_pcd2.points).shape[0])) - set(indices)
box_pcd2 = box_pcd2.select_by_index(list(remain_indices))


# 计算旋转角度
angle = np.arccos(normal.dot([0,0,1]) / np.linalg.norm(normal))     # 0.1860106104274823

# 计算旋转轴的方向向量
axis_rotate = np.cross([0,0,1], normal)
axis_rotate = axis_rotate / np.linalg.norm(axis_rotate)             # array([-0.00397526, -0.9999921 ,  0. ])

if axis_rotate[1] < 0:
    angle = -angle


# 构造旋转矩阵
R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, angle, 0]).T)

# 将长方体应用旋转矩阵
box_pcd2.rotate(R)




# 使用凸包生成封闭点云
# hull, _ = box_pcd.compute_convex_hull()
# box_pcd += hull   # o3d.visualization.draw_geometries([hull])

# 将不密封的点集转换成密封的三维模型可以通过几何体重建算法实现，这些算法通常使用点云数据来生成一个具有几何形状的三角形网格模型

# box_pcd.points.extend(hull.vertices)        # 实现将点云和三角网格合并成一个整体的效果
# o3d.geometry.PointCloud.points.extend()
# o3d.geometry.TriangleMesh.vertices.extend()

# o3d.visualization.draw_geometries([box_pcd])

# volume = box_pcd.get_oriented_bounding_box().volume()

# o3d.io.write_point_cloud("H:/Ubuntu/results/point_cloud_03/cuboid.ply", box_pcd)
# o3d.io.write_point_cloud("H:/Ubuntu/results/point_cloud_03/rib.ply", pcd)


'''
box_pcd.estimate_normals()
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(box_pcd, depth=12)

o3d.io.write_triangle_mesh("H:/Ubuntu/results/point_cloud_03/cuboid.ply", mesh)

pcd.estimate_normals()
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

o3d.io.write_triangle_mesh("H:/Ubuntu/results/point_cloud_03/rib.ply", mesh)
'''

# plane = triangle_mesh.plane_equation
# obb = o3d.geometry.TriangleMesh.create_from_triangle_mesh(triangle_mesh)

# # 获取三角形网格的顶点
# vertices = np.asarray(triangle_mesh.vertices)

# # 获取三角形网格的三角形索引
# triangles = np.asarray(triangle_mesh.triangles)

# # 计算三角形法向量的平均值
# triangle_normals = np.cross(vertices[triangles[:,1]] - vertices[triangles[:,0]], 
#                             vertices[triangles[:,2]] - vertices[triangles[:,0]])
# mean_normal = np.mean(triangle_normals, axis=0)

# # 计算平面方程
# a, b, c = mean_normal
# d = -np.dot(mean_normal, vertices[0])
# plane_equation = (a, b, c, d)

# # plane_equation = triangle_mesh.get_plane_equation()

# threshold = 2
# distances = box_pcd.compute_point_cloud_distance(plane_equation)

# box_pcd = box_pcd.select_by_index([i for i in range(len(box_pcd.points)) if distances[i] <= threshold])



# proj_points = np.dot(np.asarray(pcd.points), normal)

# z_max = np.max(proj_points)       # 要在添加长方体到pcd之前
# z_min = np.min(proj_points)


# 将新生成的点添加到原始点云的点集中
pcd.points = o3d.utility.Vector3dVector(np.append(np.asarray(pcd.points), np.asarray(box_pcd1.points), axis=0))
pcd.points = o3d.utility.Vector3dVector(np.append(np.asarray(pcd.points), np.asarray(box_pcd2.points), axis=0))

# z_mask = np.logical_and(np.asarray(pcd.points)[:, 2] > z_min, np.asarray(pcd.points)[:, 2] < z_max)
# # mask = np.logical_and(x_mask, np.logical_and(y_mask, z_mask))

# indices = np.arange(len(z_mask))[z_mask]
# pcd = pcd.select_by_index(indices)


# 可视化结果
o3d.visualization.draw_geometries([pcd], window_name="6")
pcd_down = pcd.voxel_down_sample(voxel_size=0.5)
o3d.visualization.draw_geometries([pcd_down], window_name="7")

o3d.io.write_point_cloud("H:/Ubuntu/Final/03/final_registration.ply", pcd_down)


'''
# 创建可视化窗口并添加点云和边界框: o3d.geometry.AxisAlignedBoundingBox
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(cropped_pcd)
vis.add_geometry(bbox)

# 设置相机视角
ctr = vis.get_view_control()
ctr.set_front([0.0, 0.0, -1.0])
ctr.set_up([0.0, 1.0, 0.0])
ctr.set_lookat([0.0, 0.0, 0.0])
ctr.set_zoom(0.8)

# 运行可视化窗口
vis.run()
vis.destroy_window()
'''




