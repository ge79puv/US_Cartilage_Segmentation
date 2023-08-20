import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import cluster





def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]


'''
from stl import mesh

your_mesh = mesh.Mesh.from_file('C:/Users/16967/Desktop/mesh.stl')
# volume, cog, inertia = your_mesh.get_mass_properties()
xyz = (your_mesh.max_ - your_mesh.min_)
sizel = round(xyz[0] / 10, 2)
sizew = round(xyz[1] / 10, 2)
sizeh = round(xyz[2] / 10, 2)

print(xyz, sizel, sizew, sizeh)




import vtk


#ply体积测量
#vtkReader=vtk.vtkPolyDataReader()

vtkReader=vtk.vtkPLYReader()

vtkReader.SetFileName('H:/Ubuntu/Final_mesh.ply')              # only Final_mesh.ply

vtkReader.Update()

polydata=vtkReader.GetOutput()

mass=vtk.vtkMassProperties()

mass.SetInputData(polydata)

print ("表面积:",mass.GetSurfaceArea())
print ("体积:",mass.GetVolume())
'''

 




import numpy as np
import copy
import open3d as o3d
import trimesh
import time
import mayavi.mlab as mlab
import cv2
import vtk



# Crop the bone structure manuelly 
 
def demo_crop_geometry():

    print("手动裁剪点云示例")
 
    print("按键 K 锁住点云，并进入裁剪模式")
    print("用鼠标左键拉一个矩形框选取点云，或者用 《ctrl+左键单击》 连线形成一个多边形区域")
    print("按键 C 结束裁剪并保存点云")
    print("按键 F 解除锁定，恢复自由查看点云模式")
    print("17873123809")


    pcd = o3d.io.read_point_cloud("H:/Ubuntu/results/point_cloud_01/merge_ply_cpu_maximum/final_onlysoft.ply")
    # "H:/Ubuntu/results/point_cloud_01/merge_ply_cpu_maximum/final_onlysoft.ply"      
    # "H:/Ubuntu/results/Cartilage-graph-based-US-CT-Registration/data/3/surface.ply"
    o3d.visualization.draw_geometries_with_editing([pcd])
    np_pcd = np.asarray(pcd.points)


    '''
    # Project the point cloud to xy plane
    points = np_pcd
    A, B, C, D = (0,0,1,0)              # xy: (0,0,1,0) 
    distance = A**2 + B**2 + C**2
    t = -(A*points[:, 0] + B*points[:, 1] + C*points[:, 2] + D)/distance
    x = A*t + points[:, 0]
    y = B*t + points[:, 1]
    z = C*t + points[:, 2]
    project_point = np.array([x, y, z]).T 
    
    projection_xy = o3d.geometry.PointCloud()
    projection_xy.points = o3d.utility.Vector3dVector(project_point)
    projection_xy.paint_uniform_color([0, 0.3, 0])
    o3d.visualization.draw_geometries([projection_xy], window_name="projection_xy")


    # Save the projection image
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(projection_xy)
    vis.update_geometry(projection_xy)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('H:/Ubuntu/results/point_cloud_03/projection_xy.png')
    vis.destroy_window()


    # Get the contours
    img = cv2.imread('H:/Ubuntu/results/point_cloud_03/projection_xy.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(0,0,255),3)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.imwrite('H:/Ubuntu/results/point_cloud_03/contours.png', img)
    '''

    '''
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=16, origin=[-10, -621.37679548, 180.58237842])

    pcd.paint_uniform_color(color=[0, 0, 0])

    # 点云聚类
    n_clusters = 4    # 聚类簇数
    points = np.array(pcd.points)
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(points)
    labels = kmeans.labels_     # (975294,)  0  3

    # print(kmeans.cluster_centers_)   # 聚类中心点的顺序不固定
    # [[ -41.7811968  -651.37679548  173.58237842] 左下角
    # [  20.66825072 -661.19818397  187.86996241]  右下角
    # [  27.29621535 -588.80996944  184.13554534]  右上角
    # [ -46.7474217  -596.74847969  172.23990877]] 左上角

    # 显示颜色设置
    colors = np.random.randint(0, 255, size=(n_clusters, 3)) / 255
    colors = colors[labels]
    
    pcd_cluster = deepcopy(pcd)
    pcd_cluster.colors = o3d.utility.Vector3dVector(colors)


    # Points and Lines of center
    centers_points = np.array(kmeans.cluster_centers_)
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(centers_points)
    points_pcd.paint_uniform_color([0, 0.3, 0])
    o3d.visualization.draw_geometries([pcd_cluster, points_pcd], window_name="sklearn cluster")


    lines = [[0, 1], [1, 2], [2, 3],[3,0]] #连接的顺序，封闭链接
    color = [[1, 0, 0] for i in range(len(lines))]
    #绘制线条
    lines_centers = o3d.geometry.LineSet()
    lines_centers.lines = o3d.utility.Vector2iVector(lines)
    lines_centers.colors = o3d.utility.Vector3dVector(color) #线条颜色
    lines_centers.points = o3d.utility.Vector3dVector(centers_points)


    # Points and Lines of split
    point11 = kmeans.cluster_centers_[1] + 3*(kmeans.cluster_centers_[2] - kmeans.cluster_centers_[1])/8
    point22 = kmeans.cluster_centers_[1] + 5*(kmeans.cluster_centers_[2] - kmeans.cluster_centers_[1])/8
    point1 = kmeans.cluster_centers_[1] + (kmeans.cluster_centers_[1] - kmeans.cluster_centers_[0])/2
    point2 = kmeans.cluster_centers_[2] + (kmeans.cluster_centers_[2] - kmeans.cluster_centers_[3])/2
    split_points = np.array([point11, point22, point1, point2])

    split_pcd = o3d.geometry.PointCloud()
    split_pcd.points = o3d.utility.Vector3dVector(split_points)
    split_pcd.paint_uniform_color([0, 0.3, 0])


    lines = [[0, 2], [1, 3]] #连接的顺序，封闭链接
    color = [[1, 0, 0] for i in range(len(lines))]
    #绘制线条
    lines_split = o3d.geometry.LineSet()
    lines_split.lines = o3d.utility.Vector2iVector(lines)
    lines_split.colors = o3d.utility.Vector3dVector(color) #线条颜色
    lines_split.points = o3d.utility.Vector3dVector(split_points)

    
    # 点云可视化
    o3d.visualization.draw_geometries([pcd_cluster, mesh_frame, points_pcd, lines_centers, lines_split],  # , lines_pcd
                                         window_name="sklearn cluster")
    
    x1, y1, z1 = point1
    x11, y11, z11 = point11

    a1, b1, c1 = getLinearEquation(x1, y1, x11, y11)

    x2, y2, z2 = point2
    x22, y22, z22 = point22

    a2, b2, c2 = getLinearEquation(x2, y2, x22, y22)

    print(np_pcd.shape)  # (975294, 3)
    # np_pcd_copy = deepcopy(np_pcd)
    delete_indices = []
    all_indices = []

    for i in range(len(np_pcd)):
        np_point = np_pcd[i]
        x, y, z = np_point
        all_indices.append(i)

        if (a1*x + b1*y + c1) > 0 and x > point11[0] and y < point11[1]:
            # np_pcd = np.delete(np_pcd, np.where(np_pcd == np_point))
            delete_indices.append(i)

        if (a2*x + b2*y + c2) > 0 and x > point22[0] and y > point22[1]:
            # np_pcd = np.delete(np_pcd, np.where(np_pcd == np_point))
            delete_indices.append(i)


    np_pcd = np_pcd[list(set(all_indices)-set(delete_indices)), :]          # (911713, 3)
    select_pcd = o3d.geometry.PointCloud()
    select_pcd.points = o3d.utility.Vector3dVector(np_pcd)
    select_pcd.paint_uniform_color([0, 0.3, 0])
    o3d.visualization.draw_geometries_with_editing([select_pcd], window_name="select_pcd")    # 可以继续通过圆柱体来去除
    '''
    
    '''

    
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.paint_uniform_color(color=[0, 0, 0])


    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=6.0)

    cl, ind = pcd.remove_radius_outlier(nb_points=100, radius=1)

    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    # 选中的点为灰色，未选中点为红色
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # 可视化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


    # 点云聚类
    points = np.array(pcd.points)
    dbscan = cluster.DBSCAN(eps=2, min_samples=16)      
    # 使用DBSCAN算法进行聚类, 不确定cluster的数量，不稳定(可能有outlier, 但是可以进行剔除离群点的操作)，不同的肋骨的分割取决于2个参数的选择,
    # 而且如果2个骨头相连，将会被划分为一类
    dbscan.fit(points)
    labels = dbscan.labels_

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    print(dbscan.components_)

    # 显示颜色设置
    colors = np.random.randint(0, 255, size=(max(labels) + 1, 3)) / 255    # 需要设置为n+1类，否则会数据越界造成报错
    # print(colors, labels)
    colors = colors[labels]    # 很巧妙，为每个label直接分配一个颜色
    colors[labels < 0] = 0     # 噪点直接设置为0，用黑色显示
    pcd_cluster = deepcopy(pcd)
    pcd_cluster.colors = o3d.utility.Vector3dVector(colors)

    # 点云可视化
    o3d.visualization.draw_geometries([pcd_cluster],
                                         window_name="sklearn cluster",
                                         width=800,
                                         height=600)
    '''



    '''
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.09, min_points=10, print_progress=True)) 
        # eps=0.09 cluster=9    eps=0.08 cluster=5

    # print(labels,len(labels))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

    print(colors, colors.sum(), colors.max(), colors.min(), len(colors))
    colors[labels < 0] = 0   # w: black   wo: blue
    # print(colors)

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd], window_name='Clusters', width=800, height=600)
    '''


    ###############################

    pcd = o3d.io.read_point_cloud("H:/Ubuntu/results/point_cloud_01/fragments_cpu_maximum/cropped_7_label.ply")      
    o3d.visualization.draw_geometries_with_editing([pcd])

    # pcd = o3d.io.read_point_cloud("H:/Ubuntu/point_cloud/merge_ply/final.ply")      
    # o3d.visualization.draw_geometries_with_editing([pcd])

    # pcd = o3d.io.read_point_cloud("H:/Ubuntu/point_cloud/merge_ply/label.ply")      
    # o3d.visualization.draw_geometries_with_editing([pcd])
    
    pcd.estimate_normals()

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

    o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50,
                                top=50, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True,)

    o3d.io.write_triangle_mesh("H:/Ubuntu/point_cloud/fragments_cpu_maximum/cropped_7_triangle_label_poisson.ply", mesh)
    
    # # create the triangular mesh with the vertices and faces from open3d
    # tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
    #                         vertex_normals=np.asarray(mesh.vertex_normals))

    # trimesh.convex.is_convex(tri_mesh)
    
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([0.9, 0.1, 0.1])
    # o3d.visualization.draw_geometries([mesh])
    
    # area = mesh.get_surface_area()  # 计算表面积
    # volume = mesh.get_volume()      # 计算体积
    # print("表面积为：", area)
    # print("体积为：", volume)
    

demo_crop_geometry()
 


# Get the volumn of bone point clouds

vtkReader=vtk.vtkPLYReader()

vtkReader.SetFileName('H:/Ubuntu/results/point_cloud_01/fragments_cpu_maximum/cropped_7_triangle_label_poisson.ply')              # only ply  

vtkReader.Update()

polydata=vtkReader.GetOutput()

mass=vtk.vtkMassProperties()

mass.SetInputData(polydata)
mass.Update()

print ("表面积:",mass.GetSurfaceArea())
print ("体积:",mass.GetVolume())




'''

import open3d as o3d
mesh = o3d.io.read_triangle_mesh(r"H:/Ubuntu/label.stl")
o3d.io.write_triangle_mesh(r"H:/Ubuntu/label.ply", mesh) #指定保存的类型

'''

'''
import vtk 


vtkReader=vtk.vtkPLYReader()
vtkReader.SetFileName('C:/Users/16967/Desktop/fragments/cropped_3_merge.ply')
vtkReader.Update()
polydata = vtkReader.GetOutput()


tri_converter = vtk.vtkTriangleFilter()
tri_converter.SetInputDataObject(polydata)
tri_converter.Update()
tri_mesh = tri_converter.GetOutput()


mass_props = vtk.vtkMassProperties()
mass_props.SetInputDataObject(tri_mesh)
mass_props.Update()                 # vtkMassProperties.cxx:89  No data to measure...!
volume = mass_props.GetVolume()

print ('Calculated mesh volume using VTK library')

print(volume)  

'''



