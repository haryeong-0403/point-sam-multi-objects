import open3d as o3d
import numpy as np
import random

def object_points_visualization(Object_points):
    # numpy → open3d PointCloud 객체로 변환
    pcd_object = o3d.geometry.PointCloud()
    pcd_object.points = o3d.utility.Vector3dVector(Object_points)

    # 색을 입히고 싶다면 예시 (회색):
    pcd_object.paint_uniform_color([0.0, 0.0, 1.0])

    # 시각화
    o3d.visualization.draw_geometries([pcd_object])

def visualize_object_clusters_and_centroids(clusters, centroids):
    """
    - object 클러스터는 랜덤 색상으로 시각화
    - 중심점은 빨간 점 (sphere)으로 표시
    """

    geometries = []

    # 클러스터들 각각을 다른 색상으로 시각화
    for cluster in clusters:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster)
        
        # 랜덤 색상 지정
        color = np.random.rand(3)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (cluster.shape[0], 1)))
        geometries.append(pcd)

    # 중심점들을 빨간색 구체(sphere)로 시각화
    for center in centroids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색
        geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries)

def visualize_planes_clusters_and_centroids(wall_cluster, centorids):
    """
    - wall cluster는 빨간색으로 시각화
    - 중심점은 검은색 점으로 표시
    """
    geometries = []


    if isinstance(wall_cluster, list):
        wall_points = np.vstack(wall_cluster)

    else:
        wall_points = wall_cluster

    wall_pcd = o3d.geometry.PointCloud()
    wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
    wall_pcd.paint_uniform_color([1, 0, 0]) # Red
    geometries.append(wall_pcd)

    for center in centorids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(center)
        sphere.paint_uniform_color([0.0, 0.0, 0.0])
        geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries)

def visualized_planes(wall_planes, floor_planes, ceiling_planes):
    geometries = []

    if len(wall_planes) > 0:
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(np.vstack(wall_planes))
        wall_pcd.paint_uniform_color([1, 0, 0])  # 빨강
        geometries.append(wall_pcd)

    if len(floor_planes) > 0:
        floor_pcd = o3d.geometry.PointCloud()
        floor_pcd.points = o3d.utility.Vector3dVector(np.vstack(floor_planes))
        floor_pcd.paint_uniform_color([0, 1, 0])  # 초록
        geometries.append(floor_pcd)

    if len(ceiling_planes) > 0:
        ceiling_pcd = o3d.geometry.PointCloud()
        ceiling_pcd.points = o3d.utility.Vector3dVector(np.vstack(ceiling_planes))
        ceiling_pcd.paint_uniform_color([0, 0, 1])  # 파랑
        geometries.append(ceiling_pcd)

    o3d.visualization.draw_geometries(geometries)