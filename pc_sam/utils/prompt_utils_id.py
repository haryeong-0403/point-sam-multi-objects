import numpy as np
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN
import sys
import os
from points_visualization import (
    object_points_visualization, 
    visualize_object_clusters_and_centroids, 
    visualized_planes,
    visualize_planes_clusters_and_centroids
    )

import hdbscan

current_file_dir = os.path.dirname(os.path.abspath(__file__))

point_sam_root_dir = os.path.dirname(os.path.dirname(current_file_dir))
sys.path.append(point_sam_root_dir)

def remove_plane(points, dist_threshold=0.06, ransac_n=3, num_iterations=2000):
    """
    RANSAC으로 평면 제거, 평면으로 간주된 나머지 point들은 object로 간주
    """
    Original_points = points.copy()
    wall_planes, floor_planes, ceiling_planes = [],[],[]

    # 최대 30개의 평면을 반복적으로 탐색
    for _ in range(50):
        if len(Original_points) < ransac_n:
            break

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Original_points))
        plane_models, inliers = pcd.segment_plane(dist_threshold, ransac_n, num_iterations)
        # plane_models: [a,b,c,d] -> (ax+by+cz+d=0)
        # inliers: 평면 위에 있다고 판단된 포인트들의 인덱스

        if len(inliers) < 1000:
            # 평면의 크기가 너무 작으면 노이즈로 간주하고 무시
            continue

        plane_pts = Original_points[inliers]

        # RANSAC으로 찾은 평면의 법선 벡터(normal vector)를 정규화하는 과정
        normal = np.array(plane_models[:3]) 
        normal /= np.linalg.norm(normal)

        # 평면 포인트의 중심 좌표 계산 -> 위치 판단용
        center = plane_pts.mean(axis=0)

        # 해당 플래그는: "이번 평면이 우리가 제거할 평면인지?" 여부를 표시하는 변수
        is_plane_used = False
        
        if abs(normal[1]) > 0.9: # 법선이 거의 y축 방향(=수직) -> 바닥 또는 천장 후보
            if center[1] > 0.2: # 중심이 위쪽이면 천장 -> 리스트에 추가하고 제거 대상 표시
                ceiling_planes.append(plane_pts)
                is_plane_used =True

            else: 
                # 중심이 아래쪽이면 바닥
                floor_planes.append(plane_pts)
                is_plane_used = True

        elif abs(normal[0]) > 0.9 or abs(normal[2]) > 0.9:
            # x축 또는 z축 방향이면 -> 수직 평면(벽)
            wall_planes.append(plane_pts)
            is_plane_used = True

        if is_plane_used:
            Original_points = Original_points[np.setdiff1d(np.arange(len(Original_points)), inliers)]
    
    return Original_points, wall_planes, floor_planes, ceiling_planes

def cluster_object_points(points, min_popints=10):
    """
    object 후보 point들을 클러스터링
    """
    if len(points) < min_popints:
        print("prompt_utils_id/cluster_object_points의 point가 부족함")
        return []
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_popints)
    labels = clusterer.fit_predict(points)

    unique_labels = [l for l in set(labels) if l != -1]
    clusters = [points[labels == l] for l in unique_labels]

    return clusters


def sample_centroids(clusters, max_samples=None):
    """
    각 클러스터에서 중심점을 뽑아서 프롬프트로 사용
    """
    centroids = [c.mean(axis=0) for c in clusters if len(c) > 0]
    if max_samples is not None and len(centroids) > max_samples:
        indices = np.random.choice(len(centroids), max_samples, replace=False)
        centroids = [centroids[i] for i in indices]
    
    return centroids


def generate_prompt_points(xyz: np.ndarray, max_object=10, max_wall=32, max_floor=32, max_ceiling=32):
    """
    입력 함수 xyz로부터 prompt point, label, instance ID를 자동 생성
    넘어오는 인자로부터 각 영역에서 뽑은 최대 prompt 수를 제한
    """

    # 평면 제거 함수 실행
    object_xyz, wall_planes, floor_planes, ceiling_planes = remove_plane(xyz)

    #####################Visualization############################
    # object라고 판단된 point들 시각화
    # object_points_visualization(object_xyz)
    
    # Planes이라고 판단된 poin들 시각화
    # visualized_planes(wall_planes, floor_planes, ceiling_planes)
    ###############################################################
    
    # object point들 클러스터링 진행
    object_clusters = cluster_object_points(object_xyz)
    
    # 각 object cluster에서 중심 좌표를 추출, 최대 max_object로 prompt 수 제한
    object_prompts = sample_centroids(object_clusters, max_object)
    visualize_object_clusters_and_centroids(object_clusters, object_prompts)  # clustering + centorid point 시각화

    wall_prompts = sample_centroids(wall_planes, max_wall)
    # visualize_planes_clusters_and_centroids(wall_planes, wall_prompts)
    
    floor_prompts = sample_centroids(floor_planes, max_floor)
    
    ceiling_prompts = sample_centroids(ceiling_planes, max_ceiling)

    # 위에서 뽑은 prompt point를 하나의 리스트로 결합
    prompt_points = object_prompts + wall_prompts + floor_prompts + ceiling_prompts

    # 각 영역별로 binary label 생성
    object_labels  = [1] * len(object_prompts)      # foreground
    wall_labels    = [0] * len(wall_prompts)        # background
    floor_labels   = [0] * len(floor_prompts)
    ceiling_labels = [0] * len(ceiling_prompts)
    
    prompt_labels = object_labels + wall_labels + floor_labels + ceiling_labels

    # instance ID를 생성하기 위한 영역별 오프셋 값 정의.
    instance_offset = {
        'object': 0,
        'wall': 1000,
        'floor': 2000,
        'ceiling': 3000,
    }

    # 각 prompt에 대해 instance ID 부여
    object_ids  = [instance_offset['object'] + i + 1 for i in range(len(object_prompts))]
    print(f"object ids: {object_ids}")

    wall_ids    = [instance_offset['wall'] + i + 1 for i in range(len(wall_prompts))]
    print(f"wall ids: {wall_ids}")
    
    floor_ids   = [instance_offset['floor'] + i + 1 for i in range(len(floor_prompts))]
    print(f"floor ids: {floor_ids}")
    
    ceiling_ids = [instance_offset['ceiling'] + i + 1 for i in range(len(ceiling_prompts))]
    print(f"ceiling ids: {ceiling_ids}")

    # instance ID도 point 순서에 맞춰 하나로 연결
    prompt_instance_ids = object_ids + wall_ids + floor_ids + ceiling_ids

    # prompt point들을 numpy 배열로 변환
    prompt_points_np = np.stack(prompt_points)
    
    # label과 instance ID를 PyTorch 텐서로 변환
    prompt_labels_tensor = torch.tensor(prompt_labels, dtype=torch.long)
    prompt_instance_ids_tensor = torch.tensor(prompt_instance_ids, dtype=torch.long)

    return prompt_points_np, prompt_labels_tensor, prompt_instance_ids_tensor
