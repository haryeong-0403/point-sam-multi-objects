# object-aware prompt sampling
# foreground/background 분리

import numpy as np
import open3d as o3d
import torch

def remove_place(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Use RANSAC for remove the floor, wall
    """

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) 
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    plane_points = points[inliers]
    object_points= points[np.setdiff1d(np.arange(len(points)), inliers)]
    
    return object_points, plane_points

def farthest_point_samples(points, k):
    """
    FPS Samling

    k: FPS로 뽑은 포인트수 
    """
    pts = torch.from_numpy(points).float() # Numpy를 torch로 변환
    centroids = torch.zeros(k, dtype=torch.long)
    distance = torch.ones(pts.shape[0]) * 1e10
    farthest = torch.randint(0, pts.shape[0], (), dtype=torch.long)

    for i in range(k):
        centroids[i] = farthest # i번째 FPS 샘플로 현재 선택된 farthest 인덱스 저장
        centroid = pts[farthest].unsqueeze(0) # 선택된 점의 좌표를 꺼냄 -> broadcasting위해 shape[1, 3]
        dist = torch.sum((pts - centroid) ** 2, dim=1)
        mask = dist < distance # 현재 거리보다 가까우면 마스크 표시

        distance[mask] = dist[mask] # 가장 가까운 거리만 유지 -> 나중에 멀리 있는 점을 찾기 위함

        farthest = torch.max(distance, dim=0)[1]

        return points[centroids.numpy()]
    

def generate_prompt_points(xyz: np.ndarray, num_fg=128, num_bg=64):
    """
    전체 포인트 중에서 foreground/background prompt를 각각 지정 수만큼 뽑음
    """

    object_xyz, plane_xyz = remove_place(xyz)
    # object_xyz: foreground 후보
    # plane_xyz: background 후보

    if object_xyz.shape[0] < num_fg or plane_xyz.shape[0] < num_bg:
        raise ValueError("Not enough points in object/plane regions for sampling")
        # 만약에 foreground나 background에서 FPS할 만큼의 포인트 수가 부족하면 오류 발생

    fg_prompts = farthest_point_samples(object_xyz, num_fg) # 객체 영역에서 foreground용 fps 실행
    bg_prompts = farthest_point_samples(plane_xyz, num_bg) # 평면 영역에서 background용 fps 실행

    prompt_points = np.vstack([fg_prompts, bg_prompts])

    prompt_label = torch.cat([        
        torch.ones(num_fg, dtype=torch.long),
        torch.zeros(num_bg, dtype=torch.long)
        ], dim=0)

    return prompt_points, prompt_label