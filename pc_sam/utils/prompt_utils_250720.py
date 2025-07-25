import numpy as np
import torch

def sample_fps_prompts(points: np.ndarray, n: int = 1024) -> np.ndarray:
    """
    Sample n points using Farthest Point Sampling (FPS).

    Args:
        points (np.ndarray): [N, 3] input point cloud (xyz)
        n (int): number of prompt points to sample

    Returns:
        np.ndarray: [n, 3] sampled prompt points
    """
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # [1, N, 3]
    sampled_idx = farthest_point_sampling(points_tensor, n)        # [1, n]
    sampled_idx = sampled_idx.squeeze(0).cpu().numpy()             # [n]
    return points[sampled_idx]

def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for point cloud.

    Args:
        xyz (torch.Tensor): [B, N, 3] tensor of point cloud
        npoint (int): number of samples

    Returns:
        torch.Tensor: [B, npoint] indices of sampled points
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids
