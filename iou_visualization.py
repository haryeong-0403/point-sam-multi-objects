import open3d as o3d
import numpy as np

def visualize_overlapping_masks(xyz_np, masks, iou_thresh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)

    N = len(masks)
    mask_array = np.stack(masks)  # shape: (N, num_points)

    # IoU 계산
    iou_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            inter = np.logical_and(mask_array[i], mask_array[j]).sum()
            union = np.logical_or(mask_array[i], mask_array[j]).sum()
            iou = inter / union if union > 0 else 0
            iou_matrix[i, j] = iou_matrix[j, i] = iou

    # IoU가 높은 마스크들 pair로 수집
    overlap_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if iou_matrix[i, j] > iou_thresh]

    # 각 마스크를 다른 색으로 칠하고 시각화
    colors = np.zeros((xyz_np.shape[0], 3))

    for idx, (i, j) in enumerate(overlap_pairs):
        mask_i = mask_array[i]
        mask_j = mask_array[j]

        color_i = np.random.rand(3)
        color_j = np.random.rand(3)

        colors[mask_i == 1] = color_i
        colors[mask_j == 1] = color_j

        # optional: IoU 값 프린트
        print(f"Overlap ({i}, {j}) IoU: {iou_matrix[i, j]:.2f}")

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


