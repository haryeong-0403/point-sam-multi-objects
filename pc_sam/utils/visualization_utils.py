import numpy as np
from plyfile import PlyData, PlyElement

def save_masked_point_cloud(points: np.ndarray, masks: list, save_path: str):
    """
    Save the masked points into a .ply file with per-mask coloring.

    Args:
        points (np.ndarray): [N, 3] original point cloud
        masks (list of np.ndarray): list of binary masks (0/1), one per object
        save_path (str): path to save colored .ply
    """
    N = points.shape[0]
    colors = np.zeros((N, 3), dtype=np.uint8)

    color_palette = get_color_palette(len(masks))

    for i, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        colors[mask] = color_palette[i]

    # Save as PLY
    vertices = np.array(
        [tuple(p) + tuple(c) for p, c in zip(points, colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(save_path)
    print(f"Saved colored .ply to {save_path}")

def get_color_palette(k):
    """
    Return k distinct RGB colors (uint8) using matplotlib colormap.

    Args:
        k (int): number of colors

    Returns:
        np.ndarray: [k, 3] RGB uint8
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = (np.array([cmap(i % 20)[:3] for i in range(k)]) * 255).astype(np.uint8)
    return colors
