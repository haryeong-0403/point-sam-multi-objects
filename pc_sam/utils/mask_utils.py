import numpy as np

def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks (numpy arrays)."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def apply_nms(masks, iou_thresh=0.3):
    """
    Apply IoU-based Non-Maximum Suppression to a list of binary masks.

    Args:
        masks: list of np.ndarray, shape (N,) each with shape (num_points,)
        iou_thresh: IoU threshold for suppression

    Returns:
        List of kept masks
    """
    keep = []
    for i, m in enumerate(masks):
        if all(compute_iou(m, masks[j]) < iou_thresh for j in keep):
            keep.append(i)
    return [masks[i] for i in keep]
