import numpy as np

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def apply_nms(masks, instance_ids, iou_thresh):

    keep = []
    for i, m in enumerate(masks):
        if all(compute_iou(m, masks[j]) < iou_thresh for j in keep):
            keep.append(i)

    if instance_ids is not None:
        return [(masks[i], instance_ids[i]) for i in keep]
    else:
        return [masks[i] for i in keep]
