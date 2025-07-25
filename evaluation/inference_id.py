# - 각 평면 및 오브젝트를 instance 단위로 분리하고 ID 부여
# - prompt label은 class id가 아닌 instance id로 구성
# - Point-SAM은 binary mask만 출력하므로, 후처리에서 ID 매핑

import sys
sys.path.append(".")

import argparse
import torch
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from pc_sam.model.pc_sam_250715 import PointCloudSAM
from safetensors.torch import load_model
from plyfile import PlyData
import numpy as np
import os

from pc_sam.utils.prompt_utils_id import generate_prompt_points 
from pc_sam.utils.mask_utils_id import apply_nms
from iou_visualization import visualize_overlapping_masks

current_file_dir = os.path.dirname(os.path.abspath(__file__))

point_sam_root_dir = os.path.dirname(os.path.dirname(current_file_dir))
sys.path.append(point_sam_root_dir)


def load_ply_xyzrgb(path):
    plydata = PlyData.read(path)
    xyz = np.stack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')], axis=1).astype(np.float32)
    rgb = np.stack([plydata['vertex'][axis] for axis in ('red', 'green', 'blue')], axis=1).astype(np.float32)
    return xyz, rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="large")
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="../outputs/output_250725_v13.npy")
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--nms_thresh", type=float, default=0.2)
    args = parser.parse_args()

    with hydra.initialize(config_path=args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)

    set_seed(cfg.get("seed", 42))
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    load_model(model ,args.ckpt_path)
    model = model.cuda()
    model.eval()

    xyz_np, rgb_np = load_ply_xyzrgb(args.input_path)
    prompt_points_np, prompt_labels_np, instance_ids_np = generate_prompt_points(xyz_np)

    center = xyz_np.mean(axis=0)
    scale = np.linalg.norm(xyz_np - center, axis=1).max()

    # (N, 3) -> (1, N, 3) 형태로 변환하고 gpu로 이동
    coords = torch.from_numpy((xyz_np - center) / scale).unsqueeze(0).float().cuda()
    colors = torch.from_numpy((rgb_np / 255.0)).unsqueeze(0).float().cuda()

    # prompt point도 동일하게 정규화 + gpu로 전송 + label은 long 타입으로 변경
    prompt_points = torch.from_numpy((prompt_points_np - center) / scale).float().cuda()
    prompt_labels = prompt_labels_np.long().cuda()

    masks = [] # 각 prompt point에 대해 생성된 binary mask를 저장할 리스트 shape:N=
    instance_ids = [] # 각 마스크에 대응하는 instance ID 저장 리스트

    with torch.no_grad():
        for i in range(prompt_points.shape[0]): # 모든 prompt point를 순회하면서 하나씩 마스크 생성
            print(f"{i} 번째 prompt point 진행 중")

            single_point = prompt_points[i].unsqueeze(0).unsqueeze(0) # 하나의 prompt point를 선택하고, 차원을 (1,1,3)으로 확장 -> Point-SAM의 입력 형태
            single_label =  prompt_labels[i].view(1, 1).long().cuda() # class label 0 or 1

            output = model.inference(coords, colors, single_point, single_label)

            after_mask3 = output[0]["masks"][0]  # output에서 mask 추출 -> torch.Tensor to numpy -> shape:[3,N]

            for j in range(after_mask3.shape[0]):
                m = after_mask3[j].cpu().numpy()
                m = (m > 0).astype(np.uint8)
                masks.append(m)
                instance_ids.append(int(instance_ids_np[i].item()))

    final_masks = apply_nms(masks, instance_ids, iou_thresh=args.nms_thresh)
    sorted_masks = sorted(final_masks, key=lambda x: x[0].sum(), reverse=True)
    
    # Find IoU threshold values
    # visualize_overlapping_masks(xyz_np, masks, iou_thresh=args.nms_thresh)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, {
        "masks": np.stack([m for m, _ in sorted_masks]),
        "labels": np.array([id for _, id in sorted_masks], dtype=np.int32)
    })

    print(f"Saved {len(sorted_masks)} instance masks to {args.output_path}")

if __name__ == "__main__":
    main()