# object-aware prompt sampling
# foreground/background 분리


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

from pc_sam.utils.prompt_utils_250722 import generate_prompt_points
from pc_sam.utils.mask_utils import apply_nms


def load_ply_xyzrgb(path):
    plydata = PlyData.read(path)

    xyz=np.stack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')], axis=1).astype(np.float32)
    rgb = np.stack([plydata['vertex'][axis] for axis in ('red', 'green', 'blue')], axis=1).astype(np.float32)

    return xyz, rgb

def generate_object_proposals(model, coords, colors, prompt_points, prompt_labels,
                              nms_threshold, top_k):
    """
    raw point cloud 상에서 object-aware prompt point를 extract
    가장 멀리 떨어진 포인트(FPS) -> 객체의 실제 3D 위치를 기반으로 샘플링 진행
    """
    all_masks = [] # prompt point의 갯수인 1024개의 각각의 마스크 3개를 저장할 리스트
    
    model.eval()

    with torch.no_grad():
        for i in range(prompt_points.shape[1]):
            print(f"prompt_points.shape[1]: {prompt_points.shape[1]}")
            print(f"{i} 번째 point 진행 중")
            
            prompt = prompt_points[:, i, :].unsqueeze(1) # shape: [1,1,3]
            label = prompt_labels[:, i].unsqueeze(1) # shape: [1,1]

            output = model.inference(coords, colors, prompt_coords=prompt, prompt_labels=label)

            print("--------------------output-----------------------------\n")
            print(output)
            print("-------------------------------------------------\n")

            mask = output[0]["masks"].squeeze(0).cpu().numpy()

            all_masks.extend(list(mask))

    print("---------all mask----------------\n")
    print(all_masks)
    print(f"Total raw masks before NMS: {len(all_masks)}")
    print("------------------------------------------")


    # Binary threshold 이후 NMS 적용
    binary_masks = [(m > 0).astype(np.uint8) for m in all_masks]
    final_masks = apply_nms(binary_masks, iou_thresh=nms_threshold)

    # 가장 큰 객체(많은 포인트를 포함한 마스크)부터 우선적으로 선택
    sorted_masks = sorted(final_masks, key=lambda m: m.sum(), reverse=True)

    print(f"Final Masks after NMS: {len(sorted_masks)}")

    return sorted_masks[:top_k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="large", help="path to config file")
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="../outputs/output_250721_v5.npy")
    # parser.add_argument("--num_prompts", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--nms_thresh", type=float, default=0.7)
    args = parser.parse_args()


    with hydra.initialize(config_path=args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)

    
    set_seed(cfg.get("seed", 42))
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    load_model(model ,args.ckpt_path)
    model = model.cuda()

    xyz_np, rgb_np = load_ply_xyzrgb(args.input_path) # 원본 Raw Point Cloud load

    # Prompt Point Sampling(Raw 좌표계 기준)
    prompt_points_np, prompt_labels = generate_prompt_points(xyz_np, num_fg=128,
                                                             num_bg=64)
    
    # Normalized 진행 
    center = xyz_np.mean(axis=0)
    scale = np.linalg.norm(xyz_np - center, axis=1).max()

    # 모델 입력용 좌표 정규화
    coords = torch.from_numpy((xyz_np - center) / scale).unsqueeze(0).float().cuda()        
    colors = torch.from_numpy((rgb_np.astype(np.float32) / 255.0)).unsqueeze(0).float().cuda()

    # Prompt도 똑같이 정규화
    prompt_points = torch.from_numpy((prompt_points_np - center) / scale).unsqueeze(0).float().cuda() # shape: [1,N,3]
    prompt_labels = prompt_labels.unsqueeze(0).long().cuda() # shape:[1,N]

    # Mask 생성
    masks = generate_object_proposals(model, coords, colors,
                                      prompt_points, prompt_labels,
                                      nms_threshold=args.nms_thresh, top_k=args.top_k)

    for i, m in enumerate(masks):
        print(f"[Mask {i}] Max: {m.max()}, Min: {m.min()}, Positive: {(m>0).sum()}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, np.stack(masks, axis=0))  
    print(f"Saved final {len(masks)} masks to: {args.output_path}")

if __name__ == "__main__":
    main()