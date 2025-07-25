# point prompt 하나씩 넣고 inference 
# 유일하게 여러 개의 객체가 검출이 됨!

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

from pc_sam.utils.prompt_utils_250720 import farthest_point_sampling
from pc_sam.utils.mask_utils import apply_nms


def load_ply_xyzrgb(path):
    plydata = PlyData.read(path)

    xyz=np.stack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')], axis=1).astype(np.float32)
    rgb = np.stack([plydata['vertex'][axis] for axis in ('red', 'green', 'blue')], axis=1).astype(np.float32)

    return xyz, rgb

def generate_object_proposals(model, coords, colors, num_prompts=1024, nms_threshold = 0.3, top_k=250):
    # FPS로 1024개 포인트 샘플링
    sampled_idx = farthest_point_sampling(coords, num_prompts)  # Point-SAM에서 멀리 떨어진 점 1024개 선택 -> output:[1,1024] batch가 1이라서 
    prompt_points = coords[0, sampled_idx[0]].detach().cpu()    # [1024, 3]
    prompt_labels = torch.ones(prompt_points.shape[0], dtype=torch.long)  # 각 prompt point마다 라벨 1 부여 = foreground point

    all_masks = [] # 모든 마스크를 저장할 리스트 = 1024 * 3

    model.eval()

    with torch.no_grad(): # back propagation을 위해 그래디언트 계산을 끔 -> inference(추론) 과정에서는 필요 없음

        for i in range(prompt_points.shape[0]):
            print(f"{i} 번째 point 진행 중")
            prompt = prompt_points[i].unsqueeze(0).unsqueeze(0).cuda() # [1,3] change to [1,1,3]
            label = prompt_labels[i].unsqueeze(0).unsqueeze(0).cuda() # [1,1]

            output = model.inference(coords, colors, prompt_coords=prompt, prompt_labels=label)
            
            print("--------------------output-----------------------------\n")
            print(output)
            print("-------------------------------------------------\n")

            masks = output[0]["masks"].squeeze(0).cpu().numpy()

            all_masks.extend(list(masks))

    print("---------all_masks----------------\n")
    print(all_masks)
    print(f"Total raw masks before NMS: {len(all_masks)}")
    print("------------------------------------------")

    # Binary threshold 후 NMS 적용
    binary_masks = [(m > 0).astype(np.uint8) for m in all_masks]
    final_masks = apply_nms(binary_masks, iou_thresh=nms_threshold)

    # 마스크 크기 기준 정렬
    sorted_masks = sorted(final_masks, key=lambda m: m.sum(), reverse=True)

    print(f"Final masks after NMS: {len(sorted_masks)}")
    return sorted_masks[:top_k]  # 상위 250개만 사용

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="large", help="path to config file")
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="../outputs/output_250720_v1.npy")
    parser.add_argument("--num_prompts", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--nms_thresh", type=float, default=0.9)
    args = parser.parse_args()

    with hydra.initialize(config_path=args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)

    set_seed(cfg.get("seed", 42))
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    load_model(model, args.ckpt_path)
    model = model.cuda()

    xyz_np, rgb_np = load_ply_xyzrgb(args.input_path)
    coords = torch.from_numpy(xyz_np).unsqueeze(0).cuda()        # [1, N, 3]
    colors = torch.from_numpy(rgb_np / 255.0).unsqueeze(0).cuda()  # [1, N, 3]

    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords / coords.norm(dim=2, keepdim=True).max()

    masks = generate_object_proposals(model, coords, colors, args.num_prompts, args.nms_thresh, args.top_k)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, np.stack(masks, axis=0))  # [K, N]
    
    print(f"Saved final {len(masks)} masks to: {args.output_path}")

if __name__ == "__main__":
    main()