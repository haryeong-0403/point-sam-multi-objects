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

from pc_sam.utils.prompt_utils_250722 import sample_fps_prompts  # <- 이게 있어야 함 (FPS)
from pc_sam.utils.mask_utils import apply_nms                     # <- NMS 모듈 (직접 작성 필요)
from pc_sam.utils.visualization_utils import save_masked_point_cloud  # <- 시각화 및 저장

def load_ply_xyzrgb(path):
    plydata = PlyData.read(path)
    xyz = np.stack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')], axis=1).astype(np.float32)
    rgb = np.stack([plydata['vertex'][axis] for axis in ('red', 'green', 'blue')], axis=1).astype(np.float32)
    return xyz, rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="large")
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument("--ckpt_path", type=str, default="./pretrained/ours/mixture_10k_giant/model.safetensors")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--num_prompts", type=int, default=1024)
    parser.add_argument("--iou_thresh", type=float, default=0.0)

    args = parser.parse_args()

    # Load configuration
    with hydra.initialize(config_path=args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)

    set_seed(cfg.get("seed", 42))

    # Load Model
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    load_model(model, args.ckpt_path)
    model = model.cuda().eval()

    # Load point cloud
    xyz_np, rgb_np = load_ply_xyzrgb(args.input_path)  # [N, 3], [N, 3]
    coords = torch.from_numpy(xyz_np).unsqueeze(0).cuda()
    colors = torch.from_numpy(rgb_np / 255.0).unsqueeze(0).cuda()

    # normalize coords
    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords / coords.norm(dim=2, keepdim=True).max()

    # Sample prompt points (FPS)
    prompt_points = sample_fps_prompts(xyz_np, n=args.num_prompts)  # [N_prompts, 3]

    # normalize prompt points: same with coords
    xyz_mean = xyz_np.mean(axis=0, keepdims=True)
    xyz_norm = np.linalg.norm(xyz_np - xyz_mean, axis=1).max()
    prompt_points_normalized = (prompt_points - xyz_mean) / xyz_norm
    print("prompt_points_normalized.shape:", prompt_points_normalized.shape)
    print("xyz_mean:", xyz_mean)
    print("xyz_norm: ", xyz_norm)
    
    output_dir = os.path.dirname(args.output_path)

    np.save(os.path.join(output_dir, "prompt_points_normalized.npy"), prompt_points_normalized)

    # Inference
    batch_size = 16
    all_masks = []

    with torch.no_grad():
        for i in range(args.num_prompts):
            single_prompt = prompt_points_normalized[i:i+1]  # (1, 3)
            single_prompt_tensor = torch.from_numpy(single_prompt).float().unsqueeze(0).cuda()  # (1, 1, 3)
            single_label = torch.ones((1, 1), dtype=torch.long).cuda()  # label = 1

            outputs = model.inference(coords, colors, prompt_coords=single_prompt_tensor, prompt_labels=single_label)
            print("output mask shape: ", outputs[0]["masks"].shape)

            pred_mask = outputs[0]["masks"][0, 0]  # shape: [N]
            binary_mask = (pred_mask.cpu().numpy() > 0).astype(np.uint8)
            all_masks.append(binary_mask)

    # Apply NMS
    final_masks = apply_nms(all_masks, iou_thresh=args.iou_thresh)

    # output mask 저장
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(os.path.join(output_dir, "segallobjs_masks_v3.npy"), final_masks)

    print(f"Save binary mask: {os.path.join(output_dir, 'segallobjs_masks_v3.npy')}")

    # Save Visualization
    # save_masked_point_cloud(xyz_np, final_masks,
                            # save_path=os.path.join(output_dir, "proposal_output_v3.ply"))

    print("----------------------------------------------------------\n")
    print(f"Final Saved {len(final_masks)} proposals to {output_dir}")
    print("----------------------------------------------------------\n")

if __name__ == "__main__":
    main()