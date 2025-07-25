# Original inference code 

import sys
sys.path.append(".")

import argparse
import torch
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from pc_sam.model.pc_sam_250715 import PointCloudSAM
# from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from safetensors.torch import load_model
from plyfile import PlyData
import numpy as np
import os

def load_ply_xyzrgb(path):
    plydata = PlyData.read(path)
    xyz = np.stack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')], axis=1).astype(np.float32)
    rgb = np.stack([plydata['vertex'][axis] for axis in ('red', 'green', 'blue')], axis=1).astype(np.float32)
    return xyz, rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="large", help="path to config file"
    )
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./pretrained/ours/mixture_10k_giant/model.safetensors",
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.npy")

    args = parser.parse_args()
    
    # Load configuration
    with hydra.initialize(config_path=args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)

    seed = cfg.get("seed", 42)

    # Setup model
    set_seed(seed)
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    load_model(model, args.ckpt_path)
    model = model.cuda()

    # Load input point cloud
    xyz_np, rgb_np = load_ply_xyzrgb(args.input_path)  # [N, 3], [N, 3]
    coords = torch.from_numpy(xyz_np).unsqueeze(0).cuda()        # [1, N, 3]
    colors = torch.from_numpy(rgb_np / 255.0).unsqueeze(0).cuda()  # [1, N, 3]

    # normalize coords
    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords / coords.norm(dim=2, keepdim=True).max()

    # Inference using inference() in pc_sam.py/PointCloudSAM
    model.eval()
    with torch.no_grad():
        outputs = model.inference(coords, colors)

    # Use the first iteration's first mask output
    pred_mask = outputs[0]["masks"][:, 0]  # [1, N]

    # 모델 출력은 logit(실수값) 형태
    pred_mask = pred_mask.squeeze(0).cpu().numpy()  

    # 후처리로 0을 기준으로 이진화(sigmoid > 0.5)
    binary_mask = (pred_mask > 0).astype(np.uint8)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, binary_mask)
    
    print(f"Saved binary mask: {args.output_path}")

if __name__ == "__main__":
    main()
