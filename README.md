<h2>3D Segmentation Multi Object with Point-SAM</h2>

This repository contains a customized pipeline for fully automatic 3D instance segmentation using **Point-SAM**, applied to real-world point clouds (e.g., from BLK360). 
The goal is to extract object-level masks without any manual prompts.

## 📌 Project Objective

- Automatically generate instance masks from raw 3D point clouds using Point-SAM
- Eliminate the need for manual clicks or annotations
- Enable object-level segmentation for downstream 3D scene understanding

## 🧭 Installation

1. conda 환경 생성 및 활성화

```bash

conda create -n point-sam python=3.10 -y
conda activate point-sam
```
⚠️ Python ≥3.8만 요구하지만, PyTorch 2.1+의 안정성과 호환성을 위해 3.10 추천
