<h2>3D Segmentation Multi Object with Point-SAM</h2>

This repository contains a customized pipeline for fully automatic 3D instance segmentation using **Point-SAM**, applied to real-world point clouds (e.g., from BLK360). 
The goal is to extract object-level masks without any manual prompts.

## 📌 Project Objective

- Automatically generate instance masks from raw 3D point clouds using Point-SAM
- Eliminate the need for manual clicks or annotations
- Enable object-level segmentation for downstream 3D scene understanding

## 🧭 Pipeline Overview

```mermaid
graph TD
    A[Input .ply point cloud] --> B[Preprocessing]
    B --> C[Plane removal (RANSAC)]
    C --> D[Clustering (Euclidean DBSCAN/HDBSCAN)]
    D --> E[Prompt point sampling (FPS per cluster)]
    E --> F[Segmentation with Point-SAM]
    F --> G[NMS (IoU-based)]
    G --> H[Final instance masks]
