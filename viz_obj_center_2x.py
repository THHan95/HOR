#!/usr/bin/env python3
"""Visualize object center projection on multi-view images (2x zoom)"""

import torch
import sys
import numpy as np
import cv2
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent / "lib" / "models" / "common"))

from lib.datasets import create_dataset
from lib.utils.config import get_config

def viz_obj_center_2x():
    """Visualize object center on multi-view images with 2x zoom"""
    cfg_file = "config/release/HOR_DexYCBMV.yaml"
    cfg = get_config(config_file=cfg_file, arg=None, merge=True)

    # Create dataset
    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)

    # Get one sample
    sample = train_data[0]

    print("=" * 80)
    print("Visualizing object center on multi-view images (2x zoom)")
    print("=" * 80)

    # Get images and object center
    images = sample['image']  # (N, 3, H, W)
    obj_center_uv = sample['target_obj_center_uv']  # (N, 1, 2)
    obj_center_vis = sample.get('target_obj_center_vis', None)  # (N, 1)

    N, C, H, W = images.shape
    print(f"\nImages shape: {images.shape}")
    print(f"Object center UV shape: {obj_center_uv.shape}")

    # Create output directory
    output_dir = Path("tmp/obj_center_viz_2x")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zoom factor
    zoom = 2
    zoomed_H = H * zoom
    zoomed_W = W * zoom

    # Visualize each view
    for view_idx in range(N):
        img = images[view_idx]  # (3, H, W)
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

        # Denormalize
        if img.max() <= 1.0:
            img = (img + 1) / 2
        img = (img * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        # Zoom 2x
        img = cv2.resize(img, (zoomed_W, zoomed_H), interpolation=cv2.INTER_LINEAR)

        # Get object center
        obj_uv = obj_center_uv[view_idx, 0]  # (2,)
        obj_uv_pixel = ((obj_uv + 1) * np.array([W, H]) / 2) * zoom

        print(f"\nView {view_idx}:")
        print(f"  Object center UV (normalized): {obj_uv}")
        print(f"  Object center pixel coords (2x): {obj_uv_pixel}")
        print(f"  In bounds [0, {zoomed_W}] x [0, {zoomed_H}]: {0 <= obj_uv_pixel[0] < zoomed_W and 0 <= obj_uv_pixel[1] < zoomed_H}")

        # Draw object center
        if 0 <= obj_uv_pixel[0] < zoomed_W and 0 <= obj_uv_pixel[1] < zoomed_H:
            cv2.circle(img, (int(obj_uv_pixel[0]), int(obj_uv_pixel[1])), 15, (0, 255, 0), -1)
            cv2.circle(img, (int(obj_uv_pixel[0]), int(obj_uv_pixel[1])), 15, (255, 255, 255), 3)
            status = "VISIBLE"
        else:
            clamped_x = np.clip(obj_uv_pixel[0], 0, zoomed_W - 1)
            clamped_y = np.clip(obj_uv_pixel[1], 0, zoomed_H - 1)
            cv2.circle(img, (int(clamped_x), int(clamped_y)), 15, (0, 0, 255), -1)
            cv2.circle(img, (int(clamped_x), int(clamped_y)), 15, (255, 255, 255), 3)
            status = "OUT OF BOUNDS"

        # Add text
        cv2.putText(img, f"View {view_idx}: {status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(img, f"UV: ({obj_uv_pixel[0]:.1f}, {obj_uv_pixel[1]:.1f})", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Save
        output_path = output_dir / f"view_{view_idx:02d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Saved to {output_path}")

    print(f"\n✓ All visualizations saved to {output_dir}")

if __name__ == "__main__":
    viz_obj_center_2x()
