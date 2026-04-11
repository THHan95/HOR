#!/usr/bin/env python3
"""Visualize object center projection on multi-view images"""

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

def viz_obj_center():
    """Visualize object center on multi-view images"""
    cfg_file = "config/release/HOR_DexYCBMV.yaml"
    cfg = get_config(config_file=cfg_file, arg=None, merge=True)

    # Create dataset
    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)

    # Get one sample
    sample = train_data[0]

    print("=" * 80)
    print("Visualizing object center on multi-view images")
    print("=" * 80)

    # Get images and object center
    images = sample['image']  # (N, 3, H, W)
    obj_center_uv = sample['target_obj_center_uv']  # (N, 1, 2)
    obj_center_vis = sample.get('target_obj_center_vis', None)  # (N, 1)

    N, C, H, W = images.shape
    print(f"\nImages shape: {images.shape}")
    print(f"Object center UV shape: {obj_center_uv.shape}")
    print(f"Object center vis shape: {obj_center_vis.shape if obj_center_vis is not None else 'None'}")

    # Create output directory
    output_dir = Path("tmp/obj_center_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each view with 2x zoom
    zoom_factor = 2
    zoomed_H = H * zoom_factor
    zoomed_W = W * zoom_factor
    for view_idx in range(N):
        img = images[view_idx]  # (3, H, W) - already numpy
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

        # Denormalize image (assuming it's normalized to [-1, 1] or [0, 1])
        if img.max() <= 1.0:
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = (img * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)  # Ensure C-contiguous for OpenCV

        # Get object center for this view
        obj_uv = obj_center_uv[view_idx, 0]  # (2,)
        obj_vis = obj_center_vis[view_idx, 0] if obj_center_vis is not None else 1.0

        # Denormalize to pixel coordinates
        obj_uv_pixel = (obj_uv + 1) * np.array([W, H]) / 2

        print(f"\nView {view_idx}:")
        print(f"  Object center UV (normalized): {obj_uv}")
        print(f"  Object center pixel coords: {obj_uv_pixel}")
        print(f"  Visibility: {obj_vis}")
        print(f"  In bounds: {0 <= obj_uv_pixel[0] < W and 0 <= obj_uv_pixel[1] < H}")

        # Draw object center
        if 0 <= obj_uv_pixel[0] < W and 0 <= obj_uv_pixel[1] < H:
            # In bounds - draw green circle
            cv2.circle(img, (int(obj_uv_pixel[0]), int(obj_uv_pixel[1])), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(obj_uv_pixel[0]), int(obj_uv_pixel[1])), 5, (255, 255, 255), 2)
        else:
            # Out of bounds - draw red circle at clamped position
            clamped_x = np.clip(obj_uv_pixel[0], 0, W - 1)
            clamped_y = np.clip(obj_uv_pixel[1], 0, H - 1)
            cv2.circle(img, (int(clamped_x), int(clamped_y)), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(clamped_x), int(clamped_y)), 5, (255, 255, 255), 2)

            # Draw arrow pointing to actual location
            if obj_uv_pixel[0] < 0:
                cv2.arrowedLine(img, (10, int(clamped_y)), (0, int(clamped_y)), (0, 0, 255), 2)
            elif obj_uv_pixel[0] >= W:
                cv2.arrowedLine(img, (W - 10, int(clamped_y)), (W - 1, int(clamped_y)), (0, 0, 255), 2)

            if obj_uv_pixel[1] < 0:
                cv2.arrowedLine(img, (int(clamped_x), 10), (int(clamped_x), 0), (0, 0, 255), 2)
            elif obj_uv_pixel[1] >= H:
                cv2.arrowedLine(img, (int(clamped_x), H - 10), (int(clamped_x), H - 1), (0, 0, 255), 2)

        # Add text
        status = "VISIBLE" if (0 <= obj_uv_pixel[0] < W and 0 <= obj_uv_pixel[1] < H) else "OUT OF BOUNDS"
        cv2.putText(img, f"View {view_idx}: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"UV: ({obj_uv_pixel[0]:.1f}, {obj_uv_pixel[1]:.1f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save image
        output_path = output_dir / f"view_{view_idx:02d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Saved to {output_path}")

    # Create a combined image
    print(f"\nCreating combined visualization...")
    combined = np.zeros((H * 2, W * 4, 3), dtype=np.uint8)

    for view_idx in range(N):
        img = images[view_idx]  # (3, H, W)
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

        if img.max() <= 1.0:
            img = (img + 1) / 2
        img = (img * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)  # Ensure C-contiguous for OpenCV

        obj_uv = obj_center_uv[view_idx, 0]
        obj_uv_pixel = obj_uv * np.array([W, H])

        if 0 <= obj_uv_pixel[0] < W and 0 <= obj_uv_pixel[1] < H:
            cv2.circle(img, (int(obj_uv_pixel[0]), int(obj_uv_pixel[1])), 5, (0, 255, 0), -1)
        else:
            clamped_x = np.clip(obj_uv_pixel[0], 0, W - 1)
            clamped_y = np.clip(obj_uv_pixel[1], 0, H - 1)
            cv2.circle(img, (int(clamped_x), int(clamped_y)), 5, (0, 0, 255), -1)

        row = view_idx // 4
        col = view_idx % 4
        combined[row * H:(row + 1) * H, col * W:(col + 1) * W] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    combined_path = output_dir / "combined.png"
    cv2.imwrite(str(combined_path), combined)
    print(f"Saved combined visualization to {combined_path}")

if __name__ == "__main__":
    viz_obj_center()
