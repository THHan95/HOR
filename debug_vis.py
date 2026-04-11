#!/usr/bin/env python3
"""Debug script to check visibility mask propagation"""

import torch
import sys
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent / "lib" / "models" / "common"))

from lib.datasets import create_dataset
from lib.utils.config import get_config
from lib.opt import parse_exp_args

def debug_vis():
    """Check if visibility masks are present in batch"""
    # Use default config
    cfg_file = "config/release/HOR_DexYCBMV.yaml"
    cfg = get_config(config_file=cfg_file, arg=None, merge=True)

    # Create dataset
    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)

    # Get one sample
    sample = train_data[0]

    print("=" * 80)
    print("Checking visibility masks in batch")
    print("=" * 80)

    # Check for visibility keys
    vis_keys = [k for k in sample.keys() if 'vis' in k.lower()]
    print(f"\nKeys containing 'vis': {vis_keys}")

    if 'target_joints_vis' in sample:
        joints_vis = sample['target_joints_vis']
        print(f"\ntarget_joints_vis shape: {joints_vis.shape}")
        print(f"target_joints_vis dtype: {joints_vis.dtype}")
        print(f"target_joints_vis min/max: {joints_vis.min():.4f} / {joints_vis.max():.4f}")
        print(f"target_joints_vis sum: {joints_vis.sum():.4f}")
        print(f"target_joints_vis (first 5): {joints_vis[:5]}")
    else:
        print("\n❌ target_joints_vis NOT FOUND in batch!")

    if 'target_obj_center_vis' in sample:
        obj_vis = sample['target_obj_center_vis']
        print(f"\ntarget_obj_center_vis shape: {obj_vis.shape}")
        print(f"target_obj_center_vis dtype: {obj_vis.dtype}")
        print(f"target_obj_center_vis min/max: {obj_vis.min():.4f} / {obj_vis.max():.4f}")
        print(f"target_obj_center_vis sum: {obj_vis.sum():.4f}")
        print(f"target_obj_center_vis: {obj_vis}")
    else:
        print("\n❌ target_obj_center_vis NOT FOUND in batch!")

    # Check related keys
    print("\n" + "=" * 80)
    print("Related keys:")
    print("=" * 80)

    if 'target_obj_center_uv' in sample:
        obj_uv = sample['target_obj_center_uv']
        print(f"\ntarget_obj_center_uv shape: {obj_uv.shape}")
        print(f"target_obj_center_uv dtype: {obj_uv.dtype}")
        print(f"target_obj_center_uv min/max: {obj_uv.min():.4f} / {obj_uv.max():.4f}")
        print(f"target_obj_center_uv values:\n{obj_uv}")

        # Check if values are in [0, 1]
        in_range = ((obj_uv >= 0) & (obj_uv <= 1)).all()
        print(f"All values in [0, 1]: {in_range}")

        # Denormalize to pixel coordinates
        output_size = (256, 256)  # Assuming standard size
        obj_uv_pixel = obj_uv * np.array(output_size)
        print(f"\nDenormalized to pixel coords (assuming {output_size}):")
        print(f"obj_uv_pixel min/max: {obj_uv_pixel.min():.4f} / {obj_uv_pixel.max():.4f}")
        print(f"obj_uv_pixel values:\n{obj_uv_pixel}")

        # Check visibility in pixel space
        vis_pixel = ((obj_uv_pixel >= 0) & (obj_uv_pixel < output_size[0])).all(axis=-1) & \
                    ((obj_uv_pixel >= 0) & (obj_uv_pixel < output_size[1])).all(axis=-1)
        print(f"Visibility in pixel space: {vis_pixel}")
    else:
        print("\n❌ target_obj_center_uv NOT FOUND in batch!")

if __name__ == "__main__":
    debug_vis()
