import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.datasets import create_dataset
from lib.utils.config import get_config


def to_mm_stats(values_m):
    values_m = np.asarray(values_m, dtype=np.float64)
    values_mm = values_m * 1000.0
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = {
        "count": int(values_mm.size),
        "mean_mm": float(values_mm.mean()),
        "std_mm": float(values_mm.std()),
        "min_mm": float(values_mm.min()),
        "max_mm": float(values_mm.max()),
    }
    for p in percentiles:
        stats[f"p{p:02d}_mm"] = float(np.percentile(values_mm, p))
    return stats


def analyze_split(dataset_cfg, data_preset, split_name):
    dataset = create_dataset(dataset_cfg, data_preset=data_preset)
    source_set_name = f"{dataset.setup}_{dataset.data_split}"
    source_set = dataset.set_mappings[source_set_name]

    distances_all = []
    total = len(dataset.valid_sample_idx_list)
    for order_idx, valid_idx in enumerate(dataset.valid_sample_idx_list):
        multiview_id_list = dataset.multiview_sample_idxs[valid_idx]
        source_idx = multiview_id_list[0]
        obj_info = source_set.get_obj_pose_and_points(source_idx)
        t_label_rel = np.asarray(obj_info["t_label_rel"], dtype=np.float32)
        distances_all.append(float(np.linalg.norm(t_label_rel)))
        if (order_idx + 1) % 1000 == 0 or (order_idx + 1) == total:
            print(f"[{split_name}] sample {order_idx + 1}/{total}", flush=True)

    return np.asarray(distances_all, dtype=np.float32)


def save_outputs(distances_m, split_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = to_mm_stats(distances_m)

    with open(out_dir / f"{split_name}_root_to_obj_center_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(8, 5))
    plt.hist(np.asarray(distances_m) * 1000.0, bins=80, color="#2f6db3", edgecolor="white")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Count")
    plt.title(f"{split_name} - root_to_obj_center")
    plt.tight_layout()
    plt.savefig(out_dir / f"{split_name}_root_to_obj_center.png", dpi=160)
    plt.close()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/release/HOR_DexYCBMV_sv_tri.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--out-dir", type=str, default="tmp/hor_sv_tri_hand_object_distance")
    args = parser.parse_args()

    cfg = get_config(args.cfg, merge=False)
    out_dir = Path(args.out_dir)
    split_cfg_map = {
        "train": cfg.DATASET.TRAIN,
        "test": cfg.DATASET.TEST,
    }

    all_summary = {}
    for split_name in args.splits:
        distances = analyze_split(
            dataset_cfg=split_cfg_map[split_name],
            data_preset=cfg.DATA_PRESET,
            split_name=split_name,
        )
        all_summary[split_name] = save_outputs(distances, split_name, out_dir)

    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
