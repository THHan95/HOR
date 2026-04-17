import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

from lib.datasets import create_dataset
from lib.utils.config import get_config


REPO_ROOT = Path(__file__).resolve().parents[1]
HORT_ROOT = Path("/media/hl/data/code/han/hort_train")
HORT_MANO_ROOT = HORT_ROOT / "common" / "mano" / "assets"
HORT_TRAIN_JSON = HORT_ROOT / "datasets" / "dexycb" / "dexycb_train_s0.json"
HORT_TEST_JSON = HORT_ROOT / "datasets" / "dexycb" / "dexycb_test_s0.json"
HORT_TRAIN_SPLIT = HORT_ROOT / "datasets" / "dexycb" / "splits" / "train_s0_29k.json"
HORT_TEST_SPLIT = HORT_ROOT / "datasets" / "dexycb" / "splits" / "test_s0_5k.json"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/release/HOR_DexYCBMV_sv_tri.yaml")
    parser.add_argument("--out-dir", type=str, default="tmp/hand_object_surface_distance")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["poem_train", "poem_test", "hort_train", "hort_test"],
        choices=["poem_train", "poem_test", "hort_train", "hort_test"],
    )
    return parser


def get_stats(values_m):
    values_m = np.asarray(values_m, dtype=np.float64)
    values_mm = values_m * 1000.0
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = {
        "count": int(values_mm.size),
        "mean_mm": float(values_mm.mean()),
        "std_mm": float(values_mm.std()),
        "min_mm": float(values_mm.min()),
        "max_mm": float(values_mm.max()),
        "ratio_le_5mm": float((values_mm <= 5.0).mean()),
        "ratio_le_10mm": float((values_mm <= 10.0).mean()),
        "ratio_le_20mm": float((values_mm <= 20.0).mean()),
    }
    for p in percentiles:
        stats[f"p{p:02d}_mm"] = float(np.percentile(values_mm, p))
    return stats


def save_histogram(values_m, title, out_path):
    values_mm = np.asarray(values_m, dtype=np.float64) * 1000.0
    plt.figure(figsize=(8, 5))
    plt.hist(values_mm, bins=80, color="#2f6db3", edgecolor="white")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def min_surface_distance_like_hort(hand_verts, obj_verts):
    hand_tree = KDTree(np.asarray(hand_verts, dtype=np.float32))
    dists, _ = hand_tree.query(np.asarray(obj_verts, dtype=np.float32))
    return float(dists.min())


def transform_points(verts, transform):
    verts = np.asarray(verts, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return (rot @ verts.T).T + trans[None, :]


def load_obj_mesh_cache(obj_file_map):
    cache = {}
    for obj_id, obj_path in obj_file_map.items():
        mesh = trimesh.load(obj_path, process=False)
        cache[int(obj_id)] = np.asarray(mesh.vertices, dtype=np.float32)
    return cache


def analyze_poem_split(cfg, split_name):
    dataset_cfg = cfg.DATASET.TRAIN if split_name == "train" else cfg.DATASET.TEST
    dataset = create_dataset(dataset_cfg, data_preset=cfg.DATA_PRESET)
    source_set_name = f"{dataset.setup}_{dataset.data_split}"
    source_set = dataset.set_mappings[source_set_name]
    obj_mesh_cache = load_obj_mesh_cache(source_set.dex_ycb.obj_file)

    distances = []
    total = len(dataset.valid_sample_idx_list)
    for order_idx, valid_idx in enumerate(dataset.valid_sample_idx_list):
        source_idx = dataset.multiview_sample_idxs[valid_idx][0]
        hand_verts = source_set.get_verts_3d(source_idx)
        obj_id, obj_transform = source_set.get_obj_info(source_idx)
        obj_verts = transform_points(obj_mesh_cache[int(obj_id)], obj_transform)
        distances.append(min_surface_distance_like_hort(hand_verts, obj_verts))
        if (order_idx + 1) % 1000 == 0 or (order_idx + 1) == total:
            print(f"[poem_{split_name}] {order_idx + 1}/{total}", flush=True)

    return np.asarray(distances, dtype=np.float32), source_set.dex_ycb.obj_file


def get_poem_obj_file_map(cfg):
    dataset_cfg = cfg.DATASET.TRAIN
    dataset = create_dataset(dataset_cfg, data_preset=cfg.DATA_PRESET)
    source_set_name = f"{dataset.setup}_{dataset.data_split}"
    source_set = dataset.set_mappings[source_set_name]
    return source_set.dex_ycb.obj_file


def load_hort_mano_layer():
    sys.path.insert(0, str(HORT_ROOT / "common"))
    sys.path.insert(0, str(HORT_ROOT))
    for key in list(sys.modules.keys()):
        if key == "mano" or key.startswith("mano."):
            del sys.modules[key]
    from mano.manolayer import ManoLayer

    mano_layer = ManoLayer(
        flat_hand_mean=False,
        ncomps=45,
        side="right",
        mano_root=str(HORT_MANO_ROOT),
        use_pca=True,
    )
    mano_layer.eval()
    return mano_layer


def reconstruct_hort_hand_verts(ann, mano_layer):
    hand_pose = torch.tensor(ann["hand_poses"], dtype=torch.float32).unsqueeze(0)
    hand_shape = torch.tensor(ann["hand_shapes"], dtype=torch.float32).unsqueeze(0)
    hand_trans = torch.tensor(ann["hand_trans"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        hand_verts, _, _, _, _ = mano_layer(hand_pose, hand_shape, hand_trans)
    return hand_verts[0].cpu().numpy().astype(np.float32)


def analyze_hort_split(json_path, split_path, obj_file_map):
    mano_layer = load_hort_mano_layer()
    obj_mesh_cache = load_obj_mesh_cache(obj_file_map)

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split_ids = json.load(f)

    ann_by_id = {int(ann["id"]): ann for ann in json_data["annotations"]}
    distances = []
    missing_ids = []

    total = len(split_ids)
    for order_idx, split_id in enumerate(split_ids):
        ann = ann_by_id.get(int(split_id))
        if ann is None:
            missing_ids.append(int(split_id))
            continue
        hand_verts = reconstruct_hort_hand_verts(ann, mano_layer)
        obj_verts = transform_points(obj_mesh_cache[int(ann["ycb_id"])], ann["obj_transform"])
        distances.append(min_surface_distance_like_hort(hand_verts, obj_verts))
        if (order_idx + 1) % 1000 == 0 or (order_idx + 1) == total:
            print(f"[hort_{Path(split_path).stem}] {order_idx + 1}/{total}", flush=True)

    extra = {
        "split_len": int(len(split_ids)),
        "matched_annotation_ids": int(len(distances)),
        "missing_annotation_ids": int(len(missing_ids)),
    }
    if missing_ids:
        extra["missing_annotation_id_examples"] = missing_ids[:20]
    return np.asarray(distances, dtype=np.float32), extra


def save_result(values_m, out_dir, name, extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", np.asarray(values_m, dtype=np.float32))
    stats = get_stats(values_m)
    if extra is not None:
        stats.update(extra)
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    save_histogram(values_m, name, out_dir / f"{name}.png")
    return stats


def main():
    args = build_parser().parse_args()
    cfg = get_config(args.cfg, merge=False)
    out_dir = Path(args.out_dir)

    summary = {}
    poem_obj_file_map = None

    if "poem_train" in args.targets:
        values, poem_obj_file_map = analyze_poem_split(cfg, "train")
        summary["poem_train"] = save_result(values, out_dir, "poem_train")

    if "poem_test" in args.targets:
        values, poem_obj_file_map = analyze_poem_split(cfg, "test")
        summary["poem_test"] = save_result(values, out_dir, "poem_test")

    if poem_obj_file_map is None:
        poem_obj_file_map = get_poem_obj_file_map(cfg)

    if "hort_train" in args.targets:
        values, extra = analyze_hort_split(HORT_TRAIN_JSON, HORT_TRAIN_SPLIT, poem_obj_file_map)
        summary["hort_train"] = save_result(values, out_dir, "hort_train", extra=extra)

    if "hort_test" in args.targets:
        values, extra = analyze_hort_split(HORT_TEST_JSON, HORT_TEST_SPLIT, poem_obj_file_map)
        summary["hort_test"] = save_result(values, out_dir, "hort_test", extra=extra)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
