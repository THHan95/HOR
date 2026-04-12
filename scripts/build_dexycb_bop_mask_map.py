#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build DexYCB raw-sample to BOP object-mask mapping.")
    parser.add_argument("--dexycb-root", type=Path, required=True, help="Path to the DexYCB dataset root.")
    parser.add_argument("--setup", type=str, default="s0", help="DexYCB setup, e.g. s0/s1/s2/s3.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train/val/test.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to common/cache/DexYCB/bop_mask_map_<setup>_<split>.json",
    )
    parser.add_argument("--limit-groups", type=int, default=None, help="Only process the first N (seq, cam) groups.")
    parser.add_argument("--frames-check", type=int, default=5, help="Number of frames per group used for scene matching.")
    parser.add_argument("--rot-atol", type=float, default=1e-4, help="Rotation matrix absolute tolerance.")
    parser.add_argument("--trans-atol-mm", type=float, default=0.5, help="Translation absolute tolerance in mm.")
    return parser.parse_args()


def ensure_toolkit_importable(repo_root: Path, dexycb_root: Path):
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("DEX_YCB_DIR", str(dexycb_root))
    from dex_ycb_toolkit.factory import get_dataset  # noqa: WPS433

    return get_dataset


def load_label(label_file: str):
    return np.load(label_file)


def pose_to_rt_mm(pose_3x4):
    rot = pose_3x4[:, :3].astype(np.float64)
    trans = (pose_3x4[:, 3] * 1000.0).astype(np.float64)
    return rot, trans


def frame_pose_records_from_sample(sample, label):
    pose_records = []
    for obj_idx, obj_id in enumerate(sample["ycb_ids"]):
        rot, trans = pose_to_rt_mm(label["pose_y"][obj_idx])
        pose_records.append({
            "obj_id": int(obj_id),
            "rot": rot,
            "trans_mm": trans,
        })
    return pose_records


def frame_pose_records_from_scene(scene_gt_entry):
    pose_records = []
    for ann_idx, ann in enumerate(scene_gt_entry):
        rot = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(ann["cam_t_m2c"], dtype=np.float64)
        pose_records.append({
            "ann_id": ann_idx,
            "obj_id": int(ann["obj_id"]),
            "rot": rot,
            "trans_mm": trans,
        })
    return pose_records


def match_pose_records(sample_records, scene_records, rot_atol, trans_atol_mm):
    if len(sample_records) != len(scene_records):
        return False

    used = set()
    for src in sample_records:
        matched = False
        for scene_idx, dst in enumerate(scene_records):
            if scene_idx in used:
                continue
            if src["obj_id"] != dst["obj_id"]:
                continue
            if not np.allclose(src["rot"], dst["rot"], atol=rot_atol, rtol=0.0):
                continue
            if not np.allclose(src["trans_mm"], dst["trans_mm"], atol=trans_atol_mm, rtol=0.0):
                continue
            used.add(scene_idx)
            matched = True
            break
        if not matched:
            return False
    return True


def find_matching_annotation(sample_obj_id, sample_pose_3x4, scene_records, rot_atol, trans_atol_mm):
    sample_rot, sample_trans = pose_to_rt_mm(sample_pose_3x4)
    for record in scene_records:
        if record["obj_id"] != int(sample_obj_id):
            continue
        if not np.allclose(sample_rot, record["rot"], atol=rot_atol, rtol=0.0):
            continue
        if not np.allclose(sample_trans, record["trans_mm"], atol=trans_atol_mm, rtol=0.0):
            continue
        return int(record["ann_id"])
    return None


def load_bop_scenes(dexycb_root: Path, setup: str, split: str):
    split_root = dexycb_root / "bop" / setup / split
    if not split_root.exists():
        raise FileNotFoundError(f"BOP split root not found: {split_root}")

    scenes = []
    for entry in sorted(split_root.iterdir()):
        if not entry.is_dir() and not entry.is_symlink():
            continue
        scene_path = entry.resolve()
        if not scene_path.exists():
            continue

        scene_camera_file = scene_path / "scene_camera.json"
        scene_gt_file = scene_path / "scene_gt.json"
        if not scene_camera_file.exists() or not scene_gt_file.exists():
            continue

        with scene_camera_file.open("r", encoding="utf-8") as f:
            scene_camera = json.load(f)
        with scene_gt_file.open("r", encoding="utf-8") as f:
            scene_gt = json.load(f)

        frame_ids = sorted(int(frame_id) for frame_id in scene_camera.keys())
        rgb_dir = scene_path / "rgb"
        if not rgb_dir.exists():
            continue

        rgb_lookup = {}
        for rgb_entry in sorted(rgb_dir.iterdir()):
            if not rgb_entry.is_file() and not rgb_entry.is_symlink():
                continue
            frame_id = int(rgb_entry.stem)
            raw_color_path = str(rgb_entry.resolve())
            rgb_lookup[raw_color_path] = frame_id

        scenes.append({
            "split_scene_id": int(entry.name),
            "data_scene_id": int(scene_path.name),
            "scene_path": scene_path,
            "frame_ids": frame_ids,
            "scene_gt": scene_gt,
            "rgb_lookup": rgb_lookup,
        })
    return scenes


def build_rgb_to_scene_index(scenes):
    rgb_to_scene = {}
    duplicate_raw_paths = []
    for scene in scenes:
        for raw_color_path, frame_id in scene["rgb_lookup"].items():
            if raw_color_path in rgb_to_scene:
                duplicate_raw_paths.append(raw_color_path)
                continue
            rgb_to_scene[raw_color_path] = {
                "split_scene_id": scene["split_scene_id"],
                "data_scene_id": scene["data_scene_id"],
                "scene_path": scene["scene_path"],
                "frame_id": int(frame_id),
                "scene_gt": scene["scene_gt"],
            }
    return rgb_to_scene, duplicate_raw_paths


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    get_dataset = ensure_toolkit_importable(repo_root, args.dexycb_root)

    if args.output is None:
        args.output = repo_root / "common" / "cache" / "DexYCB" / f"bop_mask_map_{args.setup}_{args.split}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(f"{args.setup}_{args.split}")
    scenes = load_bop_scenes(args.dexycb_root, args.setup, args.split)
    rgb_to_scene, duplicate_raw_paths = build_rgb_to_scene_index(scenes)
    groups = defaultdict(list)

    result = {
        "meta": {
            "setup": args.setup,
            "split": args.split,
            "dexycb_root": str(args.dexycb_root),
            "num_raw_samples": len(dataset),
            "num_bop_scenes": len(scenes),
            "num_rgb_links": len(rgb_to_scene),
            "trans_atol_mm": args.trans_atol_mm,
            "rot_atol": args.rot_atol,
            "duplicate_rgb_links": len(duplicate_raw_paths),
        },
        "groups": [],
        "samples": [],
        "errors": [],
    }

    raw_indices = list(range(len(dataset)))
    if args.limit_groups is not None:
        processed_group_keys = []
        seen_group_keys = set()
        for raw_idx, mapping in enumerate(dataset._mapping):
            group_key = (int(mapping[0]), int(mapping[1]))
            if group_key in seen_group_keys:
                continue
            seen_group_keys.add(group_key)
            processed_group_keys.append(group_key)
            if len(processed_group_keys) >= args.limit_groups:
                break
        allowed_groups = set(processed_group_keys)
        raw_indices = [
            raw_idx for raw_idx, mapping in enumerate(dataset._mapping)
            if (int(mapping[0]), int(mapping[1])) in allowed_groups
        ]
        result["meta"]["num_processed_groups"] = len(allowed_groups)

    for processed_idx, raw_idx in enumerate(raw_indices, start=1):
        sample = dataset[raw_idx]
        seq_id, cam_id, frame_id = (int(v) for v in dataset._mapping[raw_idx])
        group_key = (seq_id, cam_id)
        scene_match = rgb_to_scene.get(sample["color_file"], None)
        if scene_match is None:
            result["errors"].append({
                "raw_idx": int(raw_idx),
                "group": [seq_id, cam_id],
                "frame_id": frame_id,
                "color_file": os.path.relpath(sample["color_file"], start=args.dexycb_root),
                "error": "raw color path not found in BOP rgb links",
            })
            print(f"[{processed_idx}/{len(raw_indices)}] miss raw_idx={raw_idx} group={group_key} frame={frame_id}")
            continue

        label = load_label(sample["label_file"])
        grasp_ind = int(sample["ycb_grasp_ind"])
        obj_id = int(sample["ycb_ids"][grasp_ind])
        scene_records = frame_pose_records_from_scene(scene_match["scene_gt"][str(scene_match["frame_id"])])
        ann_id = find_matching_annotation(obj_id, label["pose_y"][grasp_ind], scene_records, args.rot_atol, args.trans_atol_mm)
        if ann_id is None:
            result["errors"].append({
                "raw_idx": int(raw_idx),
                "group": [seq_id, cam_id],
                "frame_id": frame_id,
                "color_file": os.path.relpath(sample["color_file"], start=args.dexycb_root),
                "error": (
                    f"matched rgb scene but failed to resolve grasp annotation "
                    f"(split_scene_id={scene_match['split_scene_id']}, data_scene_id={scene_match['data_scene_id']})"
                ),
            })
            print(f"[{processed_idx}/{len(raw_indices)}] ann_miss raw_idx={raw_idx} group={group_key} frame={frame_id}")
            continue

        scene_path = scene_match["scene_path"]
        frame_name = f"{scene_match['frame_id']:06d}"
        ann_name = f"{ann_id:06d}"
        mask_rel = os.path.relpath(scene_path / "mask" / f"{frame_name}_{ann_name}.png", start=args.dexycb_root)
        mask_visib_rel = os.path.relpath(scene_path / "mask_visib" / f"{frame_name}_{ann_name}.png", start=args.dexycb_root)
        rgb_rel = os.path.relpath(scene_path / "rgb" / f"{frame_name}.jpg", start=args.dexycb_root)

        result["samples"].append({
            "raw_idx": int(raw_idx),
            "seq_id": seq_id,
            "cam_id": cam_id,
            "frame_id": frame_id,
            "split_scene_id": int(scene_match["split_scene_id"]),
            "data_scene_id": int(scene_match["data_scene_id"]),
            "bop_frame_id": int(scene_match["frame_id"]),
            "ann_id": ann_id,
            "obj_id": obj_id,
            "grasp_ind": grasp_ind,
            "color_file": os.path.relpath(sample["color_file"], start=args.dexycb_root),
            "label_file": os.path.relpath(sample["label_file"], start=args.dexycb_root),
            "mask_path": mask_rel,
            "mask_visib_path": mask_visib_rel,
            "rgb_path": rgb_rel,
        })
        groups[group_key].append((raw_idx, scene_match["split_scene_id"], scene_match["data_scene_id"]))
        print(
            f"[{processed_idx}/{len(raw_indices)}] ok raw_idx={raw_idx} group={group_key} "
            f"-> split_scene={scene_match['split_scene_id']:06d} data_scene={scene_match['data_scene_id']:06d}"
        )

    for (seq_id, cam_id), items in sorted(groups.items()):
        split_scene_ids = sorted({int(item[1]) for item in items})
        data_scene_ids = sorted({int(item[2]) for item in items})
        result["groups"].append({
            "seq_id": seq_id,
            "cam_id": cam_id,
            "split_scene_ids": split_scene_ids,
            "data_scene_ids": data_scene_ids,
            "num_samples": len(items),
        })

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote mapping to {args.output}")
    print(
        f"Matched groups: {len(result['groups'])}, "
        f"samples: {len(result['samples'])}, errors: {len(result['errors'])}"
    )


if __name__ == "__main__":
    main()
