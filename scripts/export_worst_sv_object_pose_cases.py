import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import lib.models  # noqa: F401
from lib.datasets import create_dataset
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.io_utils import load_model
from lib.utils.net_utils import setup_seed
from lib.utils.transform import batch_cam_intr_projection
from lib.viztools.draw import (
    draw_batch_mesh_images_pred,
    draw_batch_object_kp_confidence_images,
    draw_batch_object_kp_images,
    tile_batch_images,
)


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % (2**31 - 1)
    np.random.seed(seed)
    random.seed(seed)


def _clone_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: _clone_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_clone_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_clone_to_device(v, device) for v in data)
    return data


def _make_eval_batch(sample, device):
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
            batch[key] = torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)
        else:
            batch[key] = value
    return batch


def _to_numpy_uint8(image_hwc):
    if torch.is_tensor(image_hwc):
        image_hwc = image_hwc.detach().cpu().numpy()
    image_hwc = np.asarray(image_hwc)
    if image_hwc.dtype != np.uint8:
        image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)
    return image_hwc


def _add_header(image_hwc, lines):
    image_hwc = _to_numpy_uint8(image_hwc)
    line_height = 24
    pad = 12
    header_h = pad * 2 + max(len(lines), 1) * line_height
    canvas = np.full((header_h + image_hwc.shape[0], image_hwc.shape[1], 3), 255, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            str(line),
            (12, pad + 18 + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (32, 32, 32),
            1,
            cv2.LINE_AA,
        )
    canvas[header_h:, :, :] = image_hwc
    return canvas


def _stack_vertical(images, pad=12):
    valid_images = [_to_numpy_uint8(img) for img in images if img is not None]
    if not valid_images:
        raise ValueError("No images to stack.")
    width = max(img.shape[1] for img in valid_images)
    total_h = sum(img.shape[0] for img in valid_images) + pad * (len(valid_images) - 1)
    canvas = np.full((total_h, width, 3), 255, dtype=np.uint8)
    y = 0
    for img in valid_images:
        canvas[y:y + img.shape[0], :img.shape[1]] = img
        y += img.shape[0] + pad
    return canvas


def _sanitize_token(value):
    value = str(value)
    safe = []
    for ch in value:
        safe.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(safe).strip("_") or "na"


def _first_scalar(value, batch_idx=0, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().cpu()
        if value.dim() > 0:
            value = value[batch_idx]
        while value.dim() > 0:
            value = value[0]
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        arr = value
        if arr.ndim > 0:
            arr = arr[batch_idx]
        while arr.ndim > 0:
            arr = arr[0]
        return arr.item() if hasattr(arr, "item") else arr
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        item = value[batch_idx] if len(value) > batch_idx else value[0]
        return _first_scalar(item, batch_idx=0, default=default)
    return value


def _resolve_checkpoint_dir(exp_dir, resume_epoch):
    if os.path.basename(exp_dir).startswith("checkpoint"):
        return exp_dir
    name = "checkpoint" if resume_epoch is None else f"checkpoint_{int(resume_epoch)}"
    return os.path.join(exp_dir, "checkpoints", name)


def _build_model_and_dataset(cfg, split, checkpoint_dir, device):
    dataset_cfg = getattr(cfg.DATASET, split.upper())
    dataset = create_dataset(dataset_cfg, data_preset=cfg.DATA_PRESET)
    if hasattr(dataset, "set_stage"):
        dataset.set_stage("stage1")

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN).to(device)
    if hasattr(model, "set_train_stage"):
        model.set_train_stage("stage1")
    load_model(model, checkpoint_dir, map_location=device, strict=False)
    model.eval()
    return model, dataset


def _compute_batch_metrics(model, preds, batch):
    pred_rot = preds["obj_pose_rot6d_cam"]
    pred_trans = preds["obj_pose_trans_cam"]
    dtype = pred_rot.dtype
    device = pred_rot.device

    gt_rot, gt_trans = model._get_gt_object_pose_abs(batch, dtype=dtype, device=device)
    if gt_rot is None or gt_trans is None:
        raise RuntimeError("Ground-truth object pose is not available for this batch.")

    bsz, num_views = pred_rot.shape[:2]
    rot_deg = torch.rad2deg(model.rotation_geodesic(pred_rot.reshape(-1, 6), gt_rot.reshape(-1, 6))).view(bsz, num_views)
    trans_epe = torch.linalg.norm(pred_trans - gt_trans, dim=-1)

    gt_obj_points = model._build_gt_object_points(batch, dtype=dtype, device=device)
    pred_obj_points = model._build_pred_object_points_for_eval(preds, batch)
    add = torch.linalg.norm(pred_obj_points - gt_obj_points, dim=-1).mean(dim=-1)
    adds = torch.cdist(pred_obj_points.float().flatten(0, 1), gt_obj_points.float().flatten(0, 1))
    adds = adds.min(dim=-1)[0].mean(dim=-1).view(bsz, num_views)

    pnp_valid = preds["obj_pose_valid"]
    return {
        "rot_deg_view": rot_deg,
        "trans_mm_view": trans_epe * 1000.0,
        "add_mm_view": add * 1000.0,
        "adds_mm_view": adds * 1000.0,
        "pnp_valid_view": pnp_valid,
        "rot_deg_mean": rot_deg.mean(dim=-1),
        "trans_mm_mean": (trans_epe * 1000.0).mean(dim=-1),
        "add_mm_mean": (add * 1000.0).mean(dim=-1),
        "adds_mm_mean": (adds * 1000.0).mean(dim=-1),
        "pnp_valid_mean": pnp_valid.mean(dim=-1),
    }


def _collect_multiview_info(dataset, dataset_idx):
    meta = {
        "dataset_idx": dataset_idx,
        "identifier": getattr(dataset, "get_sample_identifier", lambda x: f"sample_{x}")(dataset_idx),
        "seq_name": "na",
        "frame_id": -1,
        "cam_serials": "",
    }
    if hasattr(dataset, "valid_sample_idx_list") and hasattr(dataset, "multiview_sample_infos"):
        valid_idx = dataset.valid_sample_idx_list[dataset_idx]
        info_list = dataset.multiview_sample_infos[valid_idx]
        if len(info_list) > 0:
            meta["seq_name"] = info_list[0].get("seq_name", "na")
            meta["frame_id"] = info_list[0].get("frame_id", -1)
            meta["cam_serials"] = ",".join(str(info.get("cam_serial", "na")) for info in info_list)
    return meta


def _scan_split(model, dataset, split, device, batch_size, workers, max_samples=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_init_fn,
        persistent_workers=bool(workers),
    )

    records = []
    seen = 0
    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(loader):
            batch = _clone_to_device(batch_cpu, device)
            preds = model._forward_impl(batch)
            metrics = _compute_batch_metrics(model, preds, batch)

            curr_bsz = int(metrics["add_mm_mean"].shape[0])
            if max_samples is not None:
                curr_bsz = min(curr_bsz, max_samples - seen)
            for i in range(curr_bsz):
                dataset_idx = seen + i
                meta = _collect_multiview_info(dataset, dataset_idx)
                meta.update({
                    "split": split,
                    "obj_id": _first_scalar(batch_cpu.get("obj_id", None), i, default=-1),
                    "score_add_mm": float(metrics["add_mm_mean"][i].detach().cpu().item()),
                    "score_adds_mm": float(metrics["adds_mm_mean"][i].detach().cpu().item()),
                    "score_rot_deg": float(metrics["rot_deg_mean"][i].detach().cpu().item()),
                    "score_trans_mm": float(metrics["trans_mm_mean"][i].detach().cpu().item()),
                    "score_pnp_valid": float(metrics["pnp_valid_mean"][i].detach().cpu().item()),
                })
                records.append(meta)
            seen += curr_bsz
            if (batch_idx + 1) % 50 == 0 or (max_samples is not None and seen >= max_samples):
                print(f"[{split}] scanned {seen}/{len(dataset) if max_samples is None else max_samples} samples", flush=True)
            if max_samples is not None and seen >= max_samples:
                break
    return records


def _render_case(model, dataset, record, out_path):
    sample = dataset[record["dataset_idx"]]
    device = next(model.parameters()).device
    batch = _make_eval_batch(sample, device)

    with torch.no_grad():
        preds = model._forward_impl(batch)
        metrics = _compute_batch_metrics(model, preds, batch)

    img = batch["image"]
    batch_size, num_views = img.shape[:2]
    assert batch_size == 1, "Only single-sample rendering is supported."
    img_h, img_w = img.shape[-2:]

    pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"].view(
        batch_size,
        num_views,
        model.num_hand_verts + model.num_hand_joints,
        2,
    )
    pred_sv_verts_2d = model._normed_uv_to_pixel(pred_mano_2d_mesh_sv[:, :, :model.num_hand_verts], (img_h, img_w))
    gt_verts_2d = model._normed_uv_to_pixel(batch["target_verts_uvd"][..., :2], (img_h, img_w))

    pred_obj_points = model._build_pred_object_points_for_eval(preds, batch)
    gt_obj_points = model._build_gt_object_points(batch, dtype=pred_obj_points.dtype, device=pred_obj_points.device)
    pred_obj_2d = batch_cam_intr_projection(batch["target_cam_intr"], pred_obj_points)
    gt_obj_2d = batch_cam_intr_projection(batch["target_cam_intr"], gt_obj_points)

    pred_obj_kp2d = preds["pred_obj_kp2d_pixel"]
    gt_obj_kp2d = batch_cam_intr_projection(batch["target_cam_intr"], batch["target_obj_kp21"])
    pred_obj_kp_conf = preds["pred_obj_kp_conf"].squeeze(-1)

    img_views = img[0]
    hand_obj_tile = tile_batch_images(
        draw_batch_mesh_images_pred(
            gt_verts2d=gt_verts_2d[0],
            pred_verts2d=pred_sv_verts_2d[0],
            face=model.face,
            gt_obj2d=gt_obj_2d[0],
            pred_obj2d=pred_obj_2d[0],
            gt_objc2d=gt_obj_kp2d[0, :, -1:],
            pred_objc2d=pred_obj_kp2d[0, :, -1:],
            intr=batch["target_cam_intr"][0],
            tensor_image=img_views,
            pred_obj_rot_error=metrics["rot_deg_view"][0].unsqueeze(-1),
            pred_obj_trans_error=metrics["trans_mm_view"][0].unsqueeze(-1),
            n_sample=num_views,
        )
    )
    kp_pred_tile = tile_batch_images(
        draw_batch_object_kp_images(pred_obj_kp2d[0], img_views, n_sample=num_views, title="Pred Object KP21")
    )
    kp_gt_tile = tile_batch_images(
        draw_batch_object_kp_images(gt_obj_kp2d[0], img_views, n_sample=num_views, title="GT Object KP21")
    )
    kp_conf_tile = tile_batch_images(
        draw_batch_object_kp_confidence_images(
            pred_obj_kp2d[0],
            pred_obj_kp_conf[0],
            img_views,
            n_sample=num_views,
            title="Pred Object KP21 Confidence",
        )
    )

    header_lines = [
        (
            f"{record['split']} rank#{record['rank']:02d} idx={record['dataset_idx']} "
            f"seq={record['seq_name']} frame={record['frame_id']} obj={record['obj_id']}"
        ),
        (
            f"mean ADD={record['score_add_mm']:.2f} mm | ADDS={record['score_adds_mm']:.2f} mm | "
            f"rot={record['score_rot_deg']:.2f} deg | tr={record['score_trans_mm']:.2f} mm | "
            f"PnP valid={record['score_pnp_valid']:.2f}"
        ),
        f"cams={record['cam_serials']}",
    ]
    stacked = _stack_vertical([hand_obj_tile, kp_pred_tile, kp_gt_tile, kp_conf_tile], pad=10)
    stacked = _add_header(stacked, header_lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))


def _write_csv(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not records:
        return
    fieldnames = [
        "rank",
        "split",
        "dataset_idx",
        "identifier",
        "seq_name",
        "frame_id",
        "obj_id",
        "score_add_mm",
        "score_adds_mm",
        "score_rot_deg",
        "score_trans_mm",
        "score_pnp_valid",
        "cam_serials",
        "image_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Export worst single-view object pose cases for HOR_sv_tri.")
    parser.add_argument("--cfg", default="exp/default_2026_0417_0044_24/dump_cfg.yaml")
    parser.add_argument("--exp-dir", default="exp/default_2026_0417_0044_24")
    parser.add_argument("--resume-epoch", type=int, default=60)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="tmp_debug/worst_sv_object_pose_cases")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.cfg, merge=True)
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint_dir = _resolve_checkpoint_dir(args.exp_dir, args.resume_epoch)
    all_top_records = []

    for split in args.splits:
        model, dataset = _build_model_and_dataset(cfg, split, checkpoint_dir, device)
        records = _scan_split(
            model,
            dataset,
            split,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
            max_samples=args.max_samples,
        )
        records.sort(key=lambda x: x["score_add_mm"], reverse=True)

        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
            record["image_path"] = ""

        split_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        _write_csv(os.path.join(split_dir, "all_metrics.csv"), records)

        top_records = records[: args.topk]
        for record in top_records:
            filename = (
                f"rank_{record['rank']:02d}"
                f"__seq_{_sanitize_token(record['seq_name'])}"
                f"__frame_{_sanitize_token(record['frame_id'])}"
                f"__obj_{_sanitize_token(record['obj_id'])}"
                f"__add_{record['score_add_mm']:.1f}mm.png"
            )
            out_path = os.path.join(split_dir, filename)
            _render_case(model, dataset, record, out_path)
            record["image_path"] = out_path

        _write_csv(os.path.join(split_dir, "topk_metrics.csv"), top_records)
        all_top_records.extend(top_records)

    if all_top_records:
        _write_csv(os.path.join(args.out_dir, "topk_summary.csv"), all_top_records)


if __name__ == "__main__":
    main()
