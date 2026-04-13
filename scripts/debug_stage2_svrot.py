import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common"))

import lib.models  # noqa: F401
from lib.datasets import create_dataset
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.logger import logger
from lib.utils.net_utils import build_optimizer, clip_gradient, setup_seed


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % (2**31 - 1)
    np.random.seed(seed)
    random.seed(seed)


def _clone_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _clone_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_clone_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_clone_batch_to_device(v, device) for v in batch)
    return batch


def _build_loader(cfg, epoch_idx, workers):
    dataset = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    if hasattr(dataset, "set_stage"):
        dataset.set_stage("stage2")
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(epoch_idx)
    loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        worker_init_fn=_init_fn,
        persistent_workers=False,
    )
    return loader


def _compute_svrot_breakdown(model, preds, batch):
    pred_obj_view_rot6d_cam = preds.get("obj_view_rot6d_cam", None)
    target_obj_rot6d_gt = batch.get("target_rot6d_label", None)
    master_id = batch.get("master_id", None)
    if pred_obj_view_rot6d_cam is None or target_obj_rot6d_gt is None or master_id is None:
        zero = torch.tensor(0.0, device=preds["mano_3d_mesh_sv"].device)
        return {
            "sv_master_deg": zero,
            "sv_nonmaster_deg": zero,
            "sv_all_deg": zero,
            "init_rot_deg": zero,
            "master_valid_rate": zero,
            "master_center_vis_rate": zero,
        }

    pred_obj_view_rot6d_cam = torch.nan_to_num(pred_obj_view_rot6d_cam.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
    target_obj_rot6d_gt = torch.nan_to_num(target_obj_rot6d_gt.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
    batch_size, n_views = pred_obj_view_rot6d_cam.shape[:2]

    rot_deg = torch.rad2deg(
        model.rotation_geodesic(
            pred_obj_view_rot6d_cam.reshape(-1, 6),
            target_obj_rot6d_gt.reshape(-1, 6),
        )
    ).view(batch_size, n_views)
    valid_weight = torch.ones_like(rot_deg)
    pred_obj_view_rot_valid = preds.get("obj_view_rot_valid", None)
    if pred_obj_view_rot_valid is not None:
        valid_weight = valid_weight * pred_obj_view_rot_valid.detach().to(dtype=rot_deg.dtype, device=rot_deg.device)
    obj_center_vis = batch.get("target_obj_center_vis", None)
    if obj_center_vis is not None:
        valid_weight = valid_weight * obj_center_vis.view(batch_size, n_views).to(dtype=rot_deg.dtype, device=rot_deg.device)
        center_vis_mv = obj_center_vis.view(batch_size, n_views).to(dtype=rot_deg.dtype, device=rot_deg.device)
    else:
        center_vis_mv = torch.ones_like(valid_weight)

    master_mask = torch.zeros_like(valid_weight)
    batch_idx = torch.arange(batch_size, device=rot_deg.device)
    master_mask[batch_idx, master_id.long()] = 1.0
    nonmaster_mask = valid_weight * (1.0 - master_mask)
    master_mask = valid_weight * master_mask

    gt_init_rot6d = batch.get("master_obj_rot6d_label", None)
    pred_init_rot6d = preds.get("obj_init_rot6d", None)
    if gt_init_rot6d is not None and pred_init_rot6d is not None:
        init_rot_deg = torch.rad2deg(
            model.rotation_geodesic(
                torch.nan_to_num(pred_init_rot6d.detach(), nan=0.0, posinf=10.0, neginf=-10.0),
                torch.nan_to_num(gt_init_rot6d.detach(), nan=0.0, posinf=10.0, neginf=-10.0),
            )
        ).mean()
    else:
        init_rot_deg = rot_deg.new_tensor(0.0)

    return {
        "sv_master_deg": (rot_deg * master_mask).sum() / (master_mask.sum() + 1e-6),
        "sv_nonmaster_deg": (rot_deg * nonmaster_mask).sum() / (nonmaster_mask.sum() + 1e-6),
        "sv_all_deg": (rot_deg * valid_weight).sum() / (valid_weight.sum() + 1e-6),
        "init_rot_deg": init_rot_deg,
        "master_valid_rate": valid_weight[batch_idx, master_id.long()].mean(),
        "master_center_vis_rate": center_vis_mv[batch_idx, master_id.long()].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epoch_idx", type=int, default=0)
    parser.add_argument(
        "--loss-mode",
        choices=["total", "view", "sv_stage2"],
        default="total",
        help="Optimization target: total loss, only SV view loss, or stage2 SV loss (view + init).",
    )
    args = parser.parse_args()

    cfg = get_config(args.cfg, merge=True)
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this debug script.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    loader = _build_loader(cfg, args.epoch_idx, args.workers)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN).to(device)
    if hasattr(model, "set_train_stage"):
        model.set_train_stage("stage2")
    model.train()
    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)

    logger.warning(
        f"[SVRotDebug] cfg={args.cfg} epoch_idx={args.epoch_idx} steps={args.steps} "
        f"batch_size={cfg.TRAIN.BATCH_SIZE} loss_mode={args.loss_mode}"
    )

    data_iter = iter(loader)
    for step_idx in range(args.steps):
        try:
            batch_cpu = next(data_iter)
        except StopIteration:
            break
        batch = _clone_batch_to_device(batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)

        model.current_stage_name = "stage2"
        model.current_stage2_warmup = model._stage2_object_warmup(args.epoch_idx, "stage2")
        preds = model._forward_impl(batch, interaction_mode="ho")
        total_loss, loss_dict = model.compute_loss(preds, batch, stage_name="stage2", epoch_idx=args.epoch_idx)
        if args.loss_mode == "view":
            loss = loss_dict["loss_obj_view_rot"]
        elif args.loss_mode == "sv_stage2":
            loss = loss_dict["loss_obj_view_rot"] + loss_dict["loss_obj_init_rot"]
        else:
            loss = total_loss
        sv_metric = model.compute_sv_object_pose_metrics(preds, batch)
        obj_metric = model.compute_object_pose_metrics(preds, batch)
        sv_breakdown = _compute_svrot_breakdown(model, preds, batch)

        loss.backward()
        grad_clip = None
        if cfg.TRAIN.GRAD_CLIP_ENABLED:
            grad_clip = clip_gradient(
                optimizer,
                cfg.TRAIN.GRAD_CLIP.NORM,
                cfg.TRAIN.GRAD_CLIP.TYPE,
            )
        optimizer.step()

        kpj_mm = 0.0
        if preds.get("mano_3d_mesh_kp_master", None) is not None:
            pred_kp = preds["mano_3d_mesh_kp_master"][:, model.num_hand_verts:]
            gt_kp = batch["master_joints_3d"]
            kpj_mm = float(torch.linalg.norm(pred_kp - gt_kp, dim=-1).mean().item() * 1000.0)

        print(
            f"step={step_idx:03d} "
            f"loss={float(loss.detach().item()):.4f} "
            f"total_loss={float(total_loss.detach().item()):.4f} "
            f"sv_rot_deg={float(sv_metric['metric_sv_obj_rot_deg'].detach().item()):.4f} "
            f"sv_master_deg={float(sv_breakdown['sv_master_deg'].detach().item()):.4f} "
            f"sv_nonmaster_deg={float(sv_breakdown['sv_nonmaster_deg'].detach().item()):.4f} "
            f"init_rot_deg={float(sv_breakdown['init_rot_deg'].detach().item()):.4f} "
            f"master_valid_rate={float(sv_breakdown['master_valid_rate'].detach().item()):.4f} "
            f"master_center_vis_rate={float(sv_breakdown['master_center_vis_rate'].detach().item()):.4f} "
            f"sv_rot_l1={float(sv_metric['metric_sv_obj_rot_l1'].detach().item()):.4f} "
            f"master_rot_deg={float(obj_metric['metric_obj_rot_deg'].detach().item()):.4f} "
            f"master_trans_mm={float(obj_metric['metric_obj_trans_epe'].detach().item() * 1000.0):.4f} "
            f"loss_obj_view_rot={float(loss_dict['loss_obj_view_rot'].detach().item()):.4f} "
            f"loss_obj_init_rot={float(loss_dict['loss_obj_init_rot'].detach().item()):.4f} "
            f"loss_obj_pose={float(loss_dict['loss_obj_pose'].detach().item()):.4f} "
            f"loss_triang={float(loss_dict['loss_triang'].detach().item()):.4f} "
            f"kpj_mm={kpj_mm:.4f} "
            f"grad_clip={(float(grad_clip) if grad_clip is not None else -1.0):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
