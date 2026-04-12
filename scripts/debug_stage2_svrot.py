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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epoch_idx", type=int, default=0)
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
        f"[SVRotDebug] cfg={args.cfg} epoch_idx={args.epoch_idx} steps={args.steps} batch_size={cfg.TRAIN.BATCH_SIZE}"
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
        loss, loss_dict = model.compute_loss(preds, batch, stage_name="stage2", epoch_idx=args.epoch_idx)
        sv_metric = model.compute_sv_object_pose_metrics(preds, batch)
        obj_metric = model.compute_object_pose_metrics(preds, batch)

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
            f"sv_rot_deg={float(sv_metric['metric_sv_obj_rot_deg'].detach().item()):.4f} "
            f"sv_rot_l1={float(sv_metric['metric_sv_obj_rot_l1'].detach().item()):.4f} "
            f"master_rot_deg={float(obj_metric['metric_obj_rot_deg'].detach().item()):.4f} "
            f"master_trans_mm={float(obj_metric['metric_obj_trans_epe'].detach().item() * 1000.0):.4f} "
            f"loss_obj_view_rot={float(loss_dict['loss_obj_view_rot'].detach().item()):.4f} "
            f"loss_obj_pose={float(loss_dict['loss_obj_pose'].detach().item()):.4f} "
            f"loss_triang={float(loss_dict['loss_triang'].detach().item()):.4f} "
            f"kpj_mm={kpj_mm:.4f} "
            f"grad_clip={(float(grad_clip) if grad_clip is not None else -1.0):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
