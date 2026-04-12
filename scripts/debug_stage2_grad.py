import argparse
import gc
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common"))

import lib.models  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.datasets import create_dataset
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.io_utils import load_model, load_train_param
from lib.utils.logger import logger
from lib.utils.net_utils import build_optimizer, build_scheduler, clip_gradient, setup_seed


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % (2**31 - 1)
    np.random.seed(seed)
    random.seed(seed)


def _resolve_stage_name(cfg, epoch_idx):
    if epoch_idx < cfg.TRAIN.get("STAGE1_END_EPOCH", 4):
        return "stage1"
    return "stage2"


def _collect_bad_grad_names(model, max_items=16):
    bad_names = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            bad_names.append(f"{name}:nonfinite")
            if len(bad_names) >= max_items:
                break
            continue
        grad_norm = torch.norm(grad.to(dtype=torch.float64))
        if not torch.isfinite(grad_norm):
            bad_names.append(f"{name}:norm={float(grad_norm.item())}")
            if len(bad_names) >= max_items:
                break
    return bad_names


def _grad_summary(model):
    total_norm_sq = 0.0
    num_grads = 0
    focus_names = [
        "img_backbone.conv1.weight",
        "img_backbone.layer1.0.conv1.weight",
        "img_backbone.layer1.0.conv2.weight",
        "img_backbone.layer1.1.conv1.weight",
        "img_backbone.layer1.1.conv2.weight",
    ]
    focus = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().to(dtype=torch.float64)
        if torch.isfinite(grad).all():
            total_norm_sq += float(torch.sum(grad * grad).item())
        num_grads += 1
        if name in focus_names:
            focus[name] = {
                "finite": bool(torch.isfinite(param.grad).all().item()),
                "max_abs": float(torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()),
                "norm": float(torch.norm(torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float64)).item()),
            }
    return {
        "total_grad_norm": total_norm_sq ** 0.5,
        "num_grads": num_grads,
        "bad_grad_params": _collect_bad_grad_names(model),
        "focus": focus,
    }


class _NullSummaryWriter:
    def __getattr__(self, _name):
        return self._noop

    @staticmethod
    def _noop(*args, **kwargs):
        return None
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
    stage_name = _resolve_stage_name(cfg, epoch_idx)
    if hasattr(dataset, "set_stage"):
        dataset.set_stage(stage_name)
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
    return dataset, sampler, loader


def _load_model_and_optimizer(cfg, device, resume_root, resume_epoch, stage_name):
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=_NullSummaryWriter())
    model = model.to(device)
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(stage_name)
    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)
    if resume_root is not None:
        resume_dir = os.path.join(resume_root, "checkpoints", f"checkpoint_{resume_epoch}")
        load_train_param(
            optimizer,
            scheduler,
            os.path.join(resume_dir, "train_param.pth.tar"),
            map_location=device,
        )
        load_model(model, resume_dir, map_location=device)
    model.train()
    return model, optimizer


def _run_mode(cfg, batch, device, resume_root, resume_epoch, epoch_idx, inspect_losses):
    stage_name = _resolve_stage_name(cfg, epoch_idx)
    model, optimizer = _load_model_and_optimizer(cfg, device, resume_root, resume_epoch, stage_name)
    optimizer.zero_grad(set_to_none=True)

    preds, loss_dict = model(batch, 0, "train", epoch_idx=epoch_idx)
    loss = loss_dict["loss"]

    result = {
        "stage_name": stage_name,
        "loss": float(loss.detach().float().item()),
        "loss_terms": {
            key: float(value.detach().float().mean().item()) if torch.is_tensor(value) else float(value)
            for key, value in loss_dict.items()
        },
    }

    loss.backward()

    if cfg.TRAIN.GRAD_CLIP_ENABLED:
        result["grad_clip_norm"] = clip_gradient(
            optimizer,
            cfg.TRAIN.GRAD_CLIP.NORM,
            cfg.TRAIN.GRAD_CLIP.TYPE,
        )
    else:
        result["grad_clip_norm"] = None

    result.update(_grad_summary(model))

    if inspect_losses:
        per_loss = {}
        for key, value in loss_dict.items():
            if key == "loss":
                continue
            scalar = value if torch.is_tensor(value) else None
            if scalar is None or scalar.ndim != 0 or not torch.isfinite(scalar).all():
                continue
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            _, sub_loss_dict = model(batch, 0, "train", epoch_idx=epoch_idx)
            sub_loss = sub_loss_dict[key]
            if not torch.is_tensor(sub_loss) or sub_loss.ndim != 0 or not torch.isfinite(sub_loss).all():
                continue
            try:
                sub_loss.backward()
                per_loss[key] = _grad_summary(model)
            except RuntimeError as exc:
                per_loss[key] = {"runtime_error": str(exc)}
        result["per_loss"] = per_loss

    del preds, loss_dict, loss
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--epoch_idx", type=int, default=15)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inspect_losses", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.cfg, merge=True)
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this debug script.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    _, _, loader = _build_loader(cfg, args.epoch_idx, args.workers)
    batch = None
    for idx, item in enumerate(loader):
        if idx == args.batch_idx:
            batch = _clone_batch_to_device(item, device)
            break
    if batch is None:
        raise RuntimeError(f"batch_idx={args.batch_idx} is out of range")

    logger.warning(
        f"[GradDebug] cfg={args.cfg} resume={args.resume} resume_epoch={args.resume_epoch} "
        f"epoch_idx={args.epoch_idx} batch_idx={args.batch_idx}"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result = _run_mode(
        cfg=cfg,
        batch=batch,
        device=device,
        resume_root=args.resume,
        resume_epoch=args.resume_epoch,
        epoch_idx=args.epoch_idx,
        inspect_losses=args.inspect_losses,
    )
    print("\n===== fp32 =====")
    print(f"stage={result['stage_name']} loss={result['loss']:.6f}")
    print(f"grad_clip_norm={result['grad_clip_norm']}")
    print(f"total_grad_norm={result['total_grad_norm']:.6f} num_grads={result['num_grads']}")
    print(f"bad_grad_params={result['bad_grad_params']}")
    print(f"focus={result['focus']}")
    if "per_loss" in result:
        print("per_loss_bad_grad_summary=")
        for key, value in result["per_loss"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
