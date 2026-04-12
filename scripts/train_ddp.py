import os
import random
import socket
import sys
import logging
import traceback
import warnings
from argparse import Namespace
from time import time
from pathlib import Path

# Silence TensorFlow/TensorBoard backend logs before any related import happens.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

# Add external module paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common"))

import lib.models
import numpy as np
import torch
from lib.datasets import create_dataset
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_perfixes, format_args_cfg
from lib.utils.net_utils import build_optimizer, build_scheduler, clip_gradient, setup_seed
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.utils.config import CN


def _find_available_port(default_port: str, host: str) -> str:
    requested_port = int(default_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, requested_port))
            return str(requested_port)
        except OSError:
            sock.bind((host, 0))
            return str(sock.getsockname()[1])


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def _resolve_stage_name(cfg: CN, epoch_idx: int) -> str:
    if epoch_idx < cfg.TRAIN.get("STAGE1_END_EPOCH", 4):
        return "stage1"
    return "stage2"


def _debug_logs_enabled(cfg: CN) -> bool:
    return bool(cfg.TRAIN.get("DEBUG_LOGS", False))


def _build_ddp_model(model: torch.nn.Module, rank: int, cfg: CN, static_graph: bool = False, distributed: bool = True):
    if not distributed:
        return model
    # We rebuild the DDP wrapper when switching stages, so each stage can use a static graph.
    # Keep bucket views disabled too, because they already emitted stride-mismatch warnings.
    return DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=bool(cfg.TRAIN.get("FIND_UNUSED_PARAMETERS", False)),
        static_graph=static_graph,
        gradient_as_bucket_view=False,
    )


def _move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(v, device) for v in data)
    return data


def _log_cuda_memory(tag: str, rank: int):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(rank) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(rank) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(rank) / (1024 ** 3)
    logger.warning(
        f"[CudaMem] rank={rank} {tag} allocated={allocated:.2f}G reserved={reserved:.2f}G max_allocated={max_allocated:.2f}G"
    )


def _dist_barrier(arg: Namespace, rank: int):
    if not arg.distributed:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[rank])
    else:
        torch.distributed.barrier()


def _format_loss_dict_for_debug(loss_dict):
    items = []
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value).all().item())
            scalar = float(value.detach().float().mean().item())
        else:
            scalar = float(value)
            finite = np.isfinite(scalar)
        items.append(f"{key}={scalar:.6g}({'ok' if finite else 'bad'})")
    return ", ".join(items)


def _format_debug_value(value, max_items: int = 8):
    if torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1)
        values = flat[:max_items].tolist()
        suffix = "..." if flat.numel() > max_items else ""
        return f"{values}{suffix}"
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        values = flat[:max_items].tolist()
        suffix = "..." if flat.size > max_items else ""
        return f"{values}{suffix}"
    if isinstance(value, (list, tuple)):
        values = list(value[:max_items])
        suffix = "..." if len(value) > max_items else ""
        return f"{values}{suffix}"
    return str(value)


def _format_batch_dict_for_debug(batch):
    debug_keys = ["sample_idx", "master_id", "master_serial", "cam_serial"]
    parts = []
    for key in debug_keys:
        if key in batch:
            parts.append(f"{key}={_format_debug_value(batch[key])}")
    return ", ".join(parts) if parts else "no_batch_id_fields"


def _emit_rank_error(message: str):
    print(message, file=sys.stderr, flush=True)


def _distributed_loss_is_finite(loss, arg: Namespace, rank: int) -> bool:
    local_is_finite = bool(torch.isfinite(loss).all().item())
    if not arg.distributed:
        return local_is_finite

    flag = torch.tensor(
        [1 if local_is_finite else 0],
        device=loss.device if torch.is_tensor(loss) else (torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")),
        dtype=torch.int64,
    )
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _gather_bad_rank_flags(local_bad: bool, arg: Namespace, device: torch.device):
    if not arg.distributed:
        return [1 if local_bad else 0]
    local_flag = torch.tensor([1 if local_bad else 0], device=device, dtype=torch.int64)
    gathered = [torch.zeros_like(local_flag) for _ in range(arg.world_size)]
    torch.distributed.all_gather(gathered, local_flag)
    return [int(flag.item()) for flag in gathered]


def _stage_static_graph_enabled(cfg: CN, arg: Namespace) -> bool:
    if not arg.distributed:
        return False
    return bool(cfg.TRAIN.get("DDP_STATIC_GRAPH_BY_STAGE", True))


def _collect_bad_grad_names(model: torch.nn.Module, max_items: int = 8):
    bad_names = []
    named_params = model.named_parameters()
    for name, param in named_params:
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
            bad_names.append(
                f"{name}:norm={float(grad_norm.item())},max_abs={float(grad.abs().max().item()):.4e}"
            )
            if len(bad_names) >= max_items:
                break
    return bad_names


def main_worker(gpu_id: int, cfg: CN, arg: Namespace, time_f: float):
    rank = arg.n_gpus * arg.node_rank + gpu_id if arg.distributed else 0
    summary = None
    debug_logs = _debug_logs_enabled(cfg)
    stage_static_graph = _stage_static_graph_enabled(cfg, arg)
    try:
        if not debug_logs:
            warnings.filterwarnings(
                "ignore",
                message="Grad strides do not match bucket view strides.*",
                category=UserWarning,
            )
        # if the model is from the external package
        if cfg.MODEL.TYPE in EXT_PACKAGE:
            pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
            exec(f"from lib.external import {pkg}")

        if arg.distributed:
            torch.distributed.init_process_group(arg.dist_backend, rank=rank, world_size=arg.world_size)
            assert rank == torch.distributed.get_rank(), "Something wrong with nodes or gpus"
            torch.cuda.set_device(rank)

        setup_seed(cfg.TRAIN.MANUAL_SEED + rank, cfg.TRAIN.CONV_REPEATABLE)
        recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
        summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)

        _dist_barrier(arg, rank)

        train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
        train_sampler = (
            DistributedSampler(train_data, num_replicas=arg.world_size, rank=rank, shuffle=True)
            if arg.distributed else None
        )
        train_loader = DataLoader(train_data,
                                  batch_size=arg.batch_size,
                                  shuffle=(train_sampler is None),
                                  num_workers=int(arg.workers),
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=train_sampler,
                                  worker_init_fn=_init_fn,
                                  persistent_workers=False)

        if rank == 0:
            val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
            val_loader = DataLoader(val_data,
                                    batch_size=arg.val_batch_size,
                                    shuffle=True,
                                    num_workers=int(arg.workers),
                                    pin_memory=True,
                                    drop_last=False,
                                    worker_init_fn=_init_fn,
                                    persistent_workers=False)
        else:
            val_loader = None

        model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
        model.setup(summary_writer=summary)
        model = model.to(rank)
        base_model = model
        current_stage_name = _resolve_stage_name(cfg, 0)
        if hasattr(base_model, "set_train_stage"):
            base_model.set_train_stage(current_stage_name)
        model = _build_ddp_model(base_model, rank, cfg, static_graph=stage_static_graph, distributed=arg.distributed)

        optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
        scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)

        if arg.resume:
            epoch = recorder.resume_checkpoints(
                model,
                optimizer,
                scheduler,
                arg.resume,
                resume_epoch=arg.resume_epoch,
            )
        else:
            epoch = 0

        _dist_barrier(arg, rank)

        logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
        for epoch_idx in range(epoch, cfg["TRAIN"]["EPOCH"]):
            stage_name = _resolve_stage_name(cfg, epoch_idx)
            if debug_logs:
                logger.warning(f"[StageSwitch] epoch={epoch_idx} resolved_stage={stage_name}")
            if stage_name != current_stage_name:
                if hasattr(base_model, "set_train_stage"):
                    base_model.set_train_stage(stage_name)
                current_stage_name = stage_name
                if stage_static_graph:
                    _dist_barrier(arg, rank)
                    del model
                    model = _build_ddp_model(base_model, rank, cfg, static_graph=True, distributed=arg.distributed)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(rank)
                    if debug_logs:
                        _log_cuda_memory(f"after_stage_switch_{stage_name}", rank)
            if hasattr(train_data, "set_stage"):
                train_data.set_stage(stage_name)
                if debug_logs:
                    logger.warning(f"[StageSwitch] train dataset set_stage({stage_name})")
            if rank == 0 and val_loader is not None and hasattr(val_data, "set_stage"):
                val_data.set_stage(stage_name)
                if debug_logs:
                    logger.warning(f"[StageSwitch] val dataset set_stage({stage_name})")

            if arg.distributed:
                train_sampler.set_epoch(epoch_idx)

            model.train()
            trainbar = etqdm(train_loader, rank=rank)
            for bidx, batch in enumerate(trainbar):
                batch = _move_to_device(batch, rank)
                optimizer.zero_grad(set_to_none=True)
                step_idx = epoch_idx * len(train_loader) + bidx
                log_step_boundary = debug_logs and bidx < 2
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} forward_start")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} pre_forward", rank)
                preds, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
                loss = loss_dict["loss"]
                local_loss_finite = bool(torch.isfinite(loss).all().item())
                global_loss_finite = _distributed_loss_is_finite(loss, arg, rank)
                if not global_loss_finite:
                    batch_debug = _format_batch_dict_for_debug(batch)
                    bad_rank_flags = _gather_bad_rank_flags(
                        local_bad=not local_loss_finite,
                        arg=arg,
                        device=loss.device if torch.is_tensor(loss) else torch.device(f"cuda:{rank}"),
                    )
                    if not local_loss_finite:
                        _emit_rank_error(
                            f"[NonFiniteLoss] rank={rank} epoch={epoch_idx} stage={stage_name} "
                            f"bidx={bidx} step={step_idx}; batch: {batch_debug}; "
                            f"loss terms: {_format_loss_dict_for_debug(loss_dict)}"
                        )
                    elif rank == 0:
                        logger.error(
                            f"[NonFiniteLoss] bad_ranks={[i for i, flag in enumerate(bad_rank_flags) if flag]} "
                            f"epoch={epoch_idx} stage={stage_name} bidx={bidx} step={step_idx}; "
                            f"rank0_batch: {batch_debug}"
                        )
                    raise RuntimeError(
                        f"Non-finite loss detected on at least one rank: "
                        f"epoch={epoch_idx} stage={stage_name} bidx={bidx} step={step_idx}"
                    )
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} forward_done loss={loss.item():.4f}")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} post_forward", rank)
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} backward_start")
                loss.backward()
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} backward_done")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} post_backward", rank)
                grad_norm = None
                if cfg.TRAIN.GRAD_CLIP_ENABLED:
                    grad_norm = clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

                local_bad_grad = False
                local_bad_grad_names = []
                if grad_norm is not None and not np.isfinite(grad_norm):
                    local_bad_grad_names = _collect_bad_grad_names(model)
                    local_bad_grad = True
                else:
                    local_bad_grad_names = _collect_bad_grad_names(model)
                    local_bad_grad = len(local_bad_grad_names) > 0

                bad_grad_flags = _gather_bad_rank_flags(
                    local_bad=local_bad_grad,
                    arg=arg,
                    device=next(model.parameters()).device,
                )
                if any(bad_grad_flags):
                    if local_bad_grad:
                        _emit_rank_error(
                            f"[NonFiniteGrad] rank={rank} epoch={epoch_idx} stage={stage_name} "
                            f"bidx={bidx} step={step_idx} grad_norm={grad_norm}; "
                            f"bad_grad_params={local_bad_grad_names}"
                        )
                    elif rank == 0:
                        logger.error(
                            f"[NonFiniteGrad] bad_ranks={[i for i, flag in enumerate(bad_grad_flags) if flag]} "
                            f"epoch={epoch_idx} stage={stage_name} bidx={bidx} step={step_idx} grad_norm={grad_norm}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} step_done")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} post_step", rank)
                del preds, loss_dict, loss, batch
                optimizer.zero_grad(set_to_none=True)

                metric_model = model.module if hasattr(model, "module") else model
                trainbar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} "
                                         f"{metric_model.format_metric('train')}")

            scheduler.step()
            logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

            recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
            _dist_barrier(arg, rank)
            base_model.on_train_finished(recorder, epoch_idx)

            if (epoch_idx % arg.eval_interval == arg.eval_interval - 1 or epoch_idx == 0) and rank == 0:
                logger.info("do validation and save results")
                with torch.no_grad():
                    base_model.eval()
                    valbar = etqdm(val_loader, rank=rank)
                    for bidx, batch in enumerate(valbar):
                        step_idx = epoch_idx * len(val_loader) + bidx
                        batch = _move_to_device(batch, rank)
                        _ = base_model(batch, step_idx, "val", epoch_idx=epoch_idx)

                        valbar.set_description(f"{bar_perfixes['val']} Epoch {epoch_idx} "
                                               f"{base_model.format_metric('val')}")
                        del batch

                base_model.on_val_finished(recorder, epoch_idx)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if debug_logs:
                        _log_cuda_memory(f"after_val_epoch={epoch_idx}", rank)
            if arg.distributed:
                if debug_logs:
                    logger.warning(f"[EpochSync] rank={rank} epoch={epoch_idx} wait_after_val")
                _dist_barrier(arg, rank)
                if debug_logs:
                    logger.warning(f"[EpochSync] rank={rank} epoch={epoch_idx} done_after_val")

    except Exception:
        logger.error(f"[WorkerException] rank={rank} gpu_id={gpu_id}\n{traceback.format_exc()}")
        raise
    finally:
        if summary is not None and hasattr(summary, "close"):
            summary.close()
        if arg.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                _dist_barrier(arg, rank)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(rank)
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=False)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL" if _debug_logs_enabled(cfg) else "OFF"

    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    arg.dist_master_port = _find_available_port(arg.dist_master_port, arg.dist_master_addr)
    os.environ["MASTER_PORT"] = arg.dist_master_port
    # must have equal gpus on each node.
    arg.world_size = arg.n_gpus * arg.nodes
    if arg.world_size <= 1:
        arg.distributed = False
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size

    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    if arg.distributed:
        logger.info("====> Use Distributed Data Parallel <====")
        torch.multiprocessing.spawn(main_worker, args=(cfg, arg, exp_time), nprocs=arg.n_gpus)
    else:
        logger.info("====> Use Single-Process Training <====")
        main_worker(0, cfg, arg, exp_time)
