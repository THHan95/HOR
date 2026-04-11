import os
import random
import socket
import sys
import logging
import traceback
from argparse import Namespace
from time import time
from pathlib import Path

# Silence TensorFlow/TensorBoard backend logs before any related import happens.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

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
from torch.cuda.amp import GradScaler, autocast
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
    if epoch_idx < cfg.TRAIN.get("STAGE2_END_EPOCH", 11):
        return "stage2"
    return "stage3"


def _build_ddp_model(model: torch.nn.Module, rank: int, cfg: CN) -> DDP:
    # This training pipeline changes active branches across stages, so the graph is not static.
    # Keep bucket views disabled too, because they already emitted stride-mismatch warnings.
    return DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        static_graph=False,
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


def main_worker(gpu_id: int, cfg: CN, arg: Namespace, time_f: float):
    rank = arg.n_gpus * arg.node_rank + gpu_id if arg.distributed else 0
    try:
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

        torch.distributed.barrier()

        train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
        train_sampler = DistributedSampler(train_data, num_replicas=arg.world_size, rank=rank, shuffle=True)
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
        model = _build_ddp_model(base_model, rank, cfg)

        optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
        scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)
        amp_enabled = bool(cfg.TRAIN.get("AMP", True)) and torch.cuda.is_available()
        scaler = GradScaler(enabled=amp_enabled)

        if arg.resume:
            epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume)
        else:
            epoch = 0

        torch.distributed.barrier()

        logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
        for epoch_idx in range(epoch, cfg["TRAIN"]["EPOCH"]):
            stage_name = _resolve_stage_name(cfg, epoch_idx)
            logger.warning(f"[StageSwitch] epoch={epoch_idx} resolved_stage={stage_name}")
            if stage_name != current_stage_name:
                if hasattr(base_model, "set_train_stage"):
                    base_model.set_train_stage(stage_name)
                current_stage_name = stage_name
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(rank)
                    _log_cuda_memory(f"after_stage_switch_{stage_name}", rank)
            if hasattr(train_data, "set_stage"):
                train_data.set_stage(stage_name)
                logger.warning(f"[StageSwitch] train dataset set_stage({stage_name})")
            if rank == 0 and val_loader is not None and hasattr(val_data, "set_stage"):
                val_data.set_stage(stage_name)
                logger.warning(f"[StageSwitch] val dataset set_stage({stage_name})")

            if arg.distributed:
                train_sampler.set_epoch(epoch_idx)

            model.train()
            trainbar = etqdm(train_loader, rank=rank)
            for bidx, batch in enumerate(trainbar):
                optimizer.zero_grad(set_to_none=True)
                step_idx = epoch_idx * len(train_loader) + bidx
                log_step_boundary = bidx < 2
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} forward_start")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} pre_forward", rank)
                with autocast(enabled=amp_enabled):
                    preds, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
                    loss = loss_dict["loss"]
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} forward_done loss={loss.item():.4f}")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} post_forward", rank)
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} backward_start")
                scaler.scale(loss).backward()
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} backward_done")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} post_backward", rank)
                if cfg.TRAIN.GRAD_CLIP_ENABLED:
                    scaler.unscale_(optimizer)
                    clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

                scaler.step(optimizer)
                scaler.update()
                if log_step_boundary:
                    logger.warning(f"[BatchTrace] rank={rank} epoch={epoch_idx} stage={stage_name} bidx={bidx} step_done")
                    _log_cuda_memory(f"epoch={epoch_idx} stage={stage_name} bidx={bidx} post_step", rank)
                del preds, loss_dict, loss, batch
                optimizer.zero_grad(set_to_none=True)

                trainbar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} "
                                         f"{model.module.format_metric('train')}")

            scheduler.step()
            logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

            recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
            torch.distributed.barrier()
            base_model.on_train_finished(recorder, epoch_idx)

            if (epoch_idx % arg.eval_interval == arg.eval_interval - 1 or epoch_idx == 0) and rank == 0:
                logger.info("do validation and save results")
                with torch.no_grad():
                    base_model.eval()
                    valbar = etqdm(val_loader, rank=rank)
                    for bidx, batch in enumerate(valbar):
                        step_idx = epoch_idx * len(val_loader) + bidx
                        batch = _move_to_device(batch, rank)
                        with autocast(enabled=amp_enabled):
                            _ = base_model(batch, step_idx, "val", epoch_idx=epoch_idx)

                        valbar.set_description(f"{bar_perfixes['val']} Epoch {epoch_idx} "
                                               f"{base_model.format_metric('val')}")
                        del batch

                base_model.on_val_finished(recorder, epoch_idx)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    _log_cuda_memory(f"after_val_epoch={epoch_idx}", rank)
            if arg.distributed:
                logger.warning(f"[EpochSync] rank={rank} epoch={epoch_idx} wait_after_val")
                torch.distributed.barrier()
                logger.warning(f"[EpochSync] rank={rank} epoch={epoch_idx} done_after_val")

    except Exception:
        logger.error(f"[WorkerException] rank={rank} gpu_id={gpu_id}\n{traceback.format_exc()}")
        raise
    finally:
        if arg.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    exp_time = time()
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=False)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    arg.dist_master_port = _find_available_port(arg.dist_master_port, arg.dist_master_addr)
    os.environ["MASTER_PORT"] = arg.dist_master_port
    # must have equal gpus on each node.
    arg.world_size = arg.n_gpus * arg.nodes
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size

    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    logger.info("====> Use Distributed Data Parallel <====")
    torch.multiprocessing.spawn(main_worker, args=(cfg, arg, exp_time), nprocs=arg.n_gpus)
