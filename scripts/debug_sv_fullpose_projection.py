import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "models" / "common"))

import lib.models  # noqa: F401
from lib.datasets import create_dataset
from lib.utils.transform import batch_cam_intr_projection
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.net_utils import build_optimizer, clip_gradient, setup_seed
from lib.utils.transform import bchw_2_bhwc, denormalize, rot6d_to_rotmat
from lib.models.bricks.utils import orthgonalProj
from lib.viztools.draw import draw_batch_mesh_images_pred


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


def _to_numpy_image(tensor_image):
    image = tensor_image.detach().cpu().unsqueeze(0)
    image = bchw_2_bhwc(denormalize(image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).clamp_(0.0, 255.0).byte().squeeze(0).numpy()
    return image.copy()


def _draw_points(image, points, color, radius=1):
    out = image.copy()
    h, w = out.shape[:2]
    pts = np.asarray(points)
    for x, y in pts:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(out, (xi, yi), radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def _draw_text(image, lines):
    out = image.copy()
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            str(line),
            (10, 22 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _project_object_ortho(obj_rest, rot6d, trans, mano_cam, img_size):
    rotmat = rot6d_to_rotmat(rot6d.reshape(-1, 6)).view(*rot6d.shape[:-1], 3, 3)
    obj_xyz = torch.matmul(obj_rest, rotmat.transpose(-1, -2)) + trans.unsqueeze(-2)
    obj_uv = orthgonalProj(
        obj_xyz[..., :2].clone(),
        mano_cam[..., 0:1].unsqueeze(-2),
        torch.zeros_like(mano_cam[..., 1:]).unsqueeze(-2),
        img_size=img_size,
    )
    center_uv = orthgonalProj(
        trans.unsqueeze(-2)[..., :2].clone(),
        mano_cam[..., 0:1].unsqueeze(-2),
        torch.zeros_like(mano_cam[..., 1:]).unsqueeze(-2),
        img_size=img_size,
    )
    return obj_xyz, obj_uv, center_uv


def _fit_ortho_camera(points_xy, points_uv, visibility=None, eps=1e-6):
    valid = torch.isfinite(points_xy).all(dim=-1) & torch.isfinite(points_uv).all(dim=-1)
    if visibility is not None:
        valid = valid & (visibility > 0)
    valid = valid.to(dtype=points_xy.dtype)
    valid_exp = valid.unsqueeze(-1)
    valid_count = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean_xy = (points_xy * valid_exp).sum(dim=-2) / valid_count
    mean_uv = (points_uv * valid_exp).sum(dim=-2) / valid_count
    centered_xy = (points_xy - mean_xy.unsqueeze(-2)) * valid_exp
    centered_uv = (points_uv - mean_uv.unsqueeze(-2)) * valid_exp
    scale_num = (centered_xy * centered_uv).sum(dim=(-2, -1))
    scale_den = centered_xy.pow(2).sum(dim=(-2, -1)).clamp(min=eps)
    scale = scale_num / scale_den
    trans = mean_uv - scale.unsqueeze(-1) * mean_xy
    return torch.cat((scale.unsqueeze(-1), trans), dim=-1)


def _build_loader(dataset, batch_size, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_init_fn,
        persistent_workers=False,
    )


def _run_brief_training(model, cfg, device, stage_name, steps, workers):
    if steps <= 0:
        return
    dataset = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    if hasattr(dataset, "set_stage"):
        dataset.set_stage(stage_name)
    loader = _build_loader(dataset, cfg.TRAIN.BATCH_SIZE, workers)
    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    epoch_idx = 0 if stage_name == "stage1" else max(int(cfg.TRAIN.get("STAGE1_END_EPOCH", 1)), 1)
    data_iter = iter(loader)
    model.train()
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(stage_name)
    for step_idx in range(steps):
        try:
            batch_cpu = next(data_iter)
        except StopIteration:
            break
        batch = _clone_batch_to_device(batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        interaction_mode = "hand" if stage_name == "stage1" else "ho"
        preds = model._forward_impl(batch, interaction_mode=interaction_mode)
        total_loss, loss_dict = model.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        total_loss.backward()
        if cfg.TRAIN.GRAD_CLIP_ENABLED:
            clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)
        optimizer.step()
        sv_metric = model.compute_sv_object_pose_metrics(preds, batch)
        print(
            f"train_step={step_idx:03d} "
            f"loss={float(total_loss.detach().item()):.4f} "
            f"sv_rot_deg={float(sv_metric['metric_sv_obj_rot_deg'].detach().item()):.4f} "
            f"sv_trans_mm={float(sv_metric['metric_sv_obj_trans_epe'].detach().item() * 1000.0):.4f}",
            flush=True,
        )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--train_steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="tmp_debug/sv_fullpose_projection.png")
    args = parser.parse_args()

    cfg = get_config(args.cfg, merge=True)
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this debug script.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN).to(device)
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(args.stage)

    _run_brief_training(model, cfg, device, args.stage, args.train_steps, args.workers)

    dataset_cfg = getattr(cfg.DATASET, args.split.upper())
    dataset = create_dataset(dataset_cfg, data_preset=cfg.DATA_PRESET)
    if hasattr(dataset, "set_stage"):
        dataset.set_stage(args.stage)
    sample = dataset[args.sample_idx]
    batch = _make_eval_batch(sample, device)

    model.eval()
    interaction_mode = "hand" if args.stage == "stage1" else "ho"
    with torch.no_grad():
        preds = model._forward_impl(batch, interaction_mode=interaction_mode)
        sv_metric = model.compute_sv_object_pose_metrics(preds, batch)

    images = batch["image"][0]
    batch_size, n_views = 1, images.shape[0]
    img_h, img_w = images.shape[-2:]

    pred_sv_verts_2d = preds["mano_2d_mesh_sv"].view(batch_size, n_views, 799, 2)[0, :, :model.num_hand_verts]
    pred_sv_verts_2d = model._normed_uv_to_pixel(pred_sv_verts_2d, (img_h, img_w))
    gt_verts_2d = model._normed_uv_to_pixel(batch["target_verts_uvd"][..., :2], (img_h, img_w))[0]
    gt_sv_cam = preds["mano_cam_sv"].view(batch_size, n_views, 3).detach()

    obj_rest = batch["master_obj_sparse_rest"].to(dtype=preds["mano_cam_sv"].dtype, device=device).unsqueeze(1).expand(-1, n_views, -1, -1)
    pred_mano_cam_sv = preds["mano_cam_sv"].view(batch_size, n_views, 3)
    _, pred_obj_uv, _ = _project_object_ortho(
        obj_rest,
        preds["obj_view_rot6d_cam"],
        preds["obj_view_trans"],
        pred_mano_cam_sv,
        img_w,
    )
    _, gt_obj_uv, _ = _project_object_ortho(
        obj_rest,
        batch["target_rot6d_label"].to(dtype=obj_rest.dtype),
        batch["target_t_label_rel"].to(dtype=obj_rest.dtype),
        gt_sv_cam.to(dtype=obj_rest.dtype),
        img_w,
    )
    gt_obj_uv_persp = batch_cam_intr_projection(
        batch["target_cam_intr"],
        batch["target_obj_pc_sparse"],
    )[0]
    gt_obj_center_uv_persp = model._normed_uv_to_pixel(batch["target_obj_center_uv"], (img_h, img_w))[0]
    pred_obj_uv = torch.nan_to_num(pred_obj_uv[0], nan=0.0, posinf=max(img_h, img_w), neginf=0.0)
    gt_obj_uv = torch.nan_to_num(gt_obj_uv[0], nan=0.0, posinf=max(img_h, img_w), neginf=0.0)
    gt_obj_uv_persp = torch.nan_to_num(gt_obj_uv_persp, nan=0.0, posinf=max(img_h, img_w), neginf=0.0)

    rot_deg = torch.rad2deg(
        model.rotation_geodesic(
            preds["obj_view_rot6d_cam"][0],
            batch["target_rot6d_label"][0],
        )
    )
    trans_mm = torch.linalg.norm(preds["obj_view_trans"][0] - batch["target_t_label_rel"][0], dim=-1) * 1000.0

    mesh_panels = draw_batch_mesh_images_pred(
        gt_verts2d=gt_verts_2d,
        pred_verts2d=pred_sv_verts_2d,
        face=model.face,
        gt_obj2d=gt_obj_uv_persp,
        pred_obj2d=pred_obj_uv,
        gt_objc2d=gt_obj_center_uv_persp,
        pred_objc2d=pred_obj_uv.mean(dim=1, keepdim=True),
        intr=batch["target_cam_intr"][0],
        tensor_image=images,
        pred_obj_error=trans_mm.unsqueeze(-1),
        n_sample=n_views,
    )
    panels = []
    for vid in range(n_views):
        panel = mesh_panels[vid].copy()
        panel = _draw_text(
            panel,
            [
                f"view {vid}",
                f"rot {float(rot_deg[vid].item()):.2f} deg",
                f"tr {float(trans_mm[vid].item()):.2f} mm",
                "left: pred  right: gt",
            ],
        )
        panels.append(panel)

    rows = []
    row_width = 4
    pad = np.full_like(panels[0], 255)
    for i in range(0, len(panels), row_width):
        row = panels[i:i + row_width]
        while len(row) < row_width:
            row.append(pad.copy())
        rows.append(cv2.hconcat(row))
    grid = cv2.vconcat(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"saved={out_path}")
    print(f"sample_idx={args.sample_idx}")
    print(f"stage={args.stage}")
    print(f"train_steps={args.train_steps}")
    print(f"sv_rot_deg_mean={float(rot_deg.mean().item()):.4f}")
    print(f"sv_trans_mm_mean={float(trans_mm.mean().item()):.4f}")
    print(f"metric_sv_obj_rot_deg={float(sv_metric['metric_sv_obj_rot_deg'].detach().item()):.4f}")
    print(f"metric_sv_obj_trans_mm={float(sv_metric['metric_sv_obj_trans_epe'].detach().item() * 1000.0):.4f}")


if __name__ == "__main__":
    main()
