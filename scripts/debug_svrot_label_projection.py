import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.datasets import create_dataset
from lib.utils.config import get_config
from lib.utils.transform import bchw_2_bhwc, denormalize, rot6d_to_rotmat


def _to_numpy_image(tensor_image):
    if isinstance(tensor_image, np.ndarray):
        if tensor_image.ndim == 3 and tensor_image.shape[0] == 3:
            tensor_image = torch.from_numpy(tensor_image)
        else:
            return tensor_image.copy()
    image = tensor_image.detach().cpu().unsqueeze(0)
    image = bchw_2_bhwc(denormalize(image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).clamp_(0.0, 255.0).byte().squeeze(0).numpy()
    return image.copy()


def _project(points_xyz, intr):
    pts = points_xyz.astype(np.float32)
    proj = (intr @ pts.T).T
    z = proj[:, 2:3].copy()
    z[np.abs(z) < 1e-6] = 1e-6
    uv = np.concatenate([proj[:, 0:1] / z, proj[:, 1:2] / z], axis=-1)
    return uv


def _draw_points(image, points, color, radius=1):
    out = image.copy()
    h, w = out.shape[:2]
    for x, y in points:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(out, (xi, yi), radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def _draw_hand_joints(image, joints_2d, color=(255, 255, 0)):
    out = image.copy()
    h, w = out.shape[:2]
    for x, y in joints_2d:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(out, (xi, yi), radius=2, color=color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out", default="tmp_debug/svrot_label_projection.png")
    args = parser.parse_args()

    cfg = get_config(args.cfg, merge=True)
    dataset_cfg = getattr(cfg.DATASET, args.split.upper())
    dataset = create_dataset(dataset_cfg, data_preset=cfg.DATA_PRESET)
    if hasattr(dataset, "set_stage"):
        dataset.set_stage("stage2")

    sample = dataset[args.sample_idx]

    images = sample["image"]
    joints_2d = sample["target_joints_2d"]
    joints_3d = sample["target_joints_3d"]
    cam_intr = sample["target_cam_intr"]
    obj_sparse_gt = sample["target_obj_pc_sparse"]
    obj_template_rest = sample["target_obj_pc_sparse_rest"]
    obj_rot6d = sample["target_rot6d_label"]
    obj_t_rel = sample["target_t_label_rel"]

    root_xyz = joints_3d[:, 9:10, :]
    rotmat = rot6d_to_rotmat(torch.from_numpy(obj_rot6d).float()).cpu().numpy()
    obj_recon = np.matmul(obj_template_rest, np.transpose(rotmat, (0, 2, 1))) + (root_xyz + obj_t_rel[:, None, :])

    tiles = []
    errors = []
    num_views = images.shape[0]
    for vid in range(num_views):
        img = _to_numpy_image(images[vid])
        gt_uv = _project(obj_sparse_gt[vid], cam_intr[vid])
        recon_uv = _project(obj_recon[vid], cam_intr[vid])
        hand_uv = joints_2d[vid]
        reproj_err = float(np.linalg.norm(recon_uv - gt_uv, axis=-1).mean())
        errors.append(reproj_err)

        panel_gt = _draw_hand_joints(img.copy(), hand_uv, color=(255, 255, 0))
        panel_gt = _draw_points(panel_gt, gt_uv, color=(0, 0, 255), radius=1)
        cv2.putText(
            panel_gt,
            f"view {vid} GT obj",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        panel_label = _draw_hand_joints(img.copy(), hand_uv, color=(255, 255, 0))
        panel_label = _draw_points(panel_label, recon_uv, color=(0, 220, 0), radius=1)
        cv2.putText(
            panel_label,
            f"view {vid} SVRot label",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        panel_overlay = _draw_hand_joints(img.copy(), hand_uv, color=(255, 255, 0))
        panel_overlay = _draw_points(panel_overlay, gt_uv, color=(0, 0, 255), radius=1)
        panel_overlay = _draw_points(panel_overlay, recon_uv, color=(0, 220, 0), radius=1)
        cv2.putText(
            panel_overlay,
            f"view {vid} err {reproj_err:.3f}px",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel_overlay,
            "green: label recon  red: gt obj  yellow: hand joints",
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panel = cv2.hconcat([panel_gt, panel_label, panel_overlay])
        tiles.append(panel)

    rows = []
    row_width = 4
    for i in range(0, len(tiles), row_width):
        row = tiles[i:i + row_width]
        if len(row) < row_width:
            pad = np.full_like(row[0], 255)
            while len(row) < row_width:
                row.append(pad.copy())
        rows.append(cv2.hconcat(row))
    grid = cv2.vconcat(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"saved={out_path}")
    print(f"sample_idx={args.sample_idx}")
    print("mean_view_err_px=" + ",".join(f"{e:.4f}" for e in errors))
    print(f"mean_all_views_px={np.mean(errors):.4f}")


if __name__ == "__main__":
    main()
