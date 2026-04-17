import os
import uuid

import cv2
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils.heatmap import sample_with_heatmap
from ..utils.transform import bchw_2_bhwc, denormalize
from .misc import COLOR_CONST


def draw_batch_mesh_images_pred(
    gt_verts2d,
    pred_verts2d,
    face,
    gt_obj2d,
    pred_obj2d,
    gt_objc2d,
    pred_objc2d,
    intr,
    tensor_image,
    pred_obj_conf=None,
    pred_obj_error=None,
    pred_obj_rot_error=None,
    pred_obj_trans_error=None,
    n_sample=16,
):
    batch_size = gt_verts2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    gt_verts2d = gt_verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NV, 2)
    pred_verts2d = pred_verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NV, 2)
    gt_obj2d = gt_obj2d[:n_sample, ...].detach().cpu().numpy()  # (B, N_obj, 2)
    pred_obj2d = pred_obj2d[:n_sample, ...].detach().cpu().numpy()  # (B, N_obj, 2)
    if gt_objc2d is not None:
        gt_objc2d = gt_objc2d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 2)
    if pred_objc2d is not None:
        pred_objc2d = pred_objc2d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 2)
    intr = intr[:n_sample, ...].detach().cpu().numpy()  # (B, 3, 3)
    if pred_obj_conf is not None:
        pred_obj_conf = _tensor_to_safe_numpy(pred_obj_conf, n_sample=n_sample, nan_value=0.0)
    if pred_obj_error is not None:
        pred_obj_error = _tensor_to_safe_numpy(pred_obj_error, n_sample=n_sample, nan_value=0.0)
    if pred_obj_rot_error is not None:
        pred_obj_rot_error = _tensor_to_safe_numpy(pred_obj_rot_error, n_sample=n_sample, nan_value=0.0)
    if pred_obj_trans_error is not None:
        pred_obj_trans_error = _tensor_to_safe_numpy(pred_obj_trans_error, n_sample=n_sample, nan_value=0.0)

    def _broadcast_first_dim(value):
        if value is None:
            return None
        if value.shape[0] == 1 and n_sample > 1:
            repeat_shape = [n_sample] + [1] * (value.ndim - 1)
            value = np.tile(value, repeat_shape)
        return value

    gt_objc2d = _broadcast_first_dim(gt_objc2d)
    pred_objc2d = _broadcast_first_dim(pred_objc2d)
    pred_obj_conf = _broadcast_first_dim(pred_obj_conf)
    pred_obj_error = _broadcast_first_dim(pred_obj_error)
    pred_obj_rot_error = _broadcast_first_dim(pred_obj_rot_error)
    pred_obj_trans_error = _broadcast_first_dim(pred_obj_trans_error)

    sample_list = []
    for i in range(n_sample):
        gt_verts2d_i = gt_verts2d[i].copy()
        gt_obj2d_i = gt_obj2d[i].copy()
        gt_objc2d_i = None if gt_objc2d is None else gt_objc2d[i].copy()
        pred_verts2d_i = pred_verts2d[i].copy()
        pred_obj2d_i = pred_obj2d[i].copy()
        pred_objc2d_i = None if pred_objc2d is None else pred_objc2d[i].copy()

        intr_i = intr[i].copy()

        gt_mesh_img = draw_hand_obj_mesh(image[i].copy(), gt_verts2d_i, face, gt_obj2d_i, gt_objc2d_i)
        pred_mesh_img = draw_hand_obj_mesh(image[i].copy(), pred_verts2d_i, face, pred_obj2d_i, pred_objc2d_i)
        if pred_obj_conf is not None:
            conf_value = float(pred_obj_conf[i].reshape(-1)[0])
            cv2.putText(
                pred_mesh_img,
                f"obj conf {conf_value:.4f}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 90, 220),
                1,
                cv2.LINE_AA,
            )
        if pred_obj_error is not None:
            err_value = float(pred_obj_error[i].reshape(-1)[0])
            cv2.putText(
                pred_mesh_img,
                f"obj err {err_value:.1f}",
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 90, 30),
                1,
                cv2.LINE_AA,
            )
        if pred_obj_rot_error is not None:
            err_value = float(pred_obj_rot_error[i].reshape(-1)[0])
            cv2.putText(
                pred_mesh_img,
                f"rot err {err_value:.1f} deg",
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 90, 30),
                1,
                cv2.LINE_AA,
            )
        if pred_obj_trans_error is not None:
            err_value = float(pred_obj_trans_error[i].reshape(-1)[0])
            cv2.putText(
                pred_mesh_img,
                f"tr err {err_value:.1f} mm",
                (10, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 140, 220),
                1,
                cv2.LINE_AA,
            )

        sample = np.hstack([pred_mesh_img, gt_mesh_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGBA2RGB)
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_hand_mesh_images_2d(gt_verts2d, pred_verts2d, face, tensor_image, n_sample=16):
    batch_size = gt_verts2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    gt_verts2d = gt_verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NV, 2)
    pred_verts2d = pred_verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NV, 2)

    sample_list = []
    for i in range(n_sample):
        gt_mesh_img = draw_hand_mesh(image[i].copy(), gt_verts2d[i].copy(), face)
        pred_mesh_img = draw_hand_mesh(image[i].copy(), pred_verts2d[i].copy(), face)

        sample = np.hstack([pred_mesh_img, gt_mesh_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGBA2RGB)
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def draw_batch_mesh_images_gt(gt_verts3d, face, gt_obj3d, gt_objc3d, intr, tensor_image, n_sample=16):
    batch_size = gt_verts3d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    gt_verts3d = gt_verts3d[:n_sample, ...].detach().cpu().numpy()  # (B, NV, 3)
    gt_obj3d = gt_obj3d[:n_sample, ...].detach().cpu().numpy()  # (B, N_obj, 3)
    gt_objc3d = gt_objc3d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 3)
    intr = intr[:n_sample, ...].detach().cpu().numpy()  # (B, 3, 3)

    sample_list = []
    for i in range(n_sample):
        gt_verts3d_i = gt_verts3d[i].copy()
        gt_obj3d_i = gt_obj3d[i].copy()
        gt_objc3d_i = gt_objc3d[i].copy()

        intr_i = intr[i].copy()

        gt_mesh_img = draw_hand_obj_mesh(image[i].copy(), intr_i, gt_verts3d_i, face, gt_obj3d_i, gt_objc3d_i)

        sample = gt_mesh_img
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGBA2RGB)
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_joint_center_images(joints2d, gt_jointd2d, objc2d, gt_objc2d, tensor_image, n_sample=16):
    batch_size = joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    joints2d = joints2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_jointd2d = gt_jointd2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    objc2d = objc2d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 2)
    gt_objc2d = gt_objc2d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 2)


    sample_list = []
    for i in range(n_sample):
        joints_img = plot_hand_objc(image[i].copy(), joints2d[i], objc2d[i])
        gt_joints_img = plot_hand_objc(image[i].copy(), gt_jointd2d[i], gt_objc2d[i])
        sample = np.hstack([joints_img, gt_joints_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_mesh_images(verts3d, gt_verts3d, face, intr, tensor_image, step_idx, n_sample=16):
    batch_size = verts3d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    verts3d = verts3d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_verts3d = gt_verts3d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    intr = intr[:n_sample, ...].detach().cpu().numpy()  # (B, 3, 3)

    sample_list = []
    for i in range(n_sample):
        verts3d_i = verts3d[i].copy()
        gt_verts3d_i = gt_verts3d[i].copy()
        intr_i = intr[i].copy()

        pred_mesh_img = draw_mesh(image[i].copy(), intr_i, verts3d_i, face)
        gt_mesh_img = draw_mesh(image[i].copy(), intr_i, gt_verts3d_i, face)

        sample = np.hstack([pred_mesh_img, gt_mesh_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGBA2RGB)
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_verts_images(verts2d, gt_verts2d, tensor_image, step_idx, n_sample=16):
    batch_size = verts2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    verts2d = verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_verts2d = gt_verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)

    sample_list = []
    for i in range(n_sample):
        sample_img = image[i].copy()
        for j in range(verts2d[i].shape[0]):
            cx = int(verts2d[i, j, 0])
            cy = int(verts2d[i, j, 1])
            cv2.circle(sample_img, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 1.0, 0.0]) * 255)

        sample_img_2 = image[i].copy()
        for j in range(gt_verts2d[i].shape[0]):
            cx = int(gt_verts2d[i, j, 0])
            cy = int(gt_verts2d[i, j, 1])
            cv2.circle(sample_img_2, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)

        sample = np.hstack([sample_img, sample_img_2])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_joint_images(joints2d, gt_jointd2d, tensor_image, step_idx, n_sample=16):
    batch_size = joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    joints2d = joints2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_jointd2d = gt_jointd2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)

    sample_list = []
    for i in range(n_sample):
        joints_img = plot_hand(image[i].copy(), joints2d[i])
        gt_joints_img = plot_hand(image[i].copy(), gt_jointd2d[i])
        sample = np.hstack([joints_img, gt_joints_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_object_kp_images(obj_kp2d, tensor_image, n_sample=16, draw_index=True, title="Object KP21"):
    batch_size = obj_kp2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    obj_kp2d = _tensor_to_safe_numpy(obj_kp2d, n_sample=n_sample, nan_value=-1.0)

    sample_list = []
    num_points = obj_kp2d.shape[1] if obj_kp2d.ndim >= 3 else 0
    color_map = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    denom = max(num_points - 1, 1)

    for i in range(n_sample):
        overlay = image[i].copy()
        cv2.putText(
            overlay,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        for kp_idx, coord in enumerate(obj_kp2d[i]):
            if not np.isfinite(coord).all():
                continue
            x, y = np.round(coord[:2]).astype(np.int32)
            color_val = np.uint8(round(255.0 * (kp_idx / denom)))
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), color_map)[0, 0]
            color = tuple(int(c) for c in color)
            cv2.circle(overlay, (int(x), int(y)), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), radius=6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            if draw_index:
                cv2.putText(
                    overlay,
                    str(kp_idx),
                    (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        sample = cv2.copyMakeBorder(overlay, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def draw_batch_object_kp_confidence_images(obj_kp2d, obj_conf, tensor_image, n_sample=16, draw_index=True, title="Object KP21 Confidence"):
    batch_size = obj_kp2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    obj_kp2d = _tensor_to_safe_numpy(obj_kp2d, n_sample=n_sample, nan_value=-1.0)
    obj_conf = _tensor_to_safe_numpy(obj_conf, n_sample=n_sample, nan_value=0.0)

    sample_list = []
    panel_w = 180
    num_points = obj_kp2d.shape[1] if obj_kp2d.ndim >= 3 else 0
    color_map = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    denom = max(num_points - 1, 1)

    for i in range(n_sample):
        overlay = image[i].copy()
        conf_i = obj_conf[i].reshape(-1)
        conf_scale = _finite_max(conf_i, default=1.0)

        cv2.putText(overlay, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (24, 24, 24), 1, cv2.LINE_AA)

        for kp_idx, coord in enumerate(obj_kp2d[i]):
            if not np.isfinite(coord).all():
                continue
            x, y = np.round(coord[:2]).astype(np.int32)
            conf_norm = _safe_prob(conf_i[kp_idx] / conf_scale)
            color_val = np.uint8(round(255.0 * (kp_idx / denom)))
            base_color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), color_map)[0, 0]
            base_color = tuple(int(c) for c in base_color)
            conf_color = tuple(int(c) for c in _confidence_to_bgr(conf_norm))
            radius = int(3 + round(conf_norm * 5))
            cv2.circle(overlay, (int(x), int(y)), radius=radius + 2, color=base_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), radius=radius, color=conf_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), radius=max(1, radius // 2), color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            if draw_index:
                cv2.putText(
                    overlay,
                    f"{kp_idx}:{conf_i[kp_idx]:.2f}",
                    (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    conf_color,
                    1,
                    cv2.LINE_AA,
                )

        panel = np.full((overlay.shape[0], panel_w, 3), 255, dtype=np.uint8)
        cv2.putText(panel, "Obj KP Confidence", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 32, 32), 1, cv2.LINE_AA)
        cv2.putText(panel, f"max={conf_scale:.4f}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (96, 96, 96), 1, cv2.LINE_AA)

        row_h = max(9, (panel.shape[0] - 48) // max(num_points, 1))
        bar_x0 = 42
        bar_x1 = panel_w - 12
        for kp_idx in range(num_points):
            conf_norm = _safe_prob(conf_i[kp_idx] / conf_scale)
            color = _confidence_to_bgr(conf_norm)
            y = 48 + kp_idx * row_h
            cv2.putText(panel, f"{kp_idx:02d}", (6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (48, 48, 48), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + 6), (230, 230, 230), thickness=-1)
            fill_x = bar_x0 + int(round((bar_x1 - bar_x0) * np.clip(conf_norm, 0.0, 1.0)))
            cv2.rectangle(panel, (bar_x0, y), (fill_x, y + 6), tuple(int(c) for c in color), thickness=-1)

        sample = np.hstack([overlay, panel])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def _confidence_to_bgr(conf_value):
    conf_value = float(conf_value)
    if not np.isfinite(conf_value):
        conf_value = 0.0
    conf_uint8 = np.uint8(np.clip(conf_value, 0.0, 1.0) * 255.0)
    color_map = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    return cv2.applyColorMap(np.array([[conf_uint8]], dtype=np.uint8), color_map)[0, 0]


def _safe_float(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def _safe_prob(value, default=0.0):
    return float(np.clip(_safe_float(value, default=default), 0.0, 1.0))


def _finite_max(values, default=1.0):
    values = np.asarray(values).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float(default)
    return float(max(finite_values.max(), 1e-6))


def _tensor_to_safe_numpy(value, n_sample=None, nan_value=0.0, posinf_value=None, neginf_value=None):
    if value is None:
        return None
    if n_sample is not None:
        value = value[:n_sample, ...]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if posinf_value is None:
        posinf_value = nan_value
    if neginf_value is None:
        neginf_value = nan_value
    return np.nan_to_num(value, nan=nan_value, posinf=posinf_value, neginf=neginf_value)


def draw_batch_joint_confidence_images(joints2d, joint_conf, tensor_image, obj_center2d=None, obj_conf=None, n_sample=16):
    batch_size = joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    joints2d = _tensor_to_safe_numpy(joints2d, n_sample=n_sample, nan_value=-1.0)
    joint_conf = _tensor_to_safe_numpy(joint_conf, n_sample=n_sample, nan_value=0.0)
    if obj_center2d is not None:
        obj_center2d = _tensor_to_safe_numpy(obj_center2d, n_sample=n_sample, nan_value=-1.0)
    if obj_conf is not None:
        obj_conf = _tensor_to_safe_numpy(obj_conf, n_sample=n_sample, nan_value=0.0)

    sample_list = []
    panel_w = 180
    for i in range(n_sample):
        joints2d_i = joints2d[i]
        joint_conf_i = joint_conf[i]
        obj_center2d_i = None if obj_center2d is None else obj_center2d[i]
        obj_conf_i = None if obj_conf is None else obj_conf[i]
        overlay = image[i].copy()
        overlay = _plot_hand_uniform(overlay, joints2d_i, color=(180, 180, 180), linewidth=1, radius=1)

        conf_values = joint_conf_i.reshape(-1)
        if obj_conf_i is not None:
            conf_values = np.concatenate([conf_values, obj_conf_i.reshape(-1)], axis=0)
        conf_scale = _finite_max(conf_values, default=1.0)

        for joint_idx in range(joints2d_i.shape[0]):
            conf_norm = _safe_prob(joint_conf_i[joint_idx] / conf_scale)
            color = _confidence_to_bgr(conf_norm)
            center = tuple(np.round(joints2d_i[joint_idx, :2]).astype(np.int32))
            radius = int(3 + round(conf_norm * 5))
            cv2.circle(overlay, center, radius=radius, color=tuple(int(c) for c in color), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, center, radius=max(1, radius // 2), color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        if obj_center2d_i is not None:
            obj_xy = np.round(obj_center2d_i.reshape(-1, 2)[0]).astype(np.int32)
            obj_conf_norm = _safe_prob(obj_conf_i.reshape(-1)[0] / conf_scale) if obj_conf_i is not None else 0.0
            obj_conf_value = float(obj_conf_i.reshape(-1)[0]) if obj_conf_i is not None else 0.0
            obj_color = _confidence_to_bgr(obj_conf_norm)
            marker_size = int(8 + round(obj_conf_norm * 6))
            cv2.drawMarker(
                overlay,
                tuple(obj_xy),
                color=tuple(int(c) for c in obj_color),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=marker_size,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"obj {obj_conf_value:.4f}",
                (int(obj_xy[0]) + 6, int(obj_xy[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                tuple(int(c) for c in obj_color),
                1,
                cv2.LINE_AA,
            )

        panel = np.full((overlay.shape[0], panel_w, 3), 255, dtype=np.uint8)
        cv2.putText(panel, "Joint Confidence", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 32, 32), 1, cv2.LINE_AA)
        cv2.putText(panel, f"max={conf_scale:.4f}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (96, 96, 96), 1, cv2.LINE_AA)

        row_h = max(9, (panel.shape[0] - 48) // (joints2d_i.shape[0] + (1 if obj_center2d_i is not None else 0)))
        bar_x0 = 36
        bar_x1 = panel_w - 12
        for joint_idx in range(joints2d_i.shape[0]):
            conf_norm = _safe_prob(joint_conf_i[joint_idx] / conf_scale)
            color = _confidence_to_bgr(conf_norm)
            y = 48 + joint_idx * row_h
            cv2.putText(panel, f"{joint_idx:02d}", (6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (48, 48, 48), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + 6), (230, 230, 230), thickness=-1)
            fill_x = bar_x0 + int(round((bar_x1 - bar_x0) * np.clip(conf_norm, 0.0, 1.0)))
            cv2.rectangle(panel, (bar_x0, y), (fill_x, y + 6), tuple(int(c) for c in color), thickness=-1)

        if obj_center2d_i is not None:
            obj_conf_norm = _safe_prob(obj_conf_i.reshape(-1)[0] / conf_scale) if obj_conf_i is not None else 0.0
            obj_conf_value = float(obj_conf_i.reshape(-1)[0]) if obj_conf_i is not None else 0.0
            color = _confidence_to_bgr(obj_conf_norm)
            y = 48 + joints2d_i.shape[0] * row_h
            cv2.putText(panel, f"OBJ {obj_conf_value:.4f}", (6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (48, 48, 48), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + 6), (230, 230, 230), thickness=-1)
            fill_x = bar_x0 + int(round((bar_x1 - bar_x0) * np.clip(obj_conf_norm, 0.0, 1.0)))
            cv2.rectangle(panel, (bar_x0, y), (fill_x, y + 6), tuple(int(c) for c in color), thickness=-1)

        sample = np.hstack([overlay, panel])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def draw_batch_obj_view_rotation_images(
    tensor_image,
    rot_conf,
    fused_weights,
    rot_deg=None,
    valid_mask=None,
    obj_center2d=None,
    is_master=None,
    is_best=None,
    n_sample=16,
):
    batch_size = tensor_image.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    rot_conf = _tensor_to_safe_numpy(rot_conf, n_sample=n_sample, nan_value=0.0)
    fused_weights = _tensor_to_safe_numpy(fused_weights, n_sample=n_sample, nan_value=0.0)
    if rot_deg is not None:
        rot_deg = _tensor_to_safe_numpy(rot_deg, n_sample=n_sample, nan_value=-1.0)
    if valid_mask is not None:
        valid_mask = _tensor_to_safe_numpy(valid_mask, n_sample=n_sample, nan_value=0.0)
    if obj_center2d is not None:
        obj_center2d = _tensor_to_safe_numpy(obj_center2d, n_sample=n_sample, nan_value=-1.0)
    if is_master is not None:
        is_master = _tensor_to_safe_numpy(is_master, n_sample=n_sample, nan_value=0.0)
    if is_best is not None:
        is_best = _tensor_to_safe_numpy(is_best, n_sample=n_sample, nan_value=0.0)

    panel_w = 220
    bar_x0 = 84
    bar_x1 = panel_w - 16
    bar_h = 12
    sample_list = []
    for i in range(n_sample):
        overlay = image[i].copy()
        master_flag = bool(is_master[i].reshape(-1)[0]) if is_master is not None else False
        best_flag = bool(is_best[i].reshape(-1)[0]) if is_best is not None else False
        obj_center2d_i = None if obj_center2d is None else obj_center2d[i]
        if obj_center2d is not None:
            obj_xy = np.round(obj_center2d_i.reshape(-1, 2)[0]).astype(np.int32)
            cv2.drawMarker(
                overlay,
                tuple(obj_xy),
                color=(0, 180, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        if best_flag:
            cv2.rectangle(
                overlay,
                (8, 8),
                (overlay.shape[1] - 9, overlay.shape[0] - 9),
                (230, 150, 30),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "BEST",
                (10, 48 if master_flag else 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 150, 30),
                2,
                cv2.LINE_AA,
            )
        if master_flag:
            cv2.rectangle(
                overlay,
                (2, 2),
                (overlay.shape[1] - 3, overlay.shape[0] - 3),
                (40, 180, 40),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "MASTER",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 180, 40),
                2,
                cv2.LINE_AA,
            )

        panel = np.full((overlay.shape[0], panel_w, 3), 255, dtype=np.uint8)
        cv2.putText(panel, "Object Rotation View", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 32, 32), 1, cv2.LINE_AA)
        if master_flag:
            cv2.putText(panel, "master view", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 180, 40), 1, cv2.LINE_AA)
        if best_flag:
            cv2.putText(panel, "best fused weight", (10, 48 if master_flag else 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 150, 30), 1, cv2.LINE_AA)

        conf_value = _safe_prob(rot_conf[i].reshape(-1)[0])
        weight_value = _safe_prob(fused_weights[i].reshape(-1)[0])
        err_deg = _safe_float(rot_deg[i].reshape(-1)[0], default=-1.0) if rot_deg is not None else -1.0
        is_valid = _safe_float(valid_mask[i].reshape(-1)[0], default=0.0) if valid_mask is not None else 1.0

        rows = [
            ("rot_conf", conf_value, 1.0, True),
            ("fused_w", weight_value, 1.0, True),
        ]
        if err_deg >= 0.0:
            rows.append(("rot_deg", err_deg, max(30.0, err_deg), False))

        y0 = 62 if (master_flag and best_flag) else 52 if (master_flag or best_flag) else 42
        for row_idx, (label, value, denom, use_conf_cmap) in enumerate(rows):
            y = y0 + row_idx * 32
            cv2.putText(panel, label, (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (64, 64, 64), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + bar_h), (235, 235, 235), thickness=-1)
            frac = float(np.clip(value / max(denom, 1e-6), 0.0, 1.0))
            fill_x = bar_x0 + int(round((bar_x1 - bar_x0) * frac))
            if use_conf_cmap:
                color = tuple(int(c) for c in _confidence_to_bgr(frac))
            else:
                color = (60, 60 + int(160 * (1.0 - frac)), 230 - int(160 * (1.0 - frac)))
            cv2.rectangle(panel, (bar_x0, y), (fill_x, y + bar_h), color, thickness=-1)
            cv2.putText(panel, f"{value:.3f}" if use_conf_cmap else f"{value:.1f}", (bar_x0, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (48, 48, 48), 1, cv2.LINE_AA)

        status_y = y0 + len(rows) * 32 + 8
        status_color = (48, 160, 48) if is_valid > 0.5 else (64, 64, 200)
        cv2.putText(panel, f"valid: {int(is_valid > 0.5)}", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        sample = np.hstack([overlay, panel])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def _plot_hand_uniform(image, coords_hw, color, vis=None, linewidth=2, radius=3):
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    color = tuple(int(c) for c in color)
    for start, end in bones:
        if (vis[start] == False) or (vis[end] == False):
            continue
        p1 = tuple(coords_hw[start, :2].astype(np.int32))
        p2 = tuple(coords_hw[end, :2].astype(np.int32))
        cv2.line(image, p1, p2, color=color, thickness=linewidth, lineType=cv2.LINE_AA)

    for i in range(coords_hw.shape[0]):
        if vis[i] == False:
            continue
        p = tuple(coords_hw[i, :2].astype(np.int32))
        cv2.circle(image, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return image


def draw_batch_joint_triplet_overlay_images(sv_joints2d, refined_joints2d, gt_joints2d, tensor_image, n_sample=16):
    batch_size = sv_joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    sv_joints2d = sv_joints2d[:n_sample, ...].detach().cpu().numpy()
    refined_joints2d = refined_joints2d[:n_sample, ...].detach().cpu().numpy()
    gt_joints2d = gt_joints2d[:n_sample, ...].detach().cpu().numpy()

    sample_list = []
    for i in range(n_sample):
        overlay = image[i].copy()
        overlay = _plot_hand_uniform(overlay, gt_joints2d[i], color=(255, 255, 255), linewidth=2, radius=2)
        overlay = _plot_hand_uniform(overlay, sv_joints2d[i], color=(0, 255, 255), linewidth=2, radius=2)
        overlay = _plot_hand_uniform(overlay, refined_joints2d[i], color=(255, 64, 64), linewidth=2, radius=2)
        sample = cv2.copyMakeBorder(overlay, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def draw_batch_joint_pair_overlay_images(first_joints2d, second_joints2d, tensor_image, n_sample=16):
    batch_size = first_joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    first_joints2d = first_joints2d[:n_sample, ...].detach().cpu().numpy()
    second_joints2d = second_joints2d[:n_sample, ...].detach().cpu().numpy()

    sample_list = []
    for i in range(n_sample):
        overlay = image[i].copy()
        overlay = _plot_hand_uniform(overlay, first_joints2d[i], color=(0, 255, 255), linewidth=2, radius=2)
        overlay = _plot_hand_uniform(overlay, second_joints2d[i], color=(255, 64, 64), linewidth=2, radius=2)
        sample = cv2.copyMakeBorder(overlay, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def draw_batch_vertex_triplet_overlay_images(gt_verts2d, first_verts2d, second_verts2d, tensor_image, n_sample=16):
    batch_size = gt_verts2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)

    gt_verts2d = _tensor_to_safe_numpy(gt_verts2d, n_sample=n_sample, nan_value=-1.0)
    first_verts2d = _tensor_to_safe_numpy(first_verts2d, n_sample=n_sample, nan_value=-1.0)
    second_verts2d = _tensor_to_safe_numpy(second_verts2d, n_sample=n_sample, nan_value=-1.0)

    sample_list = []
    for i in range(n_sample):
        overlay = image[i].copy()
        for verts, color, label in [
            (gt_verts2d[i], (255, 255, 255), "GT"),
            (first_verts2d[i], (0, 255, 255), "SV"),
            (second_verts2d[i], (255, 72, 72), "KP"),
        ]:
            valid = np.isfinite(verts).all(axis=-1)
            pts = np.round(verts[valid]).astype(np.int32)
            for x, y in pts[:: max(1, len(pts) // 256)]:
                cv2.circle(overlay, (int(x), int(y)), radius=1, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                label,
                (10, 22 + (18 if label != "GT" else 0) + (18 if label == "KP" else 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        sample = cv2.copyMakeBorder(overlay, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def tile_batch_images(sample_array, max_cols=4, pad_value=255):
    if sample_array.ndim == 3:
        return sample_array

    batch_size, height, width, channels = sample_array.shape
    cols = min(max_cols, batch_size)
    rows = int(np.ceil(batch_size / cols))

    tiled = np.full((rows * height, cols * width, channels), pad_value, dtype=sample_array.dtype)
    for idx in range(batch_size):
        row = idx // cols
        col = idx % cols
        tiled[row * height:(row + 1) * height, col * width:(col + 1) * width] = sample_array[idx]

    return tiled


def draw_batch_master_space_3d(gt_hand_joints, pred_hand_joints, gt_hand_verts, pred_hand_verts,
                               gt_obj_pts, pred_obj_pts, pred_hand_anchors=None, n_sample=4,
                               image_size=(256, 256), obj_subsample=512, verts_subsample=256):
    batch_size = gt_hand_joints.shape[0]
    n_sample = min(n_sample, batch_size)

    gt_hand_joints = gt_hand_joints[:n_sample].detach().cpu().numpy()
    pred_hand_joints = pred_hand_joints[:n_sample].detach().cpu().numpy()
    gt_hand_verts = gt_hand_verts[:n_sample].detach().cpu().numpy()
    pred_hand_verts = pred_hand_verts[:n_sample].detach().cpu().numpy()
    if gt_obj_pts is not None:
        gt_obj_pts = gt_obj_pts[:n_sample].detach().cpu().numpy()
    if pred_obj_pts is not None:
        pred_obj_pts = pred_obj_pts[:n_sample].detach().cpu().numpy()
    if pred_hand_anchors is not None:
        pred_hand_anchors = pred_hand_anchors[:n_sample].detach().cpu().numpy()

    sample_list = []
    for i in range(n_sample):
        fig = plt.figure()
        fig.set_size_inches(float(image_size[1] * 2) / fig.dpi, float(image_size[0]) / fig.dpi, forward=True)

        gt_ax = fig.add_subplot(1, 2, 1, projection="3d")
        pred_ax = fig.add_subplot(1, 2, 2, projection="3d")

        gt_vert_idx = np.linspace(0, gt_hand_verts[i].shape[0] - 1, min(verts_subsample, gt_hand_verts[i].shape[0])).astype(np.int32)
        pred_vert_idx = np.linspace(0, pred_hand_verts[i].shape[0] - 1, min(verts_subsample, pred_hand_verts[i].shape[0])).astype(np.int32)
        gt_obj_idx = None
        pred_obj_idx = None
        if gt_obj_pts is not None:
            gt_obj_idx = np.linspace(0, gt_obj_pts[i].shape[0] - 1, min(obj_subsample, gt_obj_pts[i].shape[0])).astype(np.int32)
        if pred_obj_pts is not None:
            pred_obj_idx = np.linspace(0, pred_obj_pts[i].shape[0] - 1, min(obj_subsample, pred_obj_pts[i].shape[0])).astype(np.int32)

        gt_ax.scatter(gt_hand_verts[i][gt_vert_idx, 0], gt_hand_verts[i][gt_vert_idx, 1], gt_hand_verts[i][gt_vert_idx, 2],
                      c="#7cc7f2", s=1, alpha=0.7)
        if gt_obj_pts is not None:
            gt_ax.scatter(gt_obj_pts[i][gt_obj_idx, 0], gt_obj_pts[i][gt_obj_idx, 1], gt_obj_pts[i][gt_obj_idx, 2],
                          c="#f26d6d", s=1, alpha=0.7)
        gt_ax.scatter(gt_hand_joints[i][:, 0], gt_hand_joints[i][:, 1], gt_hand_joints[i][:, 2],
                      c="#0b5fa5", s=10)
        gt_ax.set_title("GT Master")

        pred_ax.scatter(pred_hand_verts[i][pred_vert_idx, 0], pred_hand_verts[i][pred_vert_idx, 1], pred_hand_verts[i][pred_vert_idx, 2],
                        c="#7cc7f2", s=1, alpha=0.7)
        if pred_obj_pts is not None:
            pred_ax.scatter(pred_obj_pts[i][pred_obj_idx, 0], pred_obj_pts[i][pred_obj_idx, 1], pred_obj_pts[i][pred_obj_idx, 2],
                            c="#f26d6d", s=1, alpha=0.7)
        pred_ax.scatter(pred_hand_joints[i][:, 0], pred_hand_joints[i][:, 1], pred_hand_joints[i][:, 2],
                        c="#0b5fa5", s=10)
        if pred_hand_anchors is not None:
            pred_ax.scatter(pred_hand_anchors[i][:, 0], pred_hand_anchors[i][:, 1], pred_hand_anchors[i][:, 2],
                            c="#f2c84b", s=6, alpha=0.8)
        pred_ax.set_title("Pred Master")

        all_points = [
            gt_hand_verts[i][gt_vert_idx], gt_hand_joints[i],
            pred_hand_verts[i][pred_vert_idx], pred_hand_joints[i]
        ]
        if gt_obj_pts is not None:
            all_points.append(gt_obj_pts[i][gt_obj_idx])
        if pred_obj_pts is not None:
            all_points.append(pred_obj_pts[i][pred_obj_idx])
        if pred_hand_anchors is not None:
            all_points.append(pred_hand_anchors[i])
        all_points = np.concatenate(all_points, axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        centers = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins) * 1.2 + 1e-6
        for ax in [gt_ax, pred_ax]:
            ax.set_xlim(centers[0] - radius, centers[0] + radius)
            ax.set_ylim(centers[1] - radius, centers[1] + radius)
            ax.set_zlim(centers[2] - radius, centers[2] + radius)
            ax.view_init(elev=25, azim=-60)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.02, hspace=0.0)
        sample = fig2data(fig)[..., :3]
        plt.close(fig)
        sample_list.append(sample[None, ...])

    return np.concatenate(sample_list, axis=0)


def plot_image_joints_mask(image, joints2d, mask):
    joints_img = plot_hand(image.copy(), joints2d)
    mask = mask[:, :, None].repeat(3, axis=2)
    mask = cv2.resize(mask, image.shape[:2])
    img_mask = cv2.addWeighted(image, 0.3, mask, 0.7, 0)
    comb_img = np.hstack([image, joints_img, img_mask])
    return comb_img


def plot_image_heatmap_mask(image, heatmap, mask):
    img_heatmap = sample_with_heatmap(image, heatmap)

    mask = mask[:, :, None].repeat(3, axis=2)
    mask = cv2.resize(mask, image.shape[:2])
    img_mask = cv2.addWeighted(image, 0.3, mask, 0.7, 0)
    comb_img = np.hstack([img_mask, img_heatmap])
    return comb_img


def imdesc(image, desc=""):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, desc, (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def plot_hand(image, coords_hw, vis=None, linewidth=3):
    """Plots a hand stick figure into a matplotlib figure."""

    colors = np.array(COLOR_CONST.color_hand_joints)
    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        c1x = int(coord1[0])
        c1y = int(coord1[1])
        c2x = int(coord2[0])
        c2y = int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color * 255, thickness=linewidth)

    for i in range(coords_hw.shape[0]):
        cx = int(coords_hw[i, 0])
        cy = int(coords_hw[i, 1])
        cv2.circle(image, (cx, cy), radius=2 * linewidth, thickness=-1, color=colors[i, :] * 255)

    return image


def plot_hand_objc(image, coords_hw, objc_hw, vis=None, linewidth=3):
    """Plots a hand stick figure into a matplotlib figure."""

    colors = np.array(COLOR_CONST.color_hand_joints)
    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        c1x = int(coord1[0])
        c1y = int(coord1[1])
        c2x = int(coord2[0])
        c2y = int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color * 255, thickness=linewidth)

    for i in range(coords_hw.shape[0]):
        cx = int(coords_hw[i, 0])
        cy = int(coords_hw[i, 1])
        cv2.circle(image, (cx, cy), radius=2 * linewidth, thickness=-1, color=colors[i, :] * 255)

    cv2.circle(image, (int(objc_hw[0, 0]), int(objc_hw[0, 1])), radius=5 * linewidth, thickness=-1, color=(255, 0, 0))

    return image


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_mesh(image, cam_param, mesh_xyz, face):
    """
    :param image: H x W x 3
    :param cam_param: 1 x 3 x 3
    :param mesh_xyz: 778 x 3
    :param face: N_face x 3
    :return:
    """
    vertex2uv = np.matmul(cam_param, mesh_xyz.T).T
    vertex2uv = (vertex2uv / vertex2uv[:, 2:3])[:, :2].astype(np.int)

    fig = plt.figure()
    fig.set_size_inches(float(image.shape[0]) / fig.dpi, float(image.shape[1]) / fig.dpi, forward=True)
    plt.imshow(image)
    plt.axis('off')
    if face is None:
        plt.plot(vertex2uv[:, 0], vertex2uv[:, 1], 'o', color='green', markersize=1)
    else:
        plt.triplot(vertex2uv[:, 0], vertex2uv[:, 1], face, lw=0.5, color='orange')

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)

    return ret


def draw_hand_obj_mesh(image, verts_uv, face, obj_uv, obj_center_uv=None):
    """
    :param image: H x W x 3
    :param verts_uv: 778 x 2
    :param face: N_face x 3
    :param obj_uv: N_obj x 2
    :param obj_center: 1 x 2
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(float(image.shape[0]) / fig.dpi, float(image.shape[1]) / fig.dpi, forward=True)
    plt.imshow(image)
    plt.axis('off')
    if face is None:
        plt.plot(verts_uv[:, 0], verts_uv[:, 1], 'o', color='green', markersize=1) 
    else:
        plt.triplot(verts_uv[:, 0], verts_uv[:, 1], face, lw=0.5, color='orange')
    plt.plot(obj_uv[:, 0], obj_uv[:, 1], 'o', color='red', markersize=2.0)
    
    if obj_center_uv is not None:
        plt.plot(obj_center_uv[:, 0], obj_center_uv[:, 1], 'o', color='blue', markersize=5)

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)

    return ret


def draw_hand_mesh(image, verts_uv, face):
    fig = plt.figure()
    fig.set_size_inches(float(image.shape[0]) / fig.dpi, float(image.shape[1]) / fig.dpi, forward=True)
    plt.imshow(image)
    plt.axis('off')
    if face is None:
        plt.plot(verts_uv[:, 0], verts_uv[:, 1], 'o', color='green', markersize=1)
    else:
        plt.triplot(verts_uv[:, 0], verts_uv[:, 1], face, lw=0.5, color='orange')

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)

    return ret


def draw_hand_obj_mesh_back(image, cam_param, mesh_xyz, face, obj_xyz, obj_center=None):
    """
    :param image: H x W x 3
    :param cam_param: 1 x 3 x 3
    :param mesh_xyz: 778 x 3
    :param face: 1538 x 3 x 2
    :param obj_xyz: N_obj x 3
    :param obj_center: 1 x 3
    :return:
    """
    vertex2uv = np.matmul(cam_param, mesh_xyz.T).T
    vertex2uv = (vertex2uv / vertex2uv[:, 2:3])[:, :2].astype(int)

    obj_uv = np.matmul(cam_param, obj_xyz.T).T
    obj_uv = (obj_uv / obj_uv[:, 2:3])[:, :2].astype(int)

    fig = plt.figure()
    fig.set_size_inches(float(image.shape[0]) / fig.dpi, float(image.shape[1]) / fig.dpi, forward=True)
    plt.imshow(image)
    plt.axis('off')
    plt.plot(obj_uv[:, 0], obj_uv[:, 1], 'o', color='red', markersize=0.5)
    if face is None:
        plt.plot(vertex2uv[:, 0], vertex2uv[:, 1], 'o', color='green', markersize=1) 
    else:
        plt.triplot(vertex2uv[:, 0], vertex2uv[:, 1], face, lw=0.5, color='orange')
    
    if obj_center is not None:
        objc_uv = np.matmul(cam_param, obj_center.T).T
        objc_uv = (objc_uv / objc_uv[:, 2:3])[:, :2].astype(int)
        plt.plot(objc_uv[:, 0], objc_uv[:, 1], 'o', color='blue', markersize=3)

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)

    return ret


def draw_2d_skeleton(image, joints_uv=None, corners_uv=None):
    """
    :param image: H x W x 3
    :param joints_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    skeleton_overlay = image.copy()
    # skeleton_overlay = skeleton_overlay[:, :, (2, 1, 0)]
    # skeleton_overlay = (skeleton_overlay * 255).astype("float32")
    # skeleton_overlay = skeleton_overlay.copy()

    if corners_uv is not None:
        for corner_idx in range(corners_uv.shape[0]):
            corner = corners_uv[corner_idx, 0].astype("int32"), corners_uv[corner_idx, 1].astype("int32")
            cv2.circle(
                skeleton_overlay,
                corner,
                radius=1,
                color=(255, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        # draw 12 segments
        #  [0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [1, 5], [2, 6], [3, 7], [0, 4]
        b_list = [0, 1, 3, 2, 0]
        for curr_id, next_id in zip(b_list[:-1], b_list[1:]):
            cv2.line(
                skeleton_overlay,
                tuple(corners_uv[curr_id, :].astype("int32")),
                tuple(corners_uv[next_id, :].astype("int32")),
                color=[255, 0, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        g_list = [4, 5, 7, 6, 4]
        for curr_id, next_id in zip(g_list[:-1], g_list[1:]):
            cv2.line(
                skeleton_overlay,
                tuple(corners_uv[curr_id, :].astype("int32")),
                tuple(corners_uv[next_id, :].astype("int32")),
                color=[0, 128, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        lb_list = [[1, 5], [2, 6], [3, 7], [0, 4]]
        for curr_id, next_id in lb_list:
            cv2.line(
                skeleton_overlay,
                tuple(corners_uv[curr_id, :].astype("int32")),
                tuple(corners_uv[next_id, :].astype("int32")),
                color=[192, 192, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    if joints_uv is not None:
        assert joints_uv.shape[0] == 21
        marker_sz = 6
        line_wd = 3
        root_ind = 0

        for joint_ind in range(joints_uv.shape[0]):
            joint = joints_uv[joint_ind, 0].astype("int32"), joints_uv[joint_ind, 1].astype("int32")
            cv2.circle(
                skeleton_overlay,
                joint,
                radius=marker_sz,
                color=COLOR_CONST.color_hand_joints[joint_ind] * np.array(255),
                thickness=-1,
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                root_joint = joints_uv[root_ind, 0].astype("int32"), joints_uv[root_ind, 1].astype("int32")
                cv2.line(
                    skeleton_overlay,
                    root_joint,
                    joint,
                    color=COLOR_CONST.color_hand_joints[joint_ind] * np.array(255),
                    thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
                )
            else:
                joint_2 = joints_uv[joint_ind - 1, 0].astype("int32"), joints_uv[joint_ind - 1, 1].astype("int32")
                cv2.line(
                    skeleton_overlay,
                    joint_2,
                    joint,
                    color=COLOR_CONST.color_hand_joints[joint_ind] * np.array(255),
                    thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
                )

    return skeleton_overlay


def axis_equal_3d(ax, ratio=1.2):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz)) * ratio
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def draw_3d_skeleton(image_size, joints_xyz=None, corners_xyz=None):
    """
    :param joints_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection="3d")

    if corners_xyz is not None:
        b_list = [0, 1, 3, 2, 0]
        for curr_id, next_id in zip(b_list[:-1], b_list[1:]):
            ax.plot(
                corners_xyz[(curr_id, next_id), 0],
                corners_xyz[(curr_id, next_id), 1],
                corners_xyz[(curr_id, next_id), 2],
                color=[255 / 255, 0, 0],
                linewidth=2,
            )

        g_list = [4, 5, 7, 6, 4]
        for curr_id, next_id in zip(g_list[:-1], g_list[1:]):
            ax.plot(
                corners_xyz[(curr_id, next_id), 0],
                corners_xyz[(curr_id, next_id), 1],
                corners_xyz[(curr_id, next_id), 2],
                color=[0, 128 / 255, 0],
                linewidth=2,
            )

        lb_list = [[1, 5], [2, 6], [3, 7], [0, 4]]
        for curr_id, next_id in lb_list:
            ax.plot(
                corners_xyz[(curr_id, next_id), 0],
                corners_xyz[(curr_id, next_id), 1],
                corners_xyz[(curr_id, next_id), 2],
                color=[192 / 255, 192 / 255, 0],
                linewidth=2,
            )

    if joints_xyz is not None:
        assert joints_xyz.shape[0] == 21
        marker_sz = 11
        line_wd = 2
        for joint_ind in range(joints_xyz.shape[0]):
            ax.plot(
                joints_xyz[joint_ind:joint_ind + 1, 0],
                joints_xyz[joint_ind:joint_ind + 1, 1],
                joints_xyz[joint_ind:joint_ind + 1, 2],
                ".",
                c=COLOR_CONST.color_hand_joints[joint_ind],
                markersize=marker_sz,
            )
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                ax.plot(
                    joints_xyz[[0, joint_ind], 0],
                    joints_xyz[[0, joint_ind], 1],
                    joints_xyz[[0, joint_ind], 2],
                    color=COLOR_CONST.color_hand_joints[joint_ind],
                    linewidth=line_wd,
                )
            else:
                ax.plot(
                    joints_xyz[[joint_ind - 1, joint_ind], 0],
                    joints_xyz[[joint_ind - 1, joint_ind], 1],
                    joints_xyz[[joint_ind - 1, joint_ind], 2],
                    color=COLOR_CONST.color_hand_joints[joint_ind],
                    linewidth=line_wd,
                )

    ax.view_init(elev=50, azim=-50)
    axis_equal_3d(ax)
    # turn off ticklabels
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.subplots_adjust(left=-0.06, right=0.98, top=0.93, bottom=-0.07, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)
    return ret


def draw_3d_mesh_mayavi(image_size, hand_xyz=None, hand_face=None, obj_xyz=None, obj_face=None, ratio=400 / 224):
    from mayavi import mlab

    mlab.options.offscreen = True
    cache_path = COLOR_CONST.mayavi_cache_path
    tempfile_name = "{}.png".format(str(uuid.uuid1()))
    os.makedirs(cache_path, exist_ok=True)

    # generate 400 x 400 fig
    tmp_img_size = (int(image_size[0] * ratio), int(image_size[1] * ratio))
    mlab_fig = mlab.figure(bgcolor=tuple(np.ones(3)), size=tmp_img_size)
    if hand_xyz is not None and hand_face is not None:
        mlab.triangular_mesh(
            hand_xyz[:, 0],
            hand_xyz[:, 1],
            hand_xyz[:, 2],
            np.array(hand_face),
            figure=mlab_fig,
            color=(0.4, 0.81960784, 0.95294118),
        )
    if obj_xyz is not None and obj_face is not None:
        mlab.triangular_mesh(
            obj_xyz[:, 0],
            obj_xyz[:, 1],
            obj_xyz[:, 2],
            np.array(obj_face),
            figure=mlab_fig,
            color=(1.0, 0.63921569, 0.6745098),
        )
    mlab.view(azimuth=-50, elevation=50, distance=0.6)
    mlab.savefig(os.path.join(cache_path, tempfile_name))
    mlab.close()

    # load by opencv and resize
    img = cv2.imread(os.path.join(cache_path, tempfile_name), cv2.IMREAD_COLOR)
    # resize to 224x224
    img = cv2.resize(img, image_size)
    os.remove(os.path.join(cache_path, tempfile_name))
    return img


def save_a_image_with_joints(image, cam_param, pose_uv, pose_xyz, file_name, padding=0, ret=False):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    skeleton_overlay = draw_2d_skeleton(image, joints_uv=pose_uv)
    skeleton_3d = draw_3d_skeleton(image.shape[:2], joints_xyz=pose_xyz)

    img_list = [skeleton_overlay, skeleton_3d]
    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    grid_image = np.zeros(((image_height + padding), num_column * (image_width + padding), 3), dtype=np.uint8)

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width + padding
        width_end = width_begin + image_width
    if ret:
        return grid_image

    cv2.imwrite(file_name, grid_image)


def save_a_image_with_mesh_joints(
    image,
    cam_param,
    mesh_xyz,
    face,
    pose_uv,
    pose_xyz,
    file_name,
    padding=0,
    ret=False,
    with_mayavi_mesh=True,
    with_skeleton_3d=True,
    renderer=None,
):
    frame = image.copy()
    rend_img_overlay = renderer(mesh_xyz, face, cam_param, img=frame)
    rend_img_overlay = cv2.cvtColor(rend_img_overlay, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    skeleton_overlay = draw_2d_skeleton(image, joints_uv=pose_uv)

    img_list = [skeleton_overlay, rend_img_overlay]
    if with_mayavi_mesh:
        mesh_3d = draw_3d_mesh_mayavi(image.shape[:2], hand_xyz=mesh_xyz, hand_face=face)
        img_list.append(mesh_3d)
    if with_skeleton_3d:
        skeleton_3d = draw_3d_skeleton(image.shape[:2], joints_xyz=pose_xyz)
        img_list.append(skeleton_3d)

    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    grid_image = np.zeros(((image_height + padding), num_column * (image_width + padding), 3), dtype=np.uint8)

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width + padding
        width_end = width_begin + image_width
    if ret:
        return grid_image

    cv2.imwrite(file_name, grid_image)


def save_a_image_with_mesh_joints_objects(
    image,
    cam_param,
    mesh_xyz,
    face,
    pose_uv,
    pose_xyz,
    obj_mesh_xyz,
    obj_face,
    corners_uv,
    corners_xyz,
    file_name,
    padding=0,
    ret=False,
    renderer=None,
):
    frame = image.copy()
    frame1 = renderer(
        [mesh_xyz, obj_mesh_xyz],
        [face, obj_face],
        cam_param,
        img=frame,
        vertex_color=[np.array([102 / 255, 209 / 255, 243 / 255]),
                      np.array([255 / 255, 163 / 255, 172 / 255])],
    )
    rend_img_overlay = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

    skeleton_overlay = draw_2d_skeleton(image, joints_uv=pose_uv, corners_uv=corners_uv)
    skeleton_3d = draw_3d_skeleton(image.shape[:2], joints_xyz=pose_xyz, corners_xyz=corners_xyz)
    mesh_3d = draw_3d_mesh_mayavi(image.shape[:2],
                                  hand_xyz=mesh_xyz,
                                  hand_face=face,
                                  obj_xyz=obj_mesh_xyz,
                                  obj_face=obj_face)

    img_list = [skeleton_overlay, rend_img_overlay, mesh_3d, skeleton_3d]
    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    grid_image = np.zeros(((image_height + padding), num_column * (image_width + padding), 3), dtype=np.uint8)

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width + padding
        width_end = width_begin + image_width
    if ret:
        return grid_image

    cv2.imwrite(file_name, grid_image)
