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


def draw_batch_mesh_images_pred(gt_verts2d, pred_verts2d, face, gt_obj2d, pred_obj2d, gt_objc2d, pred_objc2d, intr, tensor_image, n_sample=16):
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
    gt_objc2d = gt_objc2d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 2)
    pred_objc2d = pred_objc2d[:n_sample, ...].detach().cpu().numpy()  # (B, 1, 2)
    intr = intr[:n_sample, ...].detach().cpu().numpy()  # (B, 3, 3)

    sample_list = []
    for i in range(n_sample):
        gt_verts2d_i = gt_verts2d[i].copy()
        gt_obj2d_i = gt_obj2d[i].copy()
        gt_objc2d_i = gt_objc2d[i].copy()
        pred_verts2d_i = pred_verts2d[i].copy()
        pred_obj2d_i = pred_obj2d[i].copy()
        pred_objc2d_i = pred_objc2d[i].copy()

        intr_i = intr[i].copy()

        gt_mesh_img = draw_hand_obj_mesh(image[i].copy(), gt_verts2d_i, face, gt_obj2d_i, gt_objc2d_i)
        pred_mesh_img = draw_hand_obj_mesh(image[i].copy(), pred_verts2d_i, face, pred_obj2d_i, pred_objc2d_i)

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
    plt.plot(obj_uv[:, 0], obj_uv[:, 1], 'o', color='red', markersize=0.5)
    if face is None:
        plt.plot(verts_uv[:, 0], verts_uv[:, 1], 'o', color='green', markersize=1) 
    else:
        plt.triplot(verts_uv[:, 0], verts_uv[:, 1], face, lw=0.5, color='orange')
    
    if obj_center_uv is not None:
        plt.plot(obj_center_uv[:, 0], obj_center_uv[:, 1], 'o', color='blue', markersize=3)

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
