import math
import torch

from .transform import rotmat_to_rot6d


def _pose_work_dtype(*tensors):
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _reshape_points(points):
    if points.shape[-1] != 3:
        raise ValueError(f"Expected (..., K, 3) points, got shape {tuple(points.shape)}")
    prefix_shape = points.shape[:-2]
    num_points = points.shape[-2]
    return points.reshape(-1, num_points, 3), prefix_shape, num_points


def _broadcast_rest_points(rest_points, target_prefix_shape):
    if rest_points.dim() < 2 or rest_points.shape[-1] != 3:
        raise ValueError(f"Expected rest points shape (..., K, 3), got {tuple(rest_points.shape)}")
    target_ndim = len(target_prefix_shape) + 2
    while rest_points.dim() < target_ndim:
        rest_points = rest_points.unsqueeze(0)
    expand_shape = list(target_prefix_shape) + list(rest_points.shape[-2:])
    return rest_points.expand(*expand_shape)


def _broadcast_hand_root(hand_root, target_prefix_shape):
    if hand_root is None:
        return None
    if hand_root.shape[-1] != 3:
        raise ValueError(f"Expected hand root shape (..., 3), got {tuple(hand_root.shape)}")
    target_ndim = len(target_prefix_shape) + 1
    while hand_root.dim() < target_ndim:
        hand_root = hand_root.unsqueeze(0)
    return hand_root.expand(*target_prefix_shape, 3)


def batched_kabsch_align(src_points, dst_points, weights=None, eps=1e-6):
    if src_points.shape != dst_points.shape:
        raise ValueError(f"Shape mismatch: src={tuple(src_points.shape)} dst={tuple(dst_points.shape)}")

    work_dtype = _pose_work_dtype(src_points, dst_points, weights)
    src_points = src_points.to(dtype=work_dtype)
    dst_points = dst_points.to(dtype=work_dtype)
    if weights is not None:
        weights = weights.to(dtype=work_dtype)

    src_flat, prefix_shape, num_points = _reshape_points(src_points)
    dst_flat, _, _ = _reshape_points(dst_points)

    if weights is None:
        weights_flat = torch.ones(src_flat.shape[:-1], device=src_flat.device, dtype=src_flat.dtype)
    else:
        if weights.shape != src_points.shape[:-1]:
            raise ValueError(f"Weight shape mismatch: weights={tuple(weights.shape)} points={tuple(src_points.shape)}")
        weights_flat = weights.reshape(-1, num_points)

    weights_flat = torch.nan_to_num(weights_flat, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    weights_sum = weights_flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    weights_norm = weights_flat / weights_sum
    weights_exp = weights_norm.unsqueeze(-1)

    src_centroid = (src_flat * weights_exp).sum(dim=1, keepdim=True)
    dst_centroid = (dst_flat * weights_exp).sum(dim=1, keepdim=True)

    src_centered = src_flat - src_centroid
    dst_centered = dst_flat - dst_centroid
    cov = torch.matmul((src_centered * weights_exp).transpose(1, 2), dst_centered)

    u, _, vh = torch.linalg.svd(cov)
    v = vh.transpose(-1, -2)
    det = torch.det(torch.matmul(v, u.transpose(-1, -2)))
    sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    eye = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0).expand(cov.shape[0], -1, -1).clone()
    eye[:, -1, -1] = sign
    rotmat = torch.matmul(torch.matmul(v, eye), u.transpose(-1, -2))

    trans = dst_centroid.squeeze(1) - torch.matmul(rotmat, src_centroid.squeeze(1).unsqueeze(-1)).squeeze(-1)

    aligned = torch.matmul(src_flat, rotmat.transpose(-1, -2)) + trans.unsqueeze(1)
    fit_error = torch.sqrt(((aligned - dst_flat) ** 2).sum(dim=-1) + eps)
    fit_error = (fit_error * weights_norm).sum(dim=-1)

    rotmat = rotmat.reshape(*prefix_shape, 3, 3)
    trans = trans.reshape(*prefix_shape, 3)
    fit_error = fit_error.reshape(*prefix_shape)
    return rotmat, trans, fit_error


def pose_from_keypoints(rest_points, pred_points, hand_root=None, weights=None, eps=1e-6):
    target_prefix_shape = pred_points.shape[:-2]
    rest_points = _broadcast_rest_points(rest_points, target_prefix_shape)
    hand_root = _broadcast_hand_root(hand_root, target_prefix_shape)
    rotmat, trans_abs, fit_error = batched_kabsch_align(rest_points, pred_points, weights=weights, eps=eps)
    rot6d = rotmat_to_rot6d(rotmat.reshape(-1, 3, 3)).view(*target_prefix_shape, 6)
    if hand_root is None:
        trans_rel = trans_abs
    else:
        trans_rel = trans_abs - hand_root
    return {
        "rotmat": rotmat,
        "rot6d": rot6d,
        "trans_abs": trans_abs,
        "trans_rel": trans_rel,
        "fit_error": fit_error,
    }
