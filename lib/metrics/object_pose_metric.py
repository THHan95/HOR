import json
import math
import os
from typing import Dict

import numpy as np

import torch

from ..utils.transform import rot6d_to_rotmat
from .basic_metric import AverageMeter, Metric


class ObjectPoseMetric(Metric):

    def __init__(self, cfg, name="Obj") -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.rot_deg = AverageMeter(f"{name}_rot_deg")
        self.trans_epe = AverageMeter(f"{name}_trans_epe")
        self.add = AverageMeter(f"{name}_add")
        self.adds = AverageMeter(f"{name}_adds")
        self.mssd = AverageMeter(f"{name}_mssd")
        self.max_sym_disc_step = float(getattr(getattr(cfg, "LOSS", {}), "get", lambda *_: 0.01)("OBJ_SYM_MAX_DISC_STEP", 0.01))
        self.model_info_path = self._resolve_model_info_path(cfg)
        self._mssd_enabled = False
        self._symmetry_R = []
        self._symmetry_t = []
        self._init_mssd()
        self.reset()

    def reset(self):
        self.count = 0
        self.rot_deg.reset()
        self.trans_epe.reset()
        self.add.reset()
        self.adds.reset()
        self.mssd.reset()

    @staticmethod
    def _apply_pose(points, rot6d, trans):
        rotmat = rot6d_to_rotmat(rot6d)  # (B, 3, 3)
        posed = torch.matmul(rotmat, points.transpose(1, 2)).transpose(1, 2)
        posed = posed + trans.unsqueeze(1)
        return posed

    @staticmethod
    def _rotation_matrix_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=np.float32)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.eye(3, dtype=np.float32)
        axis = axis / axis_norm
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        one_c = 1.0 - c
        return np.array([
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ], dtype=np.float32)

    @staticmethod
    def _cfg_get(cfg_node, key, default=None):
        if cfg_node is None:
            return default
        if hasattr(cfg_node, "get"):
            return cfg_node.get(key, default)
        return getattr(cfg_node, key, default)

    def _resolve_model_info_path(self, cfg):
        loss_cfg = getattr(cfg, "LOSS", None)
        explicit = self._cfg_get(loss_cfg, "MODEL_INFO_PATH", None)
        if explicit:
            return explicit

        dataset_cfg = getattr(cfg, "DATASET", None)
        data_roots = []
        if dataset_cfg is not None:
            for split_name in ["TRAIN", "TEST"]:
                split_cfg = getattr(dataset_cfg, split_name, None)
                split_root = self._cfg_get(split_cfg, "DATA_ROOT", None)
                if split_root:
                    data_roots.append(split_root)
        for data_root in data_roots:
            candidate = os.path.join(data_root, "DexYCB", "bop", "models_eval", "models_info.json")
            if os.path.isfile(candidate):
                return candidate
        return None

    def _get_symmetry_transformations(self, model_info):
        transforms_disc = [{"R": np.eye(3, dtype=np.float32), "t": np.zeros((3, 1), dtype=np.float32)}]
        if "symmetries_discrete" in model_info:
            for sym in model_info["symmetries_discrete"]:
                sym_4x4 = np.asarray(sym, dtype=np.float32).reshape(4, 4)
                transforms_disc.append({
                    "R": sym_4x4[:3, :3],
                    "t": sym_4x4[:3, 3:].reshape(3, 1) / 1000.0,
                })

        transforms_cont = []
        if "symmetries_continuous" in model_info:
            for sym in model_info["symmetries_continuous"]:
                axis = np.asarray(sym["axis"], dtype=np.float32)
                offset = np.asarray(sym["offset"], dtype=np.float32).reshape(3, 1) / 1000.0
                discrete_steps_count = max(int(np.ceil(np.pi / max(self.max_sym_disc_step, 1e-6))), 1)
                discrete_step = 2.0 * np.pi / discrete_steps_count
                for idx in range(1, discrete_steps_count):
                    rot = self._rotation_matrix_from_axis_angle(axis, idx * discrete_step)
                    trans = (-rot.dot(offset) + offset).astype(np.float32)
                    transforms_cont.append({"R": rot, "t": trans})

        transforms = []
        for disc in transforms_disc:
            if transforms_cont:
                for cont in transforms_cont:
                    transforms.append({
                        "R": cont["R"].dot(disc["R"]).astype(np.float32),
                        "t": (cont["R"].dot(disc["t"]) + cont["t"]).astype(np.float32),
                    })
            else:
                transforms.append(disc)
        return transforms

    def _init_mssd(self):
        if self.model_info_path is None or not os.path.isfile(self.model_info_path):
            return

        with open(self.model_info_path, "r") as f_info:
            model_info = json.load(f_info)

        max_obj_id = max(int(obj_id) for obj_id in model_info.keys())
        self._symmetry_R = [torch.eye(3, dtype=torch.float32).unsqueeze(0) for _ in range(max_obj_id + 1)]
        self._symmetry_t = [torch.zeros(1, 3, 1, dtype=torch.float32) for _ in range(max_obj_id + 1)]
        for obj_id_str, obj_meta in model_info.items():
            obj_id = int(obj_id_str)
            transforms = self._get_symmetry_transformations(obj_meta)
            obj_R = np.stack([trans["R"] for trans in transforms], axis=0)
            obj_t = np.stack([trans["t"] for trans in transforms], axis=0)
            self._symmetry_R[obj_id] = torch.as_tensor(obj_R)
            self._symmetry_t[obj_id] = torch.as_tensor(obj_t)
        self._mssd_enabled = True

    @staticmethod
    def _extract_obj_ids(obj_ids, batch_size, device):
        if obj_ids is None:
            return None
        if not torch.is_tensor(obj_ids):
            obj_ids = torch.as_tensor(obj_ids, device=device)
        obj_ids = obj_ids.to(device=device, dtype=torch.long)
        while obj_ids.dim() > 1:
            obj_ids = obj_ids[..., 0]
        if obj_ids.numel() != batch_size:
            return None
        return obj_ids

    def _compute_mssd(self, pred_rot6d, pred_trans, gt_rot6d, gt_trans, obj_points_surface_rest, obj_ids):
        if not self._mssd_enabled or obj_points_surface_rest is None:
            return None

        obj_ids = self._extract_obj_ids(obj_ids, pred_rot6d.shape[0], pred_rot6d.device)
        if obj_ids is None:
            return None

        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        mssd_chunks = []
        for obj_id in obj_ids.unique(sorted=True).tolist():
            mask = obj_ids == obj_id
            if obj_id >= len(self._symmetry_R):
                continue
            sym_R = self._symmetry_R[obj_id].to(device=pred_rot6d.device, dtype=pred_rot6d.dtype)
            sym_t = self._symmetry_t[obj_id].to(device=pred_rot6d.device, dtype=pred_rot6d.dtype)
            can = obj_points_surface_rest[mask]
            if can.numel() == 0:
                continue

            sym_can = (torch.einsum("kmn,bvn->bkmv", sym_R, can) + sym_t[None, :]).transpose(-2, -1)
            gt_sym = (
                torch.einsum("bij,bklj->bkil", gt_rotmat[mask], sym_can) +
                gt_trans[mask][:, None, :, None]
            ).transpose(-2, -1)
            pred_pts = (
                torch.matmul(pred_rotmat[mask], can.transpose(1, 2)).transpose(1, 2) +
                pred_trans[mask].unsqueeze(1)
            )
            mssd_value = torch.norm(gt_sym - pred_pts.unsqueeze(1), dim=-1).max(dim=-1)[0].min(dim=-1)[0]
            mssd_chunks.append(mssd_value)

        if not mssd_chunks:
            return None
        return torch.cat(mssd_chunks, dim=0)

    def feed(self,
             pred_rot6d,
             pred_trans,
             gt_rot6d,
             gt_trans,
             obj_points_rest,
             obj_points_surface_rest=None,
             obj_ids=None,
             **kwargs):
        pred_rot6d = pred_rot6d.detach()
        pred_trans = pred_trans.detach()
        gt_rot6d = gt_rot6d.detach()
        gt_trans = gt_trans.detach()
        obj_points_rest = obj_points_rest.detach()
        if obj_points_surface_rest is not None:
            obj_points_surface_rest = obj_points_surface_rest.detach()
        batch_size = pred_rot6d.shape[0]

        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        rot_deg = torch.rad2deg(torch.acos(cos_theta))

        trans_epe = torch.norm(pred_trans - gt_trans, dim=-1)

        pred_points = self._apply_pose(obj_points_rest, pred_rot6d, pred_trans)
        gt_points = self._apply_pose(obj_points_rest, gt_rot6d, gt_trans)
        add = torch.norm(pred_points - gt_points, dim=-1).mean(dim=-1)

        pairwise = torch.cdist(pred_points.float(), gt_points.float())
        adds = pairwise.min(dim=-1)[0].mean(dim=-1)
        mssd = self._compute_mssd(
            pred_rot6d=pred_rot6d,
            pred_trans=pred_trans,
            gt_rot6d=gt_rot6d,
            gt_trans=gt_trans,
            obj_points_surface_rest=obj_points_surface_rest,
            obj_ids=obj_ids,
        )

        self.rot_deg.update(rot_deg.sum().item(), batch_size)
        self.trans_epe.update(trans_epe.sum().item(), batch_size)
        self.add.update(add.sum().item(), batch_size)
        self.adds.update(adds.sum().item(), batch_size)
        if mssd is not None:
            self.mssd.update(mssd.sum().item(), mssd.numel())
        self.count += batch_size

    def get_measures(self, **kwargs) -> Dict[str, float]:
        return {
            f"{self.name}_rot_deg": self.rot_deg.avg,
            f"{self.name}_trans_epe": self.trans_epe.avg,
            f"{self.name}_add": self.add.avg,
            f"{self.name}_adds": self.adds.avg,
            f"{self.name}_mssd": self.mssd.avg,
        }

    def get_result(self):
        return self.adds.avg

    def __str__(self) -> str:
        return (
            f"{self.name} MSSD {self.mssd.avg * 1000.0:.1f} mm | "
            f"ADD-S {self.adds.avg * 1000.0:.1f} mm | "
            f"ADD {self.add.avg * 1000.0:.1f} mm | "
            f"T {self.trans_epe.avg * 1000.0:.1f} mm"
        )
