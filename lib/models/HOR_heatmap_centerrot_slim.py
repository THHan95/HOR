import random
from itertools import combinations, product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import CONST
from ..utils.transform import batch_cam_extr_transf, batch_cam_intr_projection, rot6d_to_rotmat, rotmat_to_rot6d
from ..utils.triangulation import batch_triangulate_dlt_torch_confidence
from ..viztools.draw import (
    draw_batch_hand_mesh_images_2d,
    draw_batch_joint_confidence_images,
    draw_batch_joint_images,
    draw_batch_mesh_images_pred,
    tile_batch_images,
)
from .HOR_heatmap_centerrot import POEM_HeatmapCenterRot


def _sample_view_vectors(n_virtual_views=20, device=None, dtype=torch.float32):
    cam_vec = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
    theta = torch.rand(n_virtual_views, device=device, dtype=dtype) * 2.0 * np.pi
    u = torch.rand(n_virtual_views, device=device, dtype=dtype)
    nv_x = torch.sqrt(1.0 - u ** 2) * torch.cos(theta)
    nv_y = torch.sqrt(1.0 - u ** 2) * torch.sin(theta)
    nv_z = u
    nv = torch.stack([nv_x, nv_y, nv_z], dim=1)
    return torch.cat([cam_vec, nv], dim=0)


def _jointlevel_ordinal_relation(jpair: torch.Tensor, view_vecs: torch.Tensor):
    nviews = view_vecs.shape[1]
    npairs = jpair.shape[1]
    jpair = jpair.unsqueeze(2).expand(-1, -1, nviews, -1)
    view_vecs = view_vecs.unsqueeze(1).expand(-1, npairs, -1, -1)
    jpair_diff = jpair[..., :3] - jpair[..., 3:]
    jpair_ord = torch.einsum("bijk,bijk->bij", jpair_diff, view_vecs)
    return jpair_ord.unsqueeze(-1)


def _partlevel_ordinal_relation(ppair: torch.Tensor, view_vecs: torch.Tensor):
    nviews = view_vecs.shape[1]
    npairs = ppair.shape[1]
    ppair = ppair.unsqueeze(2).expand(-1, -1, nviews, -1)
    view_vecs = view_vecs.unsqueeze(1).expand(-1, npairs, -1, -1)
    ppair_cross = torch.cross(ppair[..., :3], ppair[..., 3:], dim=-1)
    ppair_ord = torch.einsum("bijk,bijk->bij", ppair_cross, view_vecs)
    return ppair_ord.unsqueeze(-1)


@MODEL.register_module()
class POEM_HeatmapCenterRotSlim(POEM_HeatmapCenterRot):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.direct_hidden_dim = int(cfg.get("DIRECT_HIDDEN_DIM", self.feat_size[0]))
        self.slim_hand_feat = nn.Sequential(
            nn.Linear(self.feat_size[0], self.direct_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.direct_hidden_dim, self.direct_hidden_dim),
            nn.ReLU(),
        )
        self.slim_hand_pose_head = nn.Linear(self.direct_hidden_dim, 16 * 6)
        self.slim_hand_shape_head = nn.Linear(self.direct_hidden_dim, 10)
        self.slim_hand_cam_head = nn.Linear(self.direct_hidden_dim, 3)
        self.slim_obj_feat = nn.Sequential(
            nn.Linear(self.feat_size[0], self.direct_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.direct_hidden_dim, self.direct_hidden_dim),
            nn.ReLU(),
        )
        self.slim_obj_center_head = nn.Linear(self.direct_hidden_dim, 3)  # conf/scale, tx, ty
        self.slim_obj_rot_head = nn.Linear(self.direct_hidden_dim, 6)
        self.arti_joints_3d_weight = float(cfg.LOSS.get("ARTI_JOINTS_3D_N", 1.0))
        self.arti_center_2d_weight = float(cfg.LOSS.get("ARTI_CENTER_2D_N", 1.0))
        self.arti_center_3d_weight = float(cfg.LOSS.get("ARTI_CENTER_3D_N", 1.0))
        self.arti_corners_3d_weight = float(cfg.LOSS.get("ARTI_CORNERS_3D_N", 0.2))
        self.arti_points_3d_weight = float(cfg.LOSS.get("ARTI_POINTS_3D_N", 0.0))
        self.arti_sym_corners_3d_weight = float(cfg.LOSS.get("ARTI_SYM_CORNERS_3D_N", 0.0))
        self.arti_hand_ord_weight = float(cfg.LOSS.get("ARTI_HAND_ORD_N", 0.0))
        self.arti_scene_ord_weight = float(cfg.LOSS.get("ARTI_SCENE_ORD_N", 0.0))
        self.arti_n_virtual_views = int(cfg.LOSS.get("ARTI_N_VIRTUAL_VIEWS", 20))
        self._arti_joint_pairs_idx = list(combinations(range(self.num_hand_joints), 2))
        self._arti_part_pairs_idx = list(combinations(range(self.num_hand_joints - 1), 2))
        self._arti_ho_pairs_idx = list(product(range(self.num_hand_joints), range(8)))

    @staticmethod
    def _pixels_to_norm(coords_px, image_hw):
        h, w = image_hw
        scale = coords_px.new_tensor([w, h]).view(1, 1, 2)
        return coords_px / scale * 2.0 - 1.0

    @staticmethod
    def _norm_to_pixels(coords_norm, image_hw):
        h, w = image_hw
        scale = coords_norm.new_tensor([w, h]).view(1, 1, 2)
        return (coords_norm + 1.0) * 0.5 * scale

    @staticmethod
    def _apply_rotation_only(batch_cam_extr, batch_points):
        rot = batch_cam_extr[..., :3, :3]
        return torch.einsum("...ij,...kj->...ki", rot, batch_points)

    def _gather_master_view(self, tensor, master_ids):
        batch_size = tensor.shape[0]
        index_shape = [batch_size, 1] + [1] * (tensor.dim() - 2)
        gather_index = master_ids.view(*index_shape).expand(-1, 1, *tensor.shape[2:])
        return torch.gather(tensor, 1, gather_index).squeeze(1)

    def _build_direct_forward_preds(self, batch):
        img = batch["image"]
        batch_size, num_cams = img.shape[:2]
        H, W = img.shape[-2:]
        _, _, global_feat = self.extract_img_feat(img)
        global_feat = global_feat.view(batch_size, num_cams, -1)
        global_feat_flat = global_feat.reshape(batch_size * num_cams, -1)

        hand_feat = self.slim_hand_feat(global_feat_flat)
        pred_hand_pose6d = self.slim_hand_pose_head(hand_feat)
        pred_hand_shape = self.slim_hand_shape_head(hand_feat)
        pred_hand_cam = self.slim_hand_cam_head(hand_feat)
        pred_hand_cam = torch.cat(
            (
                F.softplus(pred_hand_cam[:, :1]) + 1e-4,
                pred_hand_cam[:, 1:],
            ),
            dim=1,
        )

        coord_xyz_sv, coord_uv_sv, pose_euler_sv, shape_sv, cam_sv = self.mano_decoder(
            pred_hand_pose6d,
            pred_hand_shape,
            pred_hand_cam,
        )
        coord_xyz_sv = coord_xyz_sv * (self.data_preset_cfg.BBOX_3D_SIZE / 2)
        pred_mano_3d_mesh_sv = coord_xyz_sv
        pred_mano_2d_mesh_sv = coord_uv_sv
        pred_hand_pixel = self._norm_to_pixels(
            pred_mano_2d_mesh_sv[:, self.num_hand_verts:],
            (H, W),
        )
        pred_hand_norm = pred_mano_2d_mesh_sv[:, self.num_hand_verts:]

        pred_hand_root_pixel = pred_hand_pixel[:, self.center_idx:self.center_idx + 1].view(batch_size, num_cams, 1, 2)
        unit_conf = torch.ones(
            batch_size,
            num_cams,
            1,
            device=img.device,
            dtype=img.dtype,
        )
        T_m2c = torch.linalg.inv(batch["target_cam_extr"]).to(device=img.device, dtype=img.dtype)
        pred_root_master = batch_triangulate_dlt_torch_confidence(
            pred_hand_root_pixel,
            batch["target_cam_intr"].to(device=img.device, dtype=img.dtype),
            T_m2c,
            confidences=unit_conf,
        ).squeeze(1)

        pred_mano_3d_mesh_sv_view = pred_mano_3d_mesh_sv.view(batch_size, num_cams, self.num_hand_verts + self.num_hand_joints, 3)
        rel_hand_master_views = self._apply_rotation_only(
            batch["target_cam_extr"].to(device=img.device, dtype=img.dtype),
            pred_mano_3d_mesh_sv_view,
        )
        hand_mesh_master_views = rel_hand_master_views + pred_root_master[:, None, None, :]
        pred_mano_3d_mesh_master = hand_mesh_master_views.mean(dim=1)
        pred_ref_hand = pred_mano_3d_mesh_master[:, self.num_hand_verts:]

        obj_feat = self.slim_obj_feat(global_feat_flat)
        pred_obj_center_param = self.slim_obj_center_head(obj_feat)
        pred_obj_center_conf = torch.sigmoid(pred_obj_center_param[:, :1]).view(batch_size, num_cams, self.num_obj_joints)
        pred_obj_center_norm = torch.tanh(pred_obj_center_param[:, 1:]).view(batch_size * num_cams, self.num_obj_joints, 2)
        pred_obj_center_pixel = self._norm_to_pixels(pred_obj_center_norm, (H, W)).view(batch_size, num_cams, self.num_obj_joints, 2)
        pred_obj_rot6d_cam = self.slim_obj_rot_head(obj_feat).view(batch_size, num_cams, 6)

        pred_center_master = batch_triangulate_dlt_torch_confidence(
            pred_obj_center_pixel,
            batch["target_cam_intr"].to(device=img.device, dtype=img.dtype),
            T_m2c,
            confidences=pred_obj_center_conf,
        )
        pred_center_views = batch_cam_extr_transf(
            T_m2c,
            pred_center_master.unsqueeze(1).expand(-1, num_cams, -1, -1),
        )

        pred_hand_root_views = batch_cam_extr_transf(
            T_m2c,
            pred_root_master[:, None, None, :].expand(-1, num_cams, 1, -1),
        )
        pred_obj_view_trans = pred_center_views - pred_hand_root_views

        pred_obj_rotmat_cam = rot6d_to_rotmat(pred_obj_rot6d_cam.reshape(-1, 6)).view(batch_size, num_cams, 3, 3)
        pred_obj_rotmat_master_views = torch.matmul(
            batch["target_cam_extr"].to(device=img.device, dtype=img.dtype)[:, :, :3, :3],
            pred_obj_rotmat_cam,
        )
        master_ids = batch["master_id"].to(device=img.device, dtype=torch.long)
        pred_obj_rotmat_master = self._gather_master_view(pred_obj_rotmat_master_views, master_ids)
        pred_obj_rot6d_master = rotmat_to_rot6d(pred_obj_rotmat_master)

        pred_obj_trans_master = pred_center_master.squeeze(1) - pred_ref_hand[:, self.center_idx]

        master_obj_sparse_rest = batch["master_obj_sparse_rest"].to(device=img.device, dtype=img.dtype)
        pred_obj_sparse_master = self._build_object_points_from_pose_grad(
            master_obj_sparse_rest,
            pred_obj_rot6d_master,
            pred_center_master.squeeze(1),
        )
        pred_obj_sparse_master_states = pred_obj_sparse_master.unsqueeze(0)
        pred_hand_master_states = pred_mano_3d_mesh_master.unsqueeze(0)
        pred_obj_rot6d_states = pred_obj_rot6d_master.unsqueeze(0)
        pred_obj_trans_states = pred_obj_trans_master.unsqueeze(0)

        pred_obj_center_norm_flat = pred_obj_center_norm
        pred_uv_conf = torch.ones(
            batch_size * num_cams,
            self.num_hand_joints + self.num_obj_joints,
            2,
            device=img.device,
            dtype=img.dtype,
        )
        pred_uv_conf[:, self.num_hand_joints:] = pred_obj_center_conf.reshape(batch_size * num_cams, self.num_obj_joints, 1).expand(-1, -1, 2)
        pred_uv_heatmap = torch.zeros(
            batch_size * num_cams,
            self.num_hand_joints + self.num_obj_joints,
            self.data_preset_cfg.HEATMAP_SIZE[1],
            self.data_preset_cfg.HEATMAP_SIZE[0],
            device=img.device,
            dtype=img.dtype,
        )

        return {
            "pred_hand": pred_hand_norm,
            "pred_obj": pred_obj_center_norm_flat,
            "pred_hand_pixel": pred_hand_pixel,
            "pred_obj_pixel": pred_obj_center_pixel.view(batch_size * num_cams, self.num_obj_joints, 2),
            "pred_uv_heatmap": pred_uv_heatmap,
            "pred_uv_conf_base": pred_uv_conf,
            "pred_uv_conf": pred_uv_conf,
            "conf_hand": torch.ones(batch_size * num_cams, self.num_hand_joints, 2, device=img.device, dtype=img.dtype),
            "mano_3d_mesh_sv": pred_mano_3d_mesh_sv,
            "mano_2d_mesh_sv": pred_mano_2d_mesh_sv,
            "mano_pose_euler_sv": pose_euler_sv,
            "mano_shape_sv": shape_sv,
            "mano_cam_sv": cam_sv,
            "ref_hand": pred_ref_hand,
            "ref_obj_center": pred_center_master.squeeze(1),
            "hand_mesh_xyz_master": pred_hand_master_states,
            "obj_xyz_master": pred_obj_sparse_master_states,
            "obj_rot6d": pred_obj_rot6d_states,
            "obj_trans": pred_obj_trans_states,
            "obj_view_rot6d_cam": pred_obj_rot6d_cam,
            "obj_view_trans": pred_obj_view_trans.squeeze(-2),
            "obj_view_trans_master": pred_center_master.unsqueeze(1).expand(-1, num_cams, -1, -1).squeeze(-2),
            "obj_init_rot6d": pred_obj_rot6d_master,
            "obj_init_trans": pred_obj_trans_master,
            "obj_fused_rot6d_sv": pred_obj_rot6d_cam,
            "obj_fused_trans_sv": pred_obj_view_trans.squeeze(-2),
            "mano_3d_mesh_master": pred_mano_3d_mesh_master,
            "mano_3d_mesh_kp_master": pred_mano_3d_mesh_master,
            "mano_3d_mesh_mesh_master": None,
            "pred_pose": pose_euler_sv.view(batch_size, num_cams, -1).mean(dim=1),
            "pred_shape": shape_sv.view(batch_size, num_cams, -1).mean(dim=1),
            "pred_kp_pose": pose_euler_sv.view(batch_size, num_cams, -1).mean(dim=1),
            "pred_kp_shape": shape_sv.view(batch_size, num_cams, -1).mean(dim=1),
            "pred_mesh_pose": None,
            "pred_mesh_shape": None,
            "pred_obj_trans_master": pred_center_master.squeeze(1),
            "interaction_mode": "slim_direct",
            "all_hand_joints_xyz_master": pred_ref_hand.unsqueeze(0),
        }

    def _forward_impl(self, batch, **kwargs):
        return self._build_direct_forward_preds(batch)

    def _masked_mse(self, pred, gt, valid_mask):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=10.0, neginf=-10.0)
        valid = valid_mask.to(dtype=pred.dtype)
        diff = (pred - gt) ** 2
        diff = torch.einsum("...c,...->...c", diff, valid)
        denom = valid.sum() * pred.shape[-1]
        return diff.sum() / (denom + 1e-9)

    def _masked_l1(self, pred, gt, valid_mask):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=10.0, neginf=-10.0)
        valid = valid_mask.to(dtype=pred.dtype)
        diff = torch.abs(pred - gt)
        diff = torch.einsum("...c,...->...c", diff, valid)
        denom = valid.sum() * pred.shape[-1]
        return diff.sum() / (denom + 1e-9)

    def _masked_epe(self, pred, gt, valid_mask):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=10.0, neginf=-10.0)
        valid = valid_mask.to(dtype=pred.dtype)
        diff = torch.linalg.norm(pred - gt, dim=-1)
        denom = valid.sum()
        return (diff * valid).sum() / (denom + 1e-9)

    def _joints_to_joint_pairs(self, joints: torch.Tensor):
        pairs_idx = np.asarray(self._arti_joint_pairs_idx)
        idx1 = pairs_idx[:, 0]
        idx2 = pairs_idx[:, 1]
        return torch.cat([joints[:, idx1, :], joints[:, idx2, :]], dim=2)

    def _joints_to_part_pairs(self, joints: torch.Tensor):
        child_idx = list(range(self.num_hand_joints))
        parent_idx = CONST.JOINTS_IDX_PARENTS
        parts = (joints[:, child_idx, :] - joints[:, parent_idx, :])[:, 1:, :]
        pairs_idx = np.asarray(self._arti_part_pairs_idx)
        idx1 = pairs_idx[:, 0]
        idx2 = pairs_idx[:, 1]
        return torch.cat([parts[:, idx1, :], parts[:, idx2, :]], dim=2)

    def _joints_and_corners_to_pairs(self, joints: torch.Tensor, corners: torch.Tensor):
        pairs_idx = np.asarray(self._arti_ho_pairs_idx)
        idx1 = pairs_idx[:, 0]
        idx2 = pairs_idx[:, 1]
        return torch.cat([joints[:, idx1, :], corners[:, idx2, :]], dim=2)

    def compute_loss(self, preds, gt, stage_name="stage2", epoch_idx=None, **kwargs):
        zero = preds["mano_3d_mesh_sv"].sum() * 0.0
        obj_warmup = self._stage2_object_warmup(epoch_idx, stage_name) if epoch_idx is not None else 1.0

        loss_dict = {
            "loss_2d_sv": zero,
            "loss_heatmap_hand": zero,
            "loss_heatmap_hand_map": zero,
            "loss_heatmap_obj": zero,
            "loss_heatmap_obj_map": zero,
            "loss_triang_hand": zero,
            "loss_triang_obj": zero,
            "loss_triang": zero,
            "loss_obj_init_trans": zero,
            "loss_obj_init_rot": zero,
            "loss_obj_init_rot_geo": zero,
            "loss_obj_init_rot_l1": zero,
            "loss_obj_view_rot": zero,
            "loss_obj_view_rot_geo": zero,
            "loss_obj_view_rot_l1": zero,
            "loss_obj_view_trans": zero,
            "loss_obj_view_points": zero,
            "loss_obj_view_rot_self_geo": zero,
            "loss_obj_view_rot_self_l1": zero,
            "loss_obj_chamfer_stage1": zero,
            "loss_obj_emd_stage1": zero,
            "loss_obj_sym_corner_stage1": zero,
            "loss_obj_corner_proj_stage1": zero,
            "loss_obj_corner_3d_stage1": zero,
            "loss_mano_consist": zero,
            "loss_mano_consist_kp": zero,
            "loss_mano_consist_mesh": zero,
            "loss_pose_reg_sv": zero,
            "loss_pose_reg_master": zero,
            "loss_pose_reg_kp_master": zero,
            "loss_pose_reg_mesh_master": zero,
            "loss_pose_reg": zero,
            "loss_shape_reg_sv": zero,
            "loss_shape_reg_master": zero,
            "loss_shape_reg_kp_master": zero,
            "loss_shape_reg_mesh_master": zero,
            "loss_shape_reg": zero,
            "loss_mano_proj": zero,
            "loss_mano_proj_kp": zero,
            "loss_mano_proj_mesh": zero,
            "loss_3d_jts": zero,
            "loss_2d_proj": zero,
            "loss_obj_rot": zero,
            "loss_obj_rot_geo": zero,
            "loss_obj_rot_l1_aux": zero,
            "loss_obj_trans": zero,
            "loss_obj_points": zero,
            "loss_obj_corner_proj": zero,
            "loss_obj_corner_3d": zero,
            "loss_obj_pose": zero,
            "loss_obj_chamfer": zero,
            "loss_obj_emd": zero,
            "loss_obj_corner_proj_total": zero,
            "loss_obj_corner_3d_total": zero,
            "loss_obj_sym_corner": zero,
            "loss_penetration": zero,
            "loss_obj_warmup": preds["mano_3d_mesh_sv"].new_tensor(float(obj_warmup)),
            "loss_recon": zero,
        }
        for i in range(preds["hand_mesh_xyz_master"].shape[0]):
            loss_dict[f"dec{i}_recon"] = zero
        loss_dict.update({
            "loss_arti_center_2d": zero,
            "loss_arti_center_3d": zero,
            "loss_arti_joints_3d": zero,
            "loss_arti_corners_3d": zero,
            "loss_arti_points_3d": zero,
            "loss_arti_sym_corners_3d": zero,
            "loss_arti_hand_ord": zero,
            "loss_arti_scene_ord": zero,
            "metric_obj_center_2d_epe_px": zero,
            "metric_obj_center_2d_mse_px2": zero,
            "metric_obj_center_3d_l1_mm": zero,
            "metric_obj_center_3d_epe_mm": zero,
            "metric_obj_center_3d_mse_m2": zero,
            "metric_obj_corner_3d_l1_mm": zero,
            "metric_obj_corner_3d_epe_mm": zero,
            "metric_obj_corner_3d_mse_m2": zero,
            "metric_obj_points_3d_l1_mm": zero,
            "metric_obj_center_conf_mean": zero,
            "metric_obj_center_depth_mm": zero,
        })

        if stage_name != "stage1":
            loss_dict["loss"] = zero
            return zero, loss_dict

        gt_T_c2m = torch.linalg.inv(gt["target_cam_extr"])
        target_joints_3d = gt.get("target_joints_3d", None)
        target_obj_kp21 = gt.get("target_obj_kp21", None)
        target_obj_pc_sparse = gt.get("target_obj_pc_sparse", None)
        target_rot6d = gt.get("target_rot6d_label", None)
        master_obj_kp21_rest = gt.get("master_obj_kp21_rest", None)
        master_obj_sparse_rest = gt.get("master_obj_sparse_rest", None)
        master_obj_kp21 = gt.get("master_obj_kp21", None)
        hand_joints_sv_gt = gt.get("target_joints_uvd", None)
        target_joints_vis = gt.get("target_joints_vis", None)
        pred_ref_hand = preds.get("ref_hand", None)
        pred_obj_view_rot6d = preds.get("obj_view_rot6d_cam", None)
        pred_obj_pixel = preds.get("pred_obj_pixel", None)
        pred_uv_conf = preds.get("pred_uv_conf", None)
        pred_mano_2d_mesh_sv = preds.get("mano_2d_mesh_sv", None)
        pred_pose_sv = preds.get("mano_pose_euler_sv", None)
        pred_shape_sv = preds.get("mano_shape_sv", None)

        if (
            pred_ref_hand is None
            or pred_obj_view_rot6d is None
            or pred_obj_pixel is None
            or target_joints_3d is None
            or target_obj_kp21 is None
            or master_obj_kp21_rest is None
            or pred_mano_2d_mesh_sv is None
            or pred_pose_sv is None
            or pred_shape_sv is None
            or hand_joints_sv_gt is None
        ):
            loss_dict["loss"] = zero
            return zero, loss_dict

        batch_size, n_views = target_joints_3d.shape[:2]
        device = pred_ref_hand.device
        dtype = pred_ref_hand.dtype
        hand_joints_sv_gt = hand_joints_sv_gt.to(device=device, dtype=dtype).flatten(0, 1)
        pred_mano_2d_joints_sv = pred_mano_2d_mesh_sv[:, self.num_hand_verts:].to(device=device, dtype=dtype)
        joints_vis = target_joints_vis.flatten(0, 1) if target_joints_vis is not None else None

        gt_jts_2d = hand_joints_sv_gt[..., :2]
        diff_2d = torch.abs(pred_mano_2d_joints_sv - gt_jts_2d)
        if joints_vis is not None:
            vis_mask = joints_vis.to(device=device, dtype=dtype).unsqueeze(-1)
            diff_2d = diff_2d * vis_mask
            valid_count = vis_mask.expand_as(diff_2d).sum()
            loss_hand_2d_sv = diff_2d.sum() / (valid_count + 1e-9)
        else:
            loss_hand_2d_sv = diff_2d.mean()

        zero_target_sv = torch.zeros_like(pred_pose_sv[:, 3:])
        loss_pose_reg_sv = self.coord_loss(pred_pose_sv[:, 3:], zero_target_sv) * self.cfg.LOSS.POSE_N
        loss_shape_reg_sv = self.coord_loss(pred_shape_sv, torch.zeros_like(pred_shape_sv)) * self.cfg.LOSS.SHAPE_N

        loss_dict["loss_2d_sv"] = loss_hand_2d_sv
        loss_dict["loss_pose_reg_sv"] = loss_pose_reg_sv
        loss_dict["loss_shape_reg_sv"] = loss_shape_reg_sv
        loss_dict["loss_pose_reg"] = loss_pose_reg_sv
        loss_dict["loss_shape_reg"] = loss_shape_reg_sv

        pred_hand_views = batch_cam_extr_transf(
            gt_T_c2m,
            pred_ref_hand.unsqueeze(1).expand(-1, n_views, -1, -1),
        )
        pred_hand_flat = pred_hand_views.reshape(batch_size * n_views, self.num_hand_joints, 3)
        gt_hand_flat = target_joints_3d.to(device=device, dtype=dtype).reshape(batch_size * n_views, self.num_hand_joints, 3)

        pred_obj_pixel = pred_obj_pixel.view(batch_size, n_views, self.num_obj_joints, 2).to(device=device, dtype=dtype)
        gt_center_views = target_obj_kp21[:, :, -self.num_obj_joints:].to(device=device, dtype=dtype)
        gt_center_pixel = batch_cam_intr_projection(
            gt["target_cam_intr"].to(device=device, dtype=dtype),
            gt_center_views,
        )
        center_valid = torch.isfinite(gt_center_views).all(dim=-1) & (gt_center_views[..., 2] > 1e-6)
        center_valid = center_valid & torch.isfinite(gt_center_pixel).all(dim=-1)

        obj_center_conf = None
        if pred_uv_conf is not None:
            obj_center_conf = pred_uv_conf.view(batch_size, n_views, self.num_hand_joints + self.num_obj_joints, -1)
            obj_center_conf = obj_center_conf[:, :, self.num_hand_joints:, :].mean(dim=-1)
            obj_center_conf = obj_center_conf * center_valid.to(dtype=dtype)

        pred_center_master = batch_triangulate_dlt_torch_confidence(
            pred_obj_pixel,
            gt["target_cam_intr"].to(device=device, dtype=dtype),
            gt_T_c2m.to(device=device, dtype=dtype),
            confidences=obj_center_conf,
        )
        pred_center_views = batch_cam_extr_transf(
            gt_T_c2m.to(device=device, dtype=dtype),
            pred_center_master.unsqueeze(1).expand(-1, n_views, -1, -1),
        )

        gt_center_master = None
        gt_center_master_valid = None
        if master_obj_kp21 is not None:
            gt_center_master = master_obj_kp21[:, -self.num_obj_joints:].to(device=device, dtype=dtype)
            gt_center_master_valid = torch.isfinite(gt_center_master).all(dim=-1) & (gt_center_master[..., 2] > 1e-6)
        else:
            master_ids = gt["master_id"].to(device=device, dtype=torch.long)
            gather_index = master_ids.view(batch_size, 1, 1, 1).expand(-1, 1, self.num_obj_joints, 3)
            gt_center_master = torch.gather(gt_center_views, 1, gather_index).squeeze(1)
            gt_center_master_valid = torch.gather(center_valid, 1, master_ids.view(batch_size, 1, 1).expand(-1, 1, self.num_obj_joints)).squeeze(1)

        corner_rest = master_obj_kp21_rest[:, :8].to(device=device, dtype=dtype)
        pred_corners_views = self._build_object_points_from_pose_grad(
            corner_rest,
            pred_obj_view_rot6d,
            pred_center_views,
        )
        gt_corners_views = target_obj_kp21[:, :, :8].to(device=device, dtype=dtype)

        joint_valid = torch.isfinite(target_joints_3d).all(dim=-1)
        if target_joints_vis is not None:
            joint_valid = joint_valid & (target_joints_vis > 0)
        corner_valid = torch.isfinite(gt_corners_views).all(dim=-1) & (gt_corners_views[..., 2] > 1e-6)

        joint_valid_flat = joint_valid.reshape(batch_size * n_views, self.num_hand_joints)
        pred_corners_flat = pred_corners_views.reshape(batch_size * n_views, 8, 3)
        gt_corners_flat = gt_corners_views.reshape(batch_size * n_views, 8, 3)
        corner_valid_flat = corner_valid.reshape(batch_size * n_views, 8)

        pred_obj_norm = self._pixels_to_norm(pred_obj_pixel, (gt["image"].shape[-2], gt["image"].shape[-1]))
        gt_center_norm = self._pixels_to_norm(gt_center_pixel, (gt["image"].shape[-2], gt["image"].shape[-1]))
        raw_obj_center_2d_l1_norm = self._masked_l1(pred_obj_norm, gt_center_norm, center_valid)
        raw_obj_center_2d_mse_px2 = self._masked_mse(pred_obj_pixel, gt_center_pixel, center_valid)
        metric_obj_center_2d_epe_px = self._masked_epe(pred_obj_pixel, gt_center_pixel, center_valid).detach()
        metric_obj_center_2d_mse_px2 = raw_obj_center_2d_mse_px2.detach()
        raw_obj_center_3d_l1 = zero
        raw_obj_center_3d_mse_m2 = zero
        metric_obj_center_3d_l1_mm = zero.detach()
        metric_obj_center_3d_mse_m2 = zero.detach()
        metric_obj_center_3d_epe_mm = zero.detach()
        if gt_center_master is not None:
            raw_obj_center_3d_l1 = self._masked_l1(
                pred_center_master,
                gt_center_master,
                gt_center_master_valid,
            )
            raw_obj_center_3d_mse_m2 = self._masked_mse(
                pred_center_master,
                gt_center_master,
                gt_center_master_valid,
            )
            metric_obj_center_3d_l1_mm = (raw_obj_center_3d_l1.detach() * 1000.0)
            metric_obj_center_3d_mse_m2 = raw_obj_center_3d_mse_m2.detach()
            metric_obj_center_3d_epe_mm = (
                self._masked_epe(
                    pred_center_master,
                    gt_center_master,
                    gt_center_master_valid,
                ).detach() * 1000.0
            )

        raw_obj_corner_3d_l1 = self._masked_l1(
            pred_corners_flat,
            gt_corners_flat,
            corner_valid_flat,
        )
        raw_obj_corner_3d_mse_m2 = self._masked_mse(
            pred_corners_flat,
            gt_corners_flat,
            corner_valid_flat,
        )
        metric_obj_corner_3d_l1_mm = (raw_obj_corner_3d_l1.detach() * 1000.0)
        metric_obj_corner_3d_mse_m2 = raw_obj_corner_3d_mse_m2.detach()
        metric_obj_corner_3d_epe_mm = (
            self._masked_epe(
                pred_corners_flat,
                gt_corners_flat,
                corner_valid_flat,
            ).detach() * 1000.0
        )
        if obj_center_conf is not None:
            metric_obj_center_conf_mean = (
                (obj_center_conf.sum() / (center_valid.to(dtype=dtype).sum() + 1e-9)).detach()
            )
        else:
            metric_obj_center_conf_mean = zero.detach()
        metric_obj_center_depth_mm = (
            pred_center_views[..., 2].mean().detach() * 1000.0
        )

        loss_arti_center_2d = zero
        if self.arti_center_2d_weight > 0.0:
            loss_arti_center_2d = raw_obj_center_2d_l1_norm * self.arti_center_2d_weight

        loss_arti_center_3d = zero
        if self.arti_center_3d_weight > 0.0 and gt_center_master is not None:
            loss_arti_center_3d = raw_obj_center_3d_l1 * self.arti_center_3d_weight

        loss_arti_joints_3d = zero

        loss_arti_corners_3d = zero
        if self.arti_corners_3d_weight > 0.0:
            loss_arti_corners_3d = raw_obj_corner_3d_l1 * self.arti_corners_3d_weight

        loss_arti_points_3d = zero
        metric_obj_points_3d_l1_mm = zero.detach()
        if self.arti_points_3d_weight > 0.0 and master_obj_sparse_rest is not None and target_obj_pc_sparse is not None:
            pred_points_views = self._build_object_points_from_pose_grad(
                master_obj_sparse_rest.to(device=device, dtype=dtype),
                pred_obj_view_rot6d,
                pred_center_views,
            )
            gt_points_views = target_obj_pc_sparse.to(device=device, dtype=dtype)
            point_valid = torch.isfinite(gt_points_views).all(dim=-1) & (gt_points_views[..., 2] > 1e-6)
            raw_obj_points_3d_l1 = self._masked_l1(
                pred_points_views.reshape(batch_size * n_views, gt_points_views.shape[-2], 3),
                gt_points_views.reshape(batch_size * n_views, gt_points_views.shape[-2], 3),
                point_valid.reshape(batch_size * n_views, gt_points_views.shape[-2]),
            )
            metric_obj_points_3d_l1_mm = (raw_obj_points_3d_l1.detach() * 1000.0)
            loss_arti_points_3d = raw_obj_points_3d_l1 * self.arti_points_3d_weight

        loss_arti_sym_corners_3d = zero
        loss_arti_hand_ord = zero
        loss_arti_scene_ord = zero

        total_loss = (
            loss_hand_2d_sv
            + loss_pose_reg_sv
            + loss_shape_reg_sv
            + loss_arti_center_2d
            + loss_arti_center_3d
            + loss_arti_corners_3d
            + loss_arti_points_3d
        )
        loss_dict["loss_heatmap_obj"] = loss_arti_center_2d
        loss_dict["loss_triang_obj"] = loss_arti_center_3d
        loss_dict["loss_triang"] = loss_arti_center_3d
        loss_dict["loss_obj_points"] = loss_arti_points_3d
        loss_dict["loss_obj_corner_3d"] = loss_arti_corners_3d
        loss_dict["loss_obj_corner_3d_total"] = loss_arti_corners_3d
        loss_dict["loss_obj_pose"] = loss_arti_center_3d + loss_arti_corners_3d + loss_arti_points_3d
        loss_dict["loss_arti_center_2d"] = loss_arti_center_2d
        loss_dict["loss_arti_center_3d"] = loss_arti_center_3d
        loss_dict["loss_arti_joints_3d"] = loss_arti_joints_3d
        loss_dict["loss_arti_corners_3d"] = loss_arti_corners_3d
        loss_dict["loss_arti_points_3d"] = loss_arti_points_3d
        loss_dict["loss_arti_sym_corners_3d"] = loss_arti_sym_corners_3d
        loss_dict["loss_arti_hand_ord"] = loss_arti_hand_ord
        loss_dict["loss_arti_scene_ord"] = loss_arti_scene_ord
        loss_dict["metric_obj_center_2d_epe_px"] = metric_obj_center_2d_epe_px
        loss_dict["metric_obj_center_2d_mse_px2"] = metric_obj_center_2d_mse_px2
        loss_dict["metric_obj_center_3d_l1_mm"] = metric_obj_center_3d_l1_mm
        loss_dict["metric_obj_center_3d_epe_mm"] = metric_obj_center_3d_epe_mm
        loss_dict["metric_obj_center_3d_mse_m2"] = metric_obj_center_3d_mse_m2
        loss_dict["metric_obj_corner_3d_l1_mm"] = metric_obj_corner_3d_l1_mm
        loss_dict["metric_obj_corner_3d_epe_mm"] = metric_obj_corner_3d_epe_mm
        loss_dict["metric_obj_corner_3d_mse_m2"] = metric_obj_corner_3d_mse_m2
        loss_dict["metric_obj_points_3d_l1_mm"] = metric_obj_points_3d_l1_mm
        loss_dict["metric_obj_center_conf_mean"] = metric_obj_center_conf_mean
        loss_dict["metric_obj_center_depth_mm"] = metric_obj_center_depth_mm
        loss_dict["loss"] = total_loss
        return total_loss, loss_dict

    def _log_sv_view_debug_scalars(self, preds, batch, step_idx, suffix=""):
        if self.summary is None or not hasattr(self.summary, "add_scalar"):
            return

        img = batch["image"]
        batch_size, n_views = img.shape[:2]
        obj_center_2d = preds["pred_obj_pixel"].view(batch_size, n_views, self.num_obj_joints, 2)
        pred_uv_conf = preds.get("pred_uv_conf", None)
        obj_center_conf = None
        if pred_uv_conf is not None:
            obj_center_conf = pred_uv_conf.view(batch_size, n_views, self.num_hand_joints + self.num_obj_joints, -1)
            obj_center_conf = obj_center_conf[:, :, self.num_hand_joints:, :].mean(dim=-1)

        gt_center_views = batch["target_obj_kp21"][:, :, -self.num_obj_joints:].to(device=obj_center_2d.device, dtype=obj_center_2d.dtype)
        gt_center_pixel = batch_cam_intr_projection(
            batch["target_cam_intr"].to(device=obj_center_2d.device, dtype=obj_center_2d.dtype),
            gt_center_views,
        )
        center_2d_epe_px = torch.linalg.norm(obj_center_2d - gt_center_pixel, dim=-1).mean()
        T_c2m = torch.linalg.inv(batch["target_cam_extr"]).to(device=obj_center_2d.device, dtype=obj_center_2d.dtype)
        pred_center_master = batch_triangulate_dlt_torch_confidence(
            obj_center_2d,
            batch["target_cam_intr"].to(device=obj_center_2d.device, dtype=obj_center_2d.dtype),
            T_c2m,
            confidences=obj_center_conf,
        )
        gt_center_master = batch["master_obj_kp21"][:, -self.num_obj_joints:].to(device=obj_center_2d.device, dtype=obj_center_2d.dtype)
        center_3d_l1_mm = torch.abs(pred_center_master - gt_center_master).mean() * 1000.0

        pred_center_views = batch_cam_extr_transf(
            T_c2m,
            pred_center_master.unsqueeze(1).expand(-1, n_views, -1, -1),
        )
        corner_rest = batch["master_obj_kp21_rest"][:, :8].to(device=obj_center_2d.device, dtype=obj_center_2d.dtype)
        pred_corners_views = self._build_object_points_from_pose_grad(
            corner_rest,
            preds["obj_view_rot6d_cam"],
            pred_center_views,
        )
        gt_corners_views = batch["target_obj_kp21"][:, :, :8].to(device=obj_center_2d.device, dtype=obj_center_2d.dtype)
        corner_3d_l1_mm = torch.abs(pred_corners_views - gt_corners_views).mean() * 1000.0

        rot_deg = None
        if batch.get("target_rot6d_label", None) is not None and preds.get("obj_view_rot6d_cam", None) is not None:
            rot_deg = torch.rad2deg(
                self.rotation_geodesic(
                    preds["obj_view_rot6d_cam"].reshape(-1, 6),
                    batch["target_rot6d_label"].to(
                        device=preds["obj_view_rot6d_cam"].device,
                        dtype=preds["obj_view_rot6d_cam"].dtype,
                    ).reshape(-1, 6),
                )
            ).mean()

        trans_mm = None
        if batch.get("target_t_label_rel", None) is not None and preds.get("obj_view_trans", None) is not None:
            trans_mm = (
                torch.linalg.norm(
                    preds["obj_view_trans"] - batch["target_t_label_rel"].to(
                        device=preds["obj_view_trans"].device,
                        dtype=preds["obj_view_trans"].dtype,
                    ),
                    dim=-1,
                ).mean() * 1000.0
            )

        prefix = f"slim_debug/{self.current_stage_name}{suffix}"
        self.summary.add_scalar(f"{prefix}/obj_center_2d_epe_px", center_2d_epe_px.item(), step_idx)
        self.summary.add_scalar(f"{prefix}/obj_center_3d_l1_mm", center_3d_l1_mm.item(), step_idx)
        self.summary.add_scalar(f"{prefix}/obj_corner_3d_l1_mm", corner_3d_l1_mm.item(), step_idx)
        self.summary.add_scalar(
            f"{prefix}/obj_center_depth_mm",
            (gt_center_views[..., 2].mean() * 1000.0).item(),
            step_idx,
        )
        if rot_deg is not None:
            self.summary.add_scalar(f"{prefix}/obj_view_rot_deg", rot_deg.item(), step_idx)
        if trans_mm is not None:
            self.summary.add_scalar(f"{prefix}/obj_view_trans_mm", trans_mm.item(), step_idx)
        if obj_center_conf is not None:
            self.summary.add_scalar(f"{prefix}/obj_center_conf_mean", obj_center_conf.mean().item(), step_idx)
            self.summary.add_scalar(f"{prefix}/obj_center_conf_std", obj_center_conf.std().item(), step_idx)

    def _log_visualizations(self, mode, batch, preds, step_idx, stage_name):
        if self.summary is None or not hasattr(self.summary, "add_image"):
            return

        img = batch["image"]
        batch_size, n_views = img.shape[:2]
        img_h, img_w = img.shape[-2:]
        batch_id = 0

        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])

        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"].view(batch_size, n_views, self.num_hand_verts + self.num_hand_joints, 2)
        pred_sv_joints_2d = self._normed_uv_to_pixel(pred_mano_2d_mesh_sv[:, :, self.num_hand_verts:], (img_h, img_w))
        pred_sv_verts_2d = self._normed_uv_to_pixel(pred_mano_2d_mesh_sv[:, :, :self.num_hand_verts], (img_h, img_w))
        gt_joints_2d_from_uvd = self._normed_uv_to_pixel(batch["target_joints_uvd"][..., :2], (img_h, img_w))
        gt_joints_2d = batch.get("target_joints_2d", None)
        if gt_joints_2d is None:
            gt_joints_2d = gt_joints_2d_from_uvd
        else:
            gt_joints_2d = gt_joints_2d.to(device=img.device, dtype=img.dtype)
        gt_verts_2d = self._normed_uv_to_pixel(batch["target_verts_uvd"][..., :2], (img_h, img_w))

        pred_obj_center_2d = preds["pred_obj_pixel"].view(batch_size, n_views, self.num_obj_joints, 2)
        gt_obj_center_cam = batch["target_obj_kp21"][:, :, -self.num_obj_joints:].to(device=img.device, dtype=img.dtype)
        gt_obj_center_2d = batch_cam_intr_projection(K_gt.to(device=img.device, dtype=img.dtype), gt_obj_center_cam)

        pred_center_master = preds["ref_obj_center"]
        if pred_center_master.dim() == 2:
            pred_center_master = pred_center_master.unsqueeze(1)
        pred_center_views = batch_cam_extr_transf(
            T_c2m_gt.to(device=img.device, dtype=img.dtype),
            pred_center_master.unsqueeze(1).expand(-1, n_views, -1, -1),
        )

        pred_obj_center_conf = preds["pred_uv_conf"].view(batch_size, n_views, self.num_hand_joints + self.num_obj_joints, -1)
        pred_obj_center_conf = pred_obj_center_conf[:, :, self.num_hand_joints:, :].mean(dim=-1)

        corner_rest = batch["master_obj_kp21_rest"][:, :8].to(device=img.device, dtype=img.dtype)
        pred_corners_views = self._build_object_points_from_pose_grad(
            corner_rest,
            preds["obj_view_rot6d_cam"],
            pred_center_views,
        )
        gt_corners_views = batch["target_obj_kp21"][:, :, :8].to(device=img.device, dtype=img.dtype)
        pred_corners_2d = batch_cam_intr_projection(K_gt.to(device=img.device, dtype=img.dtype), pred_corners_views)
        gt_corners_2d = batch_cam_intr_projection(K_gt.to(device=img.device, dtype=img.dtype), gt_corners_views)

        center_2d_err_px = torch.linalg.norm(pred_obj_center_2d - gt_obj_center_2d, dim=-1)
        gt_center_master = batch["master_obj_kp21"][:, -self.num_obj_joints:].to(device=img.device, dtype=img.dtype)
        center_3d_epe_mm = torch.linalg.norm(pred_center_master - gt_center_master, dim=-1) * 1000.0
        center_3d_epe_mm_views = center_3d_epe_mm[batch_id:batch_id + 1].expand(n_views, -1)

        pred_obj_view_rot_deg = None
        if batch.get("target_rot6d_label", None) is not None:
            pred_obj_view_rot_deg = torch.rad2deg(
                self.rotation_geodesic(
                    preds["obj_view_rot6d_cam"][batch_id],
                    batch["target_rot6d_label"][batch_id].to(
                        device=preds["obj_view_rot6d_cam"].device,
                        dtype=preds["obj_view_rot6d_cam"].dtype,
                    ),
                )
            ).unsqueeze(-1)

        img_views = img[batch_id]
        pred_sv_joints_2d_views = pred_sv_joints_2d[batch_id]
        pred_sv_verts_2d_views = pred_sv_verts_2d[batch_id]
        gt_joints_2d_views = gt_joints_2d[batch_id]
        gt_verts_2d_views = gt_verts_2d[batch_id]
        pred_obj_center_2d_views = pred_obj_center_2d[batch_id]
        gt_obj_center_2d_views = gt_obj_center_2d[batch_id]
        pred_obj_center_conf_views = pred_obj_center_conf[batch_id]
        pred_corners_2d_views = pred_corners_2d[batch_id]
        gt_corners_2d_views = gt_corners_2d[batch_id]
        center_2d_err_px_views = center_2d_err_px[batch_id]

        tag_prefix = self._stage_tb_prefix(mode, stage_name)

        self.summary.add_image(
            f"{tag_prefix}/direct_hand_joints_2d",
            tile_batch_images(
                draw_batch_joint_images(
                    pred_sv_joints_2d_views,
                    gt_joints_2d_views,
                    img_views,
                    step_idx,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/debug_gt_hand_consistency_2d",
            tile_batch_images(
                draw_batch_joint_images(
                    gt_joints_2d_from_uvd[batch_id],
                    gt_joints_2d_views,
                    img_views,
                    step_idx,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/direct_hand_mesh_2d",
            tile_batch_images(
                draw_batch_hand_mesh_images_2d(
                    gt_verts2d=gt_verts_2d_views,
                    pred_verts2d=pred_sv_verts_2d_views,
                    face=self.face,
                    tensor_image=img_views,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/direct_obj_center_confidence",
            tile_batch_images(
                draw_batch_joint_confidence_images(
                    pred_sv_joints_2d_views,
                    torch.ones_like(pred_sv_joints_2d_views[..., 0]),
                    img_views,
                    obj_center2d=pred_obj_center_2d_views,
                    obj_conf=pred_obj_center_conf_views,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/direct_obj_center_2d",
            tile_batch_images(
                draw_batch_mesh_images_pred(
                    gt_verts2d=gt_verts_2d_views,
                    pred_verts2d=pred_sv_verts_2d_views,
                    face=self.face,
                    gt_obj2d=gt_obj_center_2d_views,
                    pred_obj2d=pred_obj_center_2d_views,
                    gt_objc2d=None,
                    pred_objc2d=None,
                    intr=K_gt[batch_id],
                    tensor_image=img_views,
                    pred_obj_conf=pred_obj_center_conf_views,
                    pred_obj_error=center_2d_err_px_views,
                    pred_obj_trans_error=center_3d_epe_mm_views,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/direct_obj_corners_2d",
            tile_batch_images(
                draw_batch_mesh_images_pred(
                    gt_verts2d=gt_verts_2d_views,
                    pred_verts2d=pred_sv_verts_2d_views,
                    face=self.face,
                    gt_obj2d=gt_corners_2d_views,
                    pred_obj2d=pred_corners_2d_views,
                    gt_objc2d=gt_obj_center_2d_views,
                    pred_objc2d=pred_obj_center_2d_views,
                    intr=K_gt[batch_id],
                    tensor_image=img_views,
                    pred_obj_rot_error=pred_obj_view_rot_deg,
                    pred_obj_trans_error=center_3d_epe_mm_views,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )

    def training_step(self, batch, step_idx, **kwargs):
        img = batch["image"]
        batch_size = img.size(0)

        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = self._stage2_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._maybe_log_stage_transition("train", epoch_idx, stage_name, preds, batch)

        joints_3d_master_gt = batch["master_joints_3d"]
        verts_3d_master_gt = batch["master_verts_3d"]
        obj_sparse_3d_master_gt = batch["master_obj_sparse"]
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)
        obj_pc_sparse = batch.get("target_obj_pc_sparse", None)

        loss, loss_dict = self.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        pose_metric_dict = self.compute_object_pose_metrics(preds, batch) if stage_name == "stage2" else self._zero_metric_dict(img.device)
        sv_obj_metric_dict = self.compute_sv_object_pose_metrics(preds, batch) if stage_name in ["stage1", "stage2"] else {
            "metric_sv_obj_rot_l1": img.new_tensor(0.0),
            "metric_sv_obj_rot_deg": img.new_tensor(0.0),
            "metric_sv_obj_trans_l1": img.new_tensor(0.0),
            "metric_sv_obj_trans_epe": img.new_tensor(0.0),
            "metric_sv_obj_add": img.new_tensor(0.0),
            "metric_sv_obj_adds": img.new_tensor(0.0),
            "metric_sv_obj_mssd": img.new_tensor(0.0),
        }
        loss_is_finite = self._loss_dict_is_finite(loss_dict)
        pose_metrics_finite = self._metric_dict_is_finite({**pose_metric_dict, **sv_obj_metric_dict})
        if (not loss_is_finite) or (not pose_metrics_finite):
            logger.warning(
                f"[TrainNonFiniteEarlyExit] epoch={epoch_idx} stage={stage_name} step={step_idx} "
                f"loss_finite={loss_is_finite} pose_metrics_finite={pose_metrics_finite}"
            )
            self._maybe_log_stage_transition_loss("train", epoch_idx, stage_name, loss_dict, {**pose_metric_dict, **sv_obj_metric_dict})
            return None, loss_dict

        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]
        pred_mano_3d_joints_sv = pred_mano_3d_mesh_sv[:, self.num_hand_verts:]
        pred_mano_3d_verts_sv = pred_mano_3d_mesh_sv[:, :self.num_hand_verts]
        pred_mano_3d_mesh_master = preds["mano_3d_mesh_master"]
        pred_mano_3d_joints_master = pred_mano_3d_mesh_master[:, self.num_hand_verts:]
        pred_mano_3d_verts_master = pred_mano_3d_mesh_master[:, :self.num_hand_verts]
        pred_mano_3d_mesh_kp_master = preds.get("mano_3d_mesh_kp_master", None)
        pred_mano_3d_mesh_mesh_master = preds.get("mano_3d_mesh_mesh_master", None)
        pred_ref_hand_joints_3d = preds["ref_hand"]
        pred_obj_sparse_3d = preds["obj_xyz_master"][-1]

        self.MPJPE_SV_3D.feed(pred_mano_3d_joints_sv, gt_kp=joints_3d_rel_gt)
        self.MPVPE_SV_3D.feed(pred_mano_3d_verts_sv, gt_kp=verts_3d_rel_gt)
        self.PA_SV.feed(pred_mano_3d_joints_sv, joints_3d_rel_gt, pred_mano_3d_verts_sv, verts_3d_rel_gt)
        self.MPJPE_MASTER_3D.feed(pred_mano_3d_joints_master, gt_kp=joints_3d_master_gt)
        self.MPVPE_MASTER_3D.feed(pred_mano_3d_verts_master, gt_kp=verts_3d_master_gt)
        if pred_mano_3d_mesh_kp_master is not None:
            self.MPJPE_KP_MASTER_3D.feed(pred_mano_3d_mesh_kp_master[:, self.num_hand_verts:], gt_kp=joints_3d_master_gt)
            self.MPVPE_KP_MASTER_3D.feed(pred_mano_3d_mesh_kp_master[:, :self.num_hand_verts], gt_kp=verts_3d_master_gt)
        if pred_mano_3d_mesh_mesh_master is not None:
            self.MPJPE_MESH_MASTER_3D.feed(pred_mano_3d_mesh_mesh_master[:, self.num_hand_verts:], gt_kp=joints_3d_master_gt)
            self.MPVPE_MESH_MASTER_3D.feed(pred_mano_3d_mesh_mesh_master[:, :self.num_hand_verts], gt_kp=verts_3d_master_gt)
        self.MPRPE_HAND_3D.feed(pred_ref_hand_joints_3d, gt_kp=joints_3d_master_gt)
        self.OBJ_RECON_MASTER.feed(pred_obj_sparse_3d, obj_sparse_3d_master_gt)
        if obj_pc_sparse is not None and preds.get("obj_view_rot6d_cam", None) is not None and preds.get("obj_view_trans", None) is not None:
            pred_sv_obj_sparse_3d = self._build_object_points_from_hand_pose(
                batch["master_obj_sparse_rest"],
                preds["obj_view_rot6d_cam"],
                batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1],
                preds["obj_view_trans"],
            )
            self.OBJ_RECON_SV.feed(pred_sv_obj_sparse_3d.flatten(0, 1), obj_pc_sparse.flatten(0, 1))

        pose_metric_log_dict = self._pose_metrics_for_logging(pose_metric_dict)
        metric_log_dict = {**pose_metric_log_dict, **sv_obj_metric_dict}
        self.loss_metric.feed({**loss_dict, **metric_log_dict}, batch_size)
        self._maybe_log_stage_transition_loss("train", epoch_idx, stage_name, loss_dict, metric_log_dict)

        if step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)
            for k, v in metric_log_dict.items():
                self.summary.add_scalar(k, v.item(), step_idx)
            self._log_sv_view_debug_scalars(preds, batch, step_idx)
            pa_sv = self.PA_SV.get_measures()
            self.summary.add_scalar("MPJPE_SV_3D", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D", self.MPVPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("PA_SV_J", pa_sv["pa_mpjpe"], step_idx)
            self.summary.add_scalar("PA_SV_V", pa_sv["pa_mpvpe"], step_idx)
            self.summary.add_scalar("MPJPE_MASTER_3D", self.MPJPE_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_MASTER_3D", self.MPVPE_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPJPE_KP_MASTER_3D", self.MPJPE_KP_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_KP_MASTER_3D", self.MPVPE_KP_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPJPE_MESH_MASTER_3D", self.MPJPE_MESH_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_MESH_MASTER_3D", self.MPVPE_MESH_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPRPE_HAND_3D", self.MPRPE_HAND_3D.get_result(), step_idx)
            self.summary.add_scalar("OBJREC_MASTER_CD", self.OBJ_RECON_MASTER.cd.avg, step_idx)
            self.summary.add_scalar("OBJREC_MASTER_FS_5", self.OBJ_RECON_MASTER.fs_5.avg, step_idx)
            self.summary.add_scalar("OBJREC_MASTER_FS_10", self.OBJ_RECON_MASTER.fs_10.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_CD", self.OBJ_RECON_SV.cd.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_5", self.OBJ_RECON_SV.fs_5.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_10", self.OBJ_RECON_SV.fs_10.avg, step_idx)
            if step_idx % (self.train_log_interval * 10) == 0:
                if loss_is_finite and pose_metrics_finite:
                    with torch.no_grad():
                        self._log_visualizations("train", batch, preds, step_idx, stage_name)
                else:
                    logger.warning(
                        f"[VizSkip] mode=train epoch={epoch_idx} stage={stage_name} step={step_idx} "
                        "skip visualizations because train outputs contain non-finite values"
                    )

        return None, loss_dict

    def on_train_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([
            self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.PA_SV,
            self.MPJPE_MASTER_3D, self.MPVPE_MASTER_3D,
            self.MPJPE_KP_MASTER_3D, self.MPVPE_KP_MASTER_3D,
            self.MPJPE_MESH_MASTER_3D, self.MPVPE_MESH_MASTER_3D,
            self.MPRPE_HAND_3D,
            self.OBJ_RECON_SV, self.OBJ_RECON_MASTER,
        ], epoch_idx, comment=comment, summary=self.format_metric(mode="train"))
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.PA_SV.reset()
        self.MPJPE_MASTER_3D.reset()
        self.MPVPE_MASTER_3D.reset()
        self.MPJPE_KP_MASTER_3D.reset()
        self.MPVPE_KP_MASTER_3D.reset()
        self.MPJPE_MESH_MASTER_3D.reset()
        self.MPVPE_MESH_MASTER_3D.reset()
        self.MPRPE_HAND_3D.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_RECON_MASTER.reset()

    def format_metric(self, mode="train"):
        def _mm(value):
            return f"{float(value) * 1000.0:.1f}"

        def _get_meter(name, default=0.0):
            meter = self.loss_metric._losses.get(name, None)
            return float(meter.avg) if meter is not None else float(default)

        stage_short = self._stage_display_name(self.current_stage_name, short=True)
        warmup_suffix = (
            f" W{self.current_stage2_warmup:.2f}"
            if self.current_stage_name == "stage2" and self.current_stage2_warmup < 1.0
            else ""
        )

        if mode == "train":
            l_total = _get_meter("loss")
            l_h2d = _get_meter("loss_2d_sv")
            l_pose = _get_meter("loss_pose_reg_sv")
            l_shape = _get_meter("loss_shape_reg_sv")
            l_obj_2d = _get_meter("loss_arti_center_2d")
            l_obj_c = _get_meter("loss_arti_center_3d")
            l_obj_k = _get_meter("loss_arti_corners_3d")
            m_obj_2d_px = _get_meter("metric_obj_center_2d_epe_px")
            m_obj_c_mm = _get_meter("metric_obj_center_3d_l1_mm")
            m_obj_k_mm = _get_meter("metric_obj_corner_3d_l1_mm")
            m_obj_conf = _get_meter("metric_obj_center_conf_mean")
            m_obj_depth = _get_meter("metric_obj_center_depth_mm")
            m_sv_rot_deg = _get_meter("metric_sv_obj_rot_deg")
            m_sv_trans_mm = _get_meter("metric_sv_obj_trans_epe")
            m_sv_obj_mssd = _get_meter("metric_sv_obj_mssd")
            sv_j = self.MPJPE_SV_3D.get_result()
            sv_v = self.MPVPE_SV_3D.get_result()
            pa_sv = self.PA_SV.get_measures()
            return (
                f"{stage_short}{warmup_suffix} | "
                f"L {l_total:.3f} | "
                f"H 2D/P/S {l_h2d:.3f}/{l_pose:.3f}/{l_shape:.3f} | "
                f"O 2D/C/K {l_obj_2d:.3f}/{l_obj_c:.3f}/{l_obj_k:.3f} | "
                f"H MAJ/V {_mm(sv_j)}/{_mm(sv_v)} PAJ/V {_mm(pa_sv['pa_mpjpe'])}/{_mm(pa_sv['pa_mpvpe'])} | "
                f"O px/C/K {m_obj_2d_px:.1f}/{m_obj_c_mm:.1f}/{m_obj_k_mm:.1f} | "
                f"MSSD {_mm(m_sv_obj_mssd)} RT {m_sv_rot_deg:.1f}/{_mm(m_sv_trans_mm)} | "
                f"Conf/Dep {m_obj_conf:.3f}/{m_obj_depth:.0f}"
            )

        if mode in ["val", "val_full"]:
            pa_sv = self.PA_SV.get_measures()
            sv_j = self.MPJPE_SV_3D.get_result()
            sv_v = self.MPVPE_SV_3D.get_result()
            sv_obj_rot_deg = _get_meter("metric_sv_obj_rot_deg")
            sv_obj_trans_epe = _get_meter("metric_sv_obj_trans_epe")
            sv_obj_add = _get_meter("metric_sv_obj_add")
            sv_obj_adds = _get_meter("metric_sv_obj_adds")
            sv_obj_mssd = _get_meter("metric_sv_obj_mssd")
            recon_mode = self.current_val_recon_mode.upper()
            return (
                f"{stage_short} | Rec {recon_mode} | "
                f"H MAJ/V {_mm(sv_j)}/{_mm(sv_v)} PAJ/V {_mm(pa_sv['pa_mpjpe'])}/{_mm(pa_sv['pa_mpvpe'])} | "
                f"O A/S/M {_mm(sv_obj_add)}/{_mm(sv_obj_adds)}/{_mm(sv_obj_mssd)} | "
                f"RT {sv_obj_rot_deg:.1f}/{_mm(sv_obj_trans_epe)}"
            )

        return super().format_metric(mode=mode)
