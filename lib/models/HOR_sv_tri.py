import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.basic_metric import LossMetric
from ..metrics.mean_epe import MeanEPE
from ..metrics.object_pose_metric import ObjectPoseMetric
from ..metrics.object_recon_metric import ObjectReconMetric
from ..metrics.pa_eval import PAEval
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import param_size
from ..utils.net_utils import init_weights
from ..utils.object_pose_utils import pose_from_keypoints
from ..utils.recorder import Recorder
from ..utils.transform import batch_cam_extr_transf, batch_cam_intr_projection, rot6d_to_rotmat, rotmat_to_rot6d
from ..utils.triangulation import batch_triangulate_dlt_torch, batch_triangulate_dlt_torch_confidence
from ..viztools.draw import (
    draw_batch_object_kp_confidence_images,
    draw_batch_object_kp_images,
    draw_batch_hand_mesh_images_2d,
    draw_batch_joint_confidence_images,
    draw_batch_joint_images,
    draw_batch_mesh_images_pred,
    tile_batch_images,
)
from .backbones import build_backbone
from .bricks.conv import ConvBlock
from .bricks.utils import GraphRegression, HOT, ManoDecoder, SelfAttn
from .heads import build_head
from .integal_pose import integral_heatmap2d
from .model_abstraction import ModuleAbstract


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x).flatten(1)
        scale = self.fc(scale).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.se = SEModule(channels)
        self.spatial = SpatialGate(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.spatial(out)
        out = self.act(out + residual)
        return out


class SharedHOStem(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            ResidualAttentionBlock(channels),
        )

    def forward(self, x):
        return self.block(x)


class AttentionPool2d(nn.Module):
    def __init__(self, channels, out_dim=None):
        super().__init__()
        out_dim = channels if out_dim is None else out_dim
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Linear(channels, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, feat):
        batch_size, channels, height, width = feat.shape
        weights = self.attn(feat).reshape(batch_size, 1, height * width)
        weights = torch.softmax(weights, dim=-1)
        token = feat.reshape(batch_size, channels, height * width)
        token = torch.bmm(token, weights.transpose(1, 2)).squeeze(-1)
        token = self.proj(token)
        return token, weights.reshape(batch_size, 1, height, width)


class ObjectQueryHandReference(nn.Module):
    def __init__(self, channels, num_heads=4, pooled_hw=8):
        super().__init__()
        self.channels = channels
        self.pooled_hw = int(max(pooled_hw, 2))
        self.obj_pool = nn.AdaptiveAvgPool2d((self.pooled_hw, self.pooled_hw))
        self.hand_pool = nn.AdaptiveAvgPool2d((self.pooled_hw, self.pooled_hw))
        self.obj_norm = nn.LayerNorm(channels)
        self.hand_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, obj_feat, hand_feat):
        batch_size, channels, height, width = obj_feat.shape
        obj_low = self.obj_pool(obj_feat)
        hand_low = self.hand_pool(hand_feat)

        obj_tokens = obj_low.flatten(2).transpose(1, 2)
        hand_tokens = hand_low.flatten(2).transpose(1, 2)
        obj_tokens = self.obj_norm(obj_tokens)
        hand_tokens = self.hand_norm(hand_tokens)

        ref_tokens, _ = self.cross_attn(query=obj_tokens, key=hand_tokens, value=hand_tokens)
        ref_feat = ref_tokens.transpose(1, 2).reshape(batch_size, channels, self.pooled_hw, self.pooled_hw)
        ref_feat = F.interpolate(ref_feat, size=(height, width), mode="bilinear", align_corners=False)
        ref_feat = self.out_proj(ref_feat)
        ref_gate = self.gate(torch.cat((obj_feat, ref_feat), dim=1))
        fused_obj = obj_feat + ref_gate * ref_feat
        fused_obj = torch.nan_to_num(fused_obj, nan=0.0, posinf=1.0, neginf=-1.0)
        return fused_obj, ref_gate


class ObjectHeatmapHead(nn.Module):
    def __init__(self, channels, num_keypoints):
        super().__init__()
        self.block = nn.Sequential(
            ResidualAttentionBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, num_keypoints, kernel_size=1, bias=True),
        )

    def forward(self, x):
        heatmap = torch.sigmoid(self.block(x) * 0.5)
        return torch.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class ObjectVisibilityHead(nn.Module):
    def __init__(self, token_dim, num_keypoints):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_keypoints),
        )

    def forward(self, token):
        logits = self.mlp(token)
        conf = torch.sigmoid(logits).unsqueeze(-1)
        return torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class ObjectPoseGateHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        hidden = max(in_dim, 64)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        gate = torch.sigmoid(self.mlp(feat))
        return torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class SharedMultiViewFusion(nn.Module):
    def __init__(
        self,
        channels,
        token_dim,
        canonical_focal=200.0,
        geo_ratio=0.75,
        trans_scale_m=0.25,
        master_logit_boost=0.5,
        residual_scale=0.5,
    ):
        super().__init__()
        geo_channels = max(int(channels * geo_ratio), 3)
        geo_channels = min(geo_channels, max(channels - 16, 3))
        geo_channels = max((geo_channels // 3) * 3, 3)
        if geo_channels >= channels:
            geo_channels = max(((channels - 16) // 3) * 3, 3)
        self.channels = channels
        self.geo_channels = min(geo_channels, channels)
        self.app_channels = channels - self.geo_channels
        self.canonical_focal = float(canonical_focal)
        self.trans_scale_m = max(float(trans_scale_m), 1e-6)
        self.master_logit_boost = float(master_logit_boost)
        self.residual_scale = float(residual_scale)

        self.geo_proj = nn.Sequential(
            nn.Conv2d(channels, self.geo_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.geo_channels),
            nn.GELU(),
        )
        self.app_proj = None
        if self.app_channels > 0:
            self.app_proj = nn.Sequential(
                nn.Conv2d(channels, self.app_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.app_channels),
                nn.GELU(),
            )

        self.trans_embed = nn.Sequential(
            nn.Linear(3, self.geo_channels),
            nn.GELU(),
            nn.Linear(self.geo_channels, self.geo_channels),
        )
        self.view_weight_head = nn.Sequential(
            nn.Linear(token_dim + 6, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )
        self.master_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            ResidualAttentionBlock(channels),
        )
        self.view_gate = nn.Sequential(
            nn.Linear(token_dim + 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.view_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            ResidualAttentionBlock(channels),
        )

    @staticmethod
    def _gather_by_master(tensor, master_id):
        batch_size = tensor.shape[0]
        batch_idx = torch.arange(batch_size, device=tensor.device)
        return tensor[batch_idx, master_id]

    @staticmethod
    def _build_diag(v0, v1, device, dtype):
        batch_size, num_views = v0.shape
        diag = torch.zeros(batch_size, num_views, 3, 3, device=device, dtype=dtype)
        diag[..., 0, 0] = v0
        diag[..., 1, 1] = v1
        diag[..., 2, 2] = 1.0
        return diag

    @staticmethod
    def _rotation_angle(rotmat):
        trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return torch.acos(cos_theta)

    def _build_feature_transforms(self, cam_intr, cam_extr, master_id):
        batch_size, num_views = cam_intr.shape[:2]
        device = cam_intr.device
        dtype = cam_intr.dtype
        batch_idx = torch.arange(batch_size, device=device)

        master_extr = cam_extr[batch_idx, master_id]
        master_intr = cam_intr[batch_idx, master_id]
        view_to_master = torch.matmul(torch.linalg.inv(master_extr).unsqueeze(1), cam_extr)
        master_to_view = torch.linalg.inv(view_to_master)

        fx = cam_intr[..., 0, 0].clamp_min(1e-6)
        fy = cam_intr[..., 1, 1].clamp_min(1e-6)
        master_fx = master_intr[:, 0, 0].unsqueeze(1).clamp_min(1e-6)
        master_fy = master_intr[:, 1, 1].unsqueeze(1).clamp_min(1e-6)

        sv_scale = self._build_diag(
            self.canonical_focal / fx,
            self.canonical_focal / fy,
            device=device,
            dtype=dtype,
        )
        master_scale_inv = self._build_diag(
            master_fx / self.canonical_focal,
            master_fy / self.canonical_focal,
            device=device,
            dtype=dtype,
        )
        rot_view_to_master = view_to_master[..., :3, :3]
        xf_view_to_master = torch.matmul(master_scale_inv, torch.matmul(rot_view_to_master, sv_scale))
        xf_master_to_view = torch.linalg.inv(xf_view_to_master)

        trans_view_to_master = view_to_master[..., :3, 3]
        trans_master_to_view = master_to_view[..., :3, 3]
        geom_stats = torch.stack(
            (
                torch.log(fx / master_fx),
                torch.log(fy / master_fy),
                trans_view_to_master[..., 0] / self.trans_scale_m,
                trans_view_to_master[..., 1] / self.trans_scale_m,
                trans_view_to_master[..., 2] / self.trans_scale_m,
                self._rotation_angle(rot_view_to_master) / math.pi,
            ),
            dim=-1,
        )
        return xf_view_to_master, xf_master_to_view, trans_view_to_master, trans_master_to_view, geom_stats

    def _apply_geo_transform(self, geo_feat, xf, trans_bias):
        batch_size, num_views, _, height, width = geo_feat.shape
        geo_groups = self.geo_channels // 3
        geo_feat = geo_feat.view(batch_size, num_views, geo_groups, 3, height, width).permute(0, 1, 2, 4, 5, 3)
        geo_feat = torch.einsum("bvij,bvghwj->bvghwi", xf, geo_feat)
        geo_feat = geo_feat.permute(0, 1, 2, 5, 3, 4).reshape(batch_size, num_views, self.geo_channels, height, width)
        return geo_feat + trans_bias.view(batch_size, num_views, self.geo_channels, 1, 1)

    def forward(self, feat, token_views, cam_intr, cam_extr, master_id):
        batch_size, num_views, channels, height, width = feat.shape
        flat_feat = feat.flatten(0, 1)
        geo_feat = self.geo_proj(flat_feat).view(batch_size, num_views, self.geo_channels, height, width)
        app_feat = None
        if self.app_proj is not None:
            app_feat = self.app_proj(flat_feat).view(batch_size, num_views, self.app_channels, height, width)

        xf_view_to_master, xf_master_to_view, trans_view_to_master, trans_master_to_view, geom_stats = self._build_feature_transforms(
            cam_intr,
            cam_extr,
            master_id,
        )
        trans_bias_master = self.trans_embed(trans_view_to_master / self.trans_scale_m)
        geo_feat_master = self._apply_geo_transform(geo_feat, xf_view_to_master, trans_bias_master)

        view_logits = self.view_weight_head(torch.cat((token_views, geom_stats), dim=-1)).squeeze(-1)
        master_mask = torch.zeros(batch_size, num_views, device=feat.device, dtype=feat.dtype)
        master_mask.scatter_(1, master_id.unsqueeze(-1), 1.0)
        view_logits = view_logits + master_mask * self.master_logit_boost
        view_weights = torch.softmax(view_logits, dim=1)

        weight_map = view_weights.view(batch_size, num_views, 1, 1, 1)
        fused_geo_master = torch.sum(geo_feat_master * weight_map, dim=1)
        if app_feat is None:
            fused_master = fused_geo_master
        else:
            fused_app_master = torch.sum(app_feat * weight_map, dim=1)
            fused_master = torch.cat((fused_geo_master, fused_app_master), dim=1)

        master_anchor = self._gather_by_master(feat, master_id)
        fused_master = self.master_refine(torch.cat((fused_master, master_anchor), dim=1))

        fused_geo_master = fused_master[:, :self.geo_channels]
        fused_geo_views = fused_geo_master.unsqueeze(1).expand(-1, num_views, -1, -1, -1)
        trans_bias_view = self.trans_embed(trans_master_to_view / self.trans_scale_m)
        fused_geo_views = self._apply_geo_transform(fused_geo_views, xf_master_to_view, trans_bias_view)

        if self.app_channels > 0:
            fused_app_master = fused_master[:, self.geo_channels:]
            fused_app_views = fused_app_master.unsqueeze(1).expand(-1, num_views, -1, -1, -1)
            fused_views = torch.cat((fused_geo_views, fused_app_views), dim=2)
        else:
            fused_views = fused_geo_views

        gate_input = torch.cat((token_views, view_weights.unsqueeze(-1), master_mask.unsqueeze(-1)), dim=-1)
        view_gate = self.view_gate(gate_input).view(batch_size, num_views, channels, 1, 1)
        fused_views = fused_views * view_gate
        refined_views = self.view_refine(torch.cat((feat.flatten(0, 1), fused_views.flatten(0, 1)), dim=1))
        refined_views = feat + self.residual_scale * refined_views.view(batch_size, num_views, channels, height, width)
        refined_views = torch.nan_to_num(refined_views, nan=0.0, posinf=1.0, neginf=-1.0)

        aux = {
            "view_weights": view_weights,
            "view_gate": view_gate.squeeze(-1).squeeze(-1),
            "master_mask": master_mask,
            "geom_stats": geom_stats,
            "fused_master": fused_master,
        }
        return refined_views, aux


@MODEL.register_module()
class POEM_SV_Tri(nn.Module, ModuleAbstract):
    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET
        self.summary = None
        self.debug_logs = bool(self.train_cfg.get("DEBUG_LOGS", False))

        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.num_hand_joints = cfg.get("NUM_HAND_JOINTS", 21)
        self.num_hand_verts = cfg.get("NUM_HAND_VERTS", 778)
        self.num_obj_joints = int(cfg.DATA_PRESET.get("NUM_OBJ_JOINTS", cfg.get("NUM_OBJ_JOINTS", 21)))
        self.hot_dim = 384
        self.shared_dim = 128
        self.obj_token_dim = 128
        self.obj_context_dim = 128
        self.num_hand_tokens = self.num_hand_joints

        self.joints_loss_type = cfg.LOSS.get("JOINTS_LOSS_TYPE", "l2")
        self.coord_loss = nn.L1Loss()
        self.criterion_joints = nn.MSELoss() if self.joints_loss_type == "l2" else nn.L1Loss()

        self.obj_2d_weight = cfg.LOSS.get("HFL_OBJ_REG_N", 1.0)
        self.obj_conf_weight = cfg.LOSS.get("HFL_OBJ_CONF_N", 1.0)
        self.obj_tri_reproj_weight = cfg.LOSS.get("OBJ_TRI_REPROJ_N", 1.0)
        self.obj_pose_gate_weight = cfg.LOSS.get("OBJ_POSE_GATE_N", 0.5)
        self.pose_reg_weight = cfg.LOSS.get("POSE_N", 0.01)
        self.shape_reg_weight = cfg.LOSS.get("SHAPE_N", 1.0)
        self.triangulation_weight = cfg.LOSS.get("TRIANGULATION_N", 10.0)
        self.triangulation_hand_weight = cfg.LOSS.get("TRIANGULATION_HAND_N", self.triangulation_weight)
        self.decoder_hand_weight = cfg.LOSS.get("DECODER_HAND_N", 10.0)
        self.decoder_proj_weight = cfg.LOSS.get("DECODER_PROJ_N", 10.0)
        self.mano_proj_weight = cfg.LOSS.get("MANO_PROJ_N", 50.0)
        self.obj_pose_rot_weight = cfg.LOSS.get("OBJ_POSE_ROT_N", 5.0)
        self.obj_pose_rot6d_weight = cfg.LOSS.get("OBJ_POSE_ROT6D_N", 0.1)
        self.obj_pose_trans_weight = cfg.LOSS.get("OBJ_POSE_TRANS_N", 10.0)
        self.obj_pose_points_weight = cfg.LOSS.get("OBJ_POSE_POINTS_N", 10.0)
        self.obj_init_rot_weight = cfg.LOSS.get("OBJ_INIT_ROT_N", 2.0)
        self.obj_init_rot6d_weight = cfg.LOSS.get("OBJ_INIT_ROT6D_N", self.obj_pose_rot6d_weight)
        self.obj_init_trans_weight = cfg.LOSS.get("OBJ_INIT_TRANS_N", self.obj_pose_trans_weight)
        self.obj_view_rot_weight = cfg.LOSS.get("OBJ_VIEW_ROT_N", 1.0)
        self.obj_view_rot6d_weight = cfg.LOSS.get("OBJ_VIEW_ROT6D_N", 0.05)
        self.obj_view_trans_weight = cfg.LOSS.get("OBJ_VIEW_TRANS_N", self.obj_pose_trans_weight)
        self.obj_view_points_weight = cfg.LOSS.get("OBJ_VIEW_POINTS_N", self.obj_pose_points_weight)
        self.stage1_obj_rot6d_l1_scale = float(cfg.LOSS.get("STAGE1_OBJ_ROT6D_L1_SCALE", 0.2))
        self.stage1_obj_init_rot6d_l1_scale = float(
            cfg.LOSS.get("STAGE1_OBJ_INIT_ROT6D_L1_SCALE", self.stage1_obj_rot6d_l1_scale)
        )
        self.stage1_obj_rot6d_l1_cd_tau_m = float(cfg.LOSS.get("STAGE1_OBJ_ROT6D_L1_CD_TAU_M", 0.01))
        self.stage1_obj_rot6d_l1_mask_power = float(cfg.LOSS.get("STAGE1_OBJ_ROT6D_L1_MASK_POWER", 1.0))
        self.stage1_obj_hand_ref_detach = bool(cfg.get("STAGE1_OBJ_HAND_REF_DETACH", True))
        self.stage1_obj_hand_ref_heads = int(cfg.get("STAGE1_OBJ_HAND_REF_HEADS", 4))
        self.stage1_obj_hand_ref_pooled_hw = int(cfg.get("STAGE1_OBJ_HAND_REF_POOLED_HW", 8))
        self.conf_tri_hand_tau_px = float(cfg.get("CONF_TRI_HAND_TAU_PX", 24.0))
        self.conf_tri_refine_iters = int(cfg.get("CONF_TRI_REFINE_ITERS", 2))
        self.conf_obj_tau_px = float(cfg.get("CONF_OBJ_TAU_PX", 24.0))
        self.conf_obj_tri_tau_px = float(cfg.get("CONF_OBJ_TRI_TAU_PX", self.conf_obj_tau_px))
        self.obj_pnp_topk = int(cfg.get("OBJ_PNP_TOPK", 12))
        self.obj_pnp_min_points = int(cfg.get("OBJ_PNP_MIN_POINTS", 6))
        self.obj_pnp_min_conf = float(cfg.get("OBJ_PNP_MIN_CONF", 0.05))
        self.obj_pnp_reproj_px = float(cfg.get("OBJ_PNP_REPROJ_PX", 8.0))
        self.obj_pnp_max_reproj_px = float(cfg.get("OBJ_PNP_MAX_REPROJ_PX", 48.0))
        self.obj_pnp_max_trans_m = float(cfg.get("OBJ_PNP_MAX_TRANS_M", 2.0))
        self.obj_pose_gate_rot_tau_deg = float(cfg.get("OBJ_POSE_GATE_ROT_TAU_DEG", 20.0))
        self.obj_pose_gate_trans_tau_m = float(cfg.get("OBJ_POSE_GATE_TRANS_TAU_M", 0.08))
        self.obj_pose_loss_rot_tau_deg = float(cfg.get("OBJ_POSE_LOSS_ROT_TAU_DEG", self.obj_pose_gate_rot_tau_deg))
        self.obj_pose_loss_trans_tau_m = float(cfg.get("OBJ_POSE_LOSS_TRANS_TAU_M", self.obj_pose_gate_trans_tau_m))
        self.num_obj_classes = int(cfg.get("NUM_OBJ_CLASSES", 32))
        self.obj_id_embed_dim = int(cfg.get("OBJ_ID_EMBED_DIM", 32))
        self.shared_mv_fusion_enabled = bool(cfg.get("SHARED_MV_FUSION_ENABLED", cfg.get("OBJ_MV_FUSION_ENABLED", True)))
        self.shared_mv_canonical_focal = float(cfg.get("SHARED_MV_CANONICAL_FOCAL", cfg.get("OBJ_MV_CANONICAL_FOCAL", 200.0)))
        self.shared_mv_geo_ratio = float(cfg.get("SHARED_MV_GEO_RATIO", cfg.get("OBJ_MV_GEO_RATIO", 0.75)))
        self.shared_mv_trans_scale_m = float(cfg.get("SHARED_MV_TRANS_SCALE_M", cfg.get("OBJ_MV_TRANS_SCALE_M", 0.25)))
        self.shared_mv_master_logit_boost = float(cfg.get("SHARED_MV_MASTER_LOGIT_BOOST", cfg.get("OBJ_MV_MASTER_LOGIT_BOOST", 0.5)))
        self.shared_mv_residual_scale = float(cfg.get("SHARED_MV_RESIDUAL_SCALE", cfg.get("OBJ_MV_RESIDUAL_SCALE", 0.5)))
        self.stage1_end_epoch = cfg.TRAIN.get("STAGE1_END_EPOCH", 999999)
        self.stage2_warmup_epochs = cfg.TRAIN.get("STAGE2_WARMUP_EPOCHS", 0)

        self.current_stage_name = "stage1"
        self.current_stage2_warmup = 0.0
        self.current_val_recon_mode = "sv"

        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)
        assert self.img_backbone.name in ["resnet18", "resnet34", "resnet50"], "Wrong backbone for HOR"
        if hasattr(self.img_backbone, "fc") and isinstance(self.img_backbone.fc, nn.Module):
            for param in self.img_backbone.fc.parameters():
                param.requires_grad = False

        if self.img_backbone.name in ["resnet18", "resnet34"]:
            self.feat_size = (512, 256, 128, 64)
        else:
            self.feat_size = (2048, 1024, 512, 256)

        self.feat_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True, norm="bn"),
            ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True, norm="bn"),
            ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True, norm="bn"),
        ])
        self.feat_in = ConvBlock(self.feat_size[3], self.shared_dim, kernel_size=1, padding=0, relu=False, norm=None)

        self.shared_stem = SharedHOStem(self.shared_dim)
        self.hand_adapter = nn.Sequential(
            ResidualAttentionBlock(self.shared_dim),
            ResidualAttentionBlock(self.shared_dim),
        )
        self.obj_adapter = nn.Sequential(
            ResidualAttentionBlock(self.shared_dim),
            ResidualAttentionBlock(self.shared_dim),
        )
        self.obj_2d_head = ObjectHeatmapHead(self.shared_dim, self.num_obj_joints)
        self.obj_vis_head = ObjectVisibilityHead(self.obj_token_dim, self.num_obj_joints)
        self.obj_pose_gate_head = ObjectPoseGateHead(self.obj_token_dim + self.obj_id_embed_dim + 1)
        self.obj_id_embed = nn.Embedding(self.num_obj_classes, self.obj_id_embed_dim)
        self.obj_id_film = nn.Sequential(
            nn.Linear(self.obj_id_embed_dim, self.shared_dim * 2),
            nn.GELU(),
            nn.Linear(self.shared_dim * 2, self.shared_dim * 2),
        )
        self.shared_pool = AttentionPool2d(self.shared_dim, out_dim=self.obj_token_dim)
        self.shared_mv_fusion = SharedMultiViewFusion(
            channels=self.shared_dim,
            token_dim=self.obj_token_dim,
            canonical_focal=self.shared_mv_canonical_focal,
            geo_ratio=self.shared_mv_geo_ratio,
            trans_scale_m=self.shared_mv_trans_scale_m,
            master_logit_boost=self.shared_mv_master_logit_boost,
            residual_scale=self.shared_mv_residual_scale,
        ) if self.shared_mv_fusion_enabled else None

        self.hand_pool = nn.AdaptiveAvgPool2d(1)
        self.obj_pool = AttentionPool2d(self.shared_dim, out_dim=self.obj_token_dim)
        self.obj_query_hand_ref = ObjectQueryHandReference(
            channels=self.shared_dim,
            num_heads=self.stage1_obj_hand_ref_heads,
            pooled_hw=self.stage1_obj_hand_ref_pooled_hw,
        )

        self.hot_pose = HOT(self.shared_dim, self.hot_dim, self.num_hand_joints, 0)
        self.att_0 = SelfAttn(self.hot_dim, self.hot_dim, dropout=0.1)
        self.att_1 = SelfAttn(self.hot_dim, self.hot_dim, dropout=0.1)
        self.mano_fuse = GraphRegression(self.num_hand_tokens, self.hot_dim, 32, last=False)
        self.hand_context_proj = nn.Sequential(
            nn.Linear(self.hot_dim, self.obj_context_dim),
            nn.GELU(),
            nn.Linear(self.obj_context_dim, self.obj_context_dim),
        )
        self.mano_fc = nn.Sequential(
            nn.Linear(32 * self.num_hand_tokens, 32 * self.num_hand_tokens),
            nn.LeakyReLU(0.1),
            nn.Linear(32 * self.num_hand_tokens, 16 * 6 + 10 + 3),
        )
        self.mano_decoder = ManoDecoder(
            self.data_preset_cfg.CENTER_IDX,
            self.data_preset_cfg.BBOX_3D_SIZE,
            self.data_preset_cfg.IMAGE_SIZE,
        )
        self.face = self.mano_decoder.face
        self.ptEmb_head = build_head(cfg.HEAD, data_preset=self.data_preset_cfg)

        self.fc_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]

        self.loss_metric = LossMetric(cfg)
        self.MPJPE_SV_3D = MeanEPE(cfg, "SV_J")
        self.MPVPE_SV_3D = MeanEPE(cfg, "SV_V")
        self.PA_SV = PAEval(cfg, mesh_score=True)
        self.OBJ_RECON_SV = ObjectReconMetric(cfg, name="SVObjRec")
        self.OBJ_POSE_VAL = ObjectPoseMetric(cfg, name="Obj")
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL

        self.init_weights()
        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} stage2-ready SV-Tri frontend enabled")

    def init_weights(self):
        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        init_weights(self, pretrained=self.cfg.PRETRAINED)

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def set_train_stage(self, stage_name: str):
        self.current_stage_name = stage_name
        for param in self.parameters():
            param.requires_grad = True
        if hasattr(self.img_backbone, "fc") and isinstance(self.img_backbone.fc, nn.Module):
            for param in self.img_backbone.fc.parameters():
                param.requires_grad = False

    def _resolve_stage(self, epoch_idx):
        if epoch_idx < self.stage1_end_epoch:
            return "stage1"
        return "stage2"

    @staticmethod
    def _stage_to_interaction_mode(stage_name):
        return "ho" if stage_name == "stage2" else "hand"

    def _stage2_object_warmup(self, epoch_idx, stage_name):
        if stage_name != "stage2":
            return 0.0
        if self.stage2_warmup_epochs <= 0:
            return 1.0
        progress = (epoch_idx - self.stage1_end_epoch + 1) / float(self.stage2_warmup_epochs)
        return float(max(0.0, min(1.0, progress)))

    @staticmethod
    def _zero_metric_dict(device):
        zero = torch.tensor(0.0, device=device)
        return {
            "metric_obj_rot_l1": zero,
            "metric_obj_rot_deg": zero,
            "metric_obj_trans_l1": zero,
            "metric_obj_trans_epe": zero,
            "metric_obj_add": zero,
            "metric_obj_adds": zero,
        }

    def extract_img_feat(self, img):
        batch = img.size(0)
        if img.dim() == 5:
            batch, num_cams, channels, height, width = img.size()
            img = img.view(batch * num_cams, channels, height, width)
        img_feats = self.img_backbone(image=img)
        global_feat = img_feats["res_layer4_mean"]
        if isinstance(img_feats, dict):
            img_feats = [value for value in img_feats.values() if len(value.size()) == 4]
        return img_feats, global_feat

    def feat_decode(self, mlvl_feats):
        mlvl_feats_rev = list(reversed(mlvl_feats))
        x = mlvl_feats_rev[0]
        for i, fde in enumerate(self.feat_delayer):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
            x = fde(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.feat_in(x)
        return x

    @staticmethod
    def _normed_uv_to_pixel(coords_uv, image_hw):
        img_h, img_w = image_hw
        scale = coords_uv.new_tensor([img_w, img_h])
        return (coords_uv + 1.0) * 0.5 * scale

    @staticmethod
    def _fit_ortho_camera(points_xy, points_uv, visibility=None, eps=1e-6):
        points_xy = torch.nan_to_num(points_xy, nan=0.0, posinf=0.0, neginf=0.0)
        points_uv = torch.nan_to_num(points_uv, nan=0.0, posinf=0.0, neginf=0.0)
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

    @staticmethod
    def _ortho_project_points(points_xyz, ortho_cam):
        scale = ortho_cam[..., 0:1].unsqueeze(-2)
        trans = ortho_cam[..., 1:].unsqueeze(-2)
        points_uv = points_xyz[..., :2] * scale + trans
        return torch.nan_to_num(points_uv, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _project_matrix_to_rotation(rot6d):
        return rotmat_to_rot6d(rot6d_to_rotmat(rot6d))

    @staticmethod
    def _heatmap_peak_confidence(heatmap_map):
        flat_map = heatmap_map.flatten(2)
        peak_conf = flat_map.max(dim=-1, keepdim=True).values
        peak_conf = torch.nan_to_num(peak_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return peak_conf

    @staticmethod
    def _heatmap_spread_confidence(uv_pdf, uv_coord, image_shape, tau_px=18.0):
        img_h, img_w = image_shape
        _, _, hm_h, hm_w = uv_pdf.shape
        device = uv_pdf.device
        dtype = uv_pdf.dtype

        ys = torch.linspace(0.0, 1.0, hm_h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, hm_w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.view(1, 1, hm_h, hm_w)
        grid_y = grid_y.view(1, 1, hm_h, hm_w)

        mean_x = uv_coord[..., 0].unsqueeze(-1).unsqueeze(-1)
        mean_y = uv_coord[..., 1].unsqueeze(-1).unsqueeze(-1)
        var_x = torch.sum(uv_pdf * torch.pow(grid_x - mean_x, 2), dim=(-2, -1))
        var_y = torch.sum(uv_pdf * torch.pow(grid_y - mean_y, 2), dim=(-2, -1))
        std_px = torch.sqrt(var_x * float(img_w * img_w) + var_y * float(img_h * img_h) + 1e-9)
        conf = (1.0 / (1.0 + std_px / max(float(tau_px), 1e-6))).unsqueeze(-1)
        conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return conf

    @staticmethod
    def rotation_geodesic(pred_rot6d, gt_rot6d):
        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return torch.acos(cos_theta)

    @staticmethod
    def _build_object_points_from_pose(obj_points_rest, rot6d, trans_abs):
        obj_points_rest = torch.nan_to_num(obj_points_rest, nan=0.0, posinf=10.0, neginf=-10.0).float()
        rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=10.0, neginf=-10.0).float()
        trans_abs = torch.nan_to_num(trans_abs, nan=0.0, posinf=10.0, neginf=-10.0).float()
        if rot6d.dim() == 2:
            if obj_points_rest.dim() == 4:
                obj_points_rest = obj_points_rest[:, 0]
            rotmat = rot6d_to_rotmat(rot6d)
            return torch.matmul(obj_points_rest, rotmat.transpose(1, 2)) + trans_abs.unsqueeze(1)

        batch_size, num_views = rot6d.shape[:2]
        if obj_points_rest.dim() == 3:
            obj_points_rest = obj_points_rest.unsqueeze(1).expand(-1, num_views, -1, -1)
        rotmat = rot6d_to_rotmat(rot6d.reshape(-1, 6)).view(batch_size, num_views, 3, 3)
        return torch.matmul(obj_points_rest, rotmat.transpose(-1, -2)) + trans_abs.unsqueeze(2)

    def _build_gt_object_points(self, batch, dtype, device):
        gt_obj_points = batch.get("target_obj_pc_sparse", None)
        if gt_obj_points is not None:
            return gt_obj_points.to(device=device, dtype=dtype)
        gt_obj_transform = batch.get("target_obj_transform", None)
        gt_obj_rotmat = batch.get("target_R_label", None)
        if gt_obj_transform is None or gt_obj_rotmat is None:
            return None
        obj_rest = batch["master_obj_sparse_rest"].to(device=device, dtype=dtype)
        gt_rot6d = rotmat_to_rot6d(gt_obj_rotmat.reshape(-1, 3, 3)).view(gt_obj_rotmat.shape[0], gt_obj_rotmat.shape[1], 6)
        gt_trans = gt_obj_transform[..., :3, 3]
        return self._build_object_points_from_pose(
            obj_rest,
            gt_rot6d.to(device=device, dtype=dtype),
            gt_trans.to(device=device, dtype=dtype),
        )

    @staticmethod
    def _bidirectional_chamfer_distance(pred_points, target_points):
        pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=10.0, neginf=-10.0).float()
        target_points = torch.nan_to_num(target_points, nan=0.0, posinf=10.0, neginf=-10.0).float()
        pairwise = torch.cdist(pred_points, target_points)
        pred_to_target = pairwise.min(dim=-1)[0].mean(dim=-1)
        target_to_pred = pairwise.min(dim=-2)[0].mean(dim=-1)
        return 0.5 * (pred_to_target + target_to_pred)

    def _stage1_rot6d_l1_weight(self, chamfer_m):
        tau = max(float(self.stage1_obj_rot6d_l1_cd_tau_m), 1e-6)
        weight = (chamfer_m / tau).clamp_(0.0, 1.0)
        power = max(float(self.stage1_obj_rot6d_l1_mask_power), 1e-6)
        if abs(power - 1.0) > 1e-6:
            weight = weight.pow(power)
        return weight

    def _build_pred_object_points_for_eval(self, preds, batch):
        dtype = preds["mano_3d_mesh_sv"].dtype
        device = preds["mano_3d_mesh_sv"].device
        obj_rest = batch["master_obj_sparse_rest"].to(device=device, dtype=dtype)
        return self._build_object_points_from_pose(
            obj_rest,
            preds["obj_pose_rot6d_cam"],
            preds["obj_pose_trans_cam"],
        )

    def _build_pred_object_kp21_for_eval(self, preds, batch):
        dtype = preds["mano_3d_mesh_sv"].dtype
        device = preds["mano_3d_mesh_sv"].device
        obj_rest = batch["master_obj_kp21_rest"].to(device=device, dtype=dtype)
        return self._build_object_points_from_pose(
            obj_rest,
            preds["obj_pose_rot6d_cam"],
            preds["obj_pose_trans_cam"],
        )

    def _extract_object_id(self, batch, device):
        obj_id = batch.get("obj_id", None)
        if obj_id is None:
            return None
        if not torch.is_tensor(obj_id):
            obj_id = torch.as_tensor(obj_id, device=device)
        obj_id = obj_id.to(device=device, dtype=torch.long)
        while obj_id.dim() > 1:
            obj_id = obj_id[..., 0]
        obj_id = obj_id.clamp_(min=0, max=max(self.num_obj_classes - 1, 0))
        return obj_id

    def _extract_master_id(self, batch, batch_size, device):
        master_id = batch.get("master_id", None)
        if master_id is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        if not torch.is_tensor(master_id):
            master_id = torch.as_tensor(master_id, device=device)
        master_id = master_id.to(device=device, dtype=torch.long)
        while master_id.dim() > 1:
            master_id = master_id[..., 0]
        return master_id.clamp_(min=0)

    def _apply_object_identity_conditioning(self, obj_feat, batch, batch_size, num_views):
        obj_id = self._extract_object_id(batch, obj_feat.device)
        if obj_id is None:
            zero = obj_feat.new_zeros(batch_size, self.obj_id_embed_dim)
            cond_embed = zero
        else:
            cond_embed = self.obj_id_embed(obj_id)
        gamma_beta = self.obj_id_film(cond_embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma).repeat_interleave(num_views, dim=0).view(-1, self.shared_dim, 1, 1)
        beta = 0.1 * torch.tanh(beta).repeat_interleave(num_views, dim=0).view(-1, self.shared_dim, 1, 1)
        obj_feat = obj_feat * (1.0 + gamma) + beta
        return obj_feat, cond_embed

    def _decode_object_keypoints(self, obj_feat, image_hw, obj_token):
        img_h, img_w = image_hw
        obj_heatmap = self.obj_2d_head(obj_feat)
        uv_peak_conf = self._heatmap_peak_confidence(obj_heatmap)
        uv_pdf = obj_heatmap.reshape(*obj_heatmap.shape[:2], -1)
        uv_pdf = uv_pdf / (uv_pdf.sum(dim=-1, keepdim=True) + 1e-6)
        uv_pdf = uv_pdf.contiguous().view(*obj_heatmap.shape)
        uv_coord = integral_heatmap2d(uv_pdf)
        uv_spread_conf = self._heatmap_spread_confidence(uv_pdf, uv_coord, (img_h, img_w), tau_px=self.conf_obj_tau_px)
        uv_heat_conf = torch.sqrt(torch.clamp(uv_peak_conf * uv_spread_conf, min=0.0))
        uv_heat_conf = torch.nan_to_num(uv_heat_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        uv_vis_conf = self.obj_vis_head(obj_token)
        uv_base_conf = torch.sqrt(torch.clamp(uv_heat_conf * uv_vis_conf, min=0.0))
        uv_base_conf = torch.nan_to_num(uv_base_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        uv_coord_pixel = torch.einsum(
            "bij,j->bij",
            uv_coord,
            torch.tensor([img_w, img_h], dtype=uv_coord.dtype, device=uv_coord.device),
        )
        uv_coord_norm = uv_coord * 2.0 - 1.0
        return obj_heatmap, uv_coord_norm, uv_coord_pixel, uv_heat_conf, uv_vis_conf, uv_base_conf

    def _solve_object_pnp(self, obj_kp2d_pixel, obj_kp_conf, obj_kp3d_rest, cam_intr):
        batch_size, num_views, _, _ = obj_kp2d_pixel.shape
        device = obj_kp2d_pixel.device
        dtype = obj_kp2d_pixel.dtype
        pred_rot6d = torch.zeros(batch_size, num_views, 6, device=device, dtype=dtype)
        pred_rot6d[..., 0] = 1.0
        pred_rot6d[..., 4] = 1.0
        pred_trans = torch.zeros(batch_size, num_views, 3, device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, num_views, device=device, dtype=dtype)

        kp2d_np = obj_kp2d_pixel.detach().cpu().numpy()
        conf_np = obj_kp_conf.detach().cpu().numpy()[..., 0]
        kp3d_np = obj_kp3d_rest.detach().cpu().numpy()
        cam_np = cam_intr.detach().cpu().numpy()

        for b_idx in range(batch_size):
            object_points_all = np.asarray(kp3d_np[b_idx], dtype=np.float32)
            for v_idx in range(num_views):
                image_points_all = np.asarray(kp2d_np[b_idx, v_idx], dtype=np.float32)
                conf_all = np.asarray(conf_np[b_idx, v_idx], dtype=np.float32)
                valid = (
                    np.isfinite(image_points_all).all(axis=1)
                    & np.isfinite(object_points_all).all(axis=1)
                    & (conf_all >= float(self.obj_pnp_min_conf))
                )
                if valid.sum() < max(self.obj_pnp_min_points, 4):
                    continue

                valid_indices = np.where(valid)[0]
                conf_valid = conf_all[valid_indices]
                if valid_indices.shape[0] > self.obj_pnp_topk:
                    order = np.argsort(-conf_valid)
                    valid_indices = valid_indices[order[:self.obj_pnp_topk]]

                if valid_indices.shape[0] < 4:
                    continue

                object_points = object_points_all[valid_indices]
                image_points = image_points_all[valid_indices]
                camera_matrix = np.asarray(cam_np[b_idx, v_idx], dtype=np.float32)
                success = False
                rvec = None
                tvec = None
                try:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=object_points,
                        imagePoints=image_points,
                        cameraMatrix=camera_matrix,
                        distCoeffs=None,
                        flags=cv2.SOLVEPNP_EPNP,
                        reprojectionError=float(self.obj_pnp_reproj_px),
                        iterationsCount=100,
                        confidence=0.99,
                    )
                    if success and inliers is not None and len(inliers) >= 4:
                        inlier_points = object_points[inliers[:, 0]]
                        inlier_image = image_points[inliers[:, 0]]
                        success, rvec, tvec = cv2.solvePnP(
                            objectPoints=inlier_points,
                            imagePoints=inlier_image,
                            cameraMatrix=camera_matrix,
                            distCoeffs=None,
                            rvec=rvec,
                            tvec=tvec,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                except cv2.error:
                    success = False

                if not success:
                    try:
                        success, rvec, tvec = cv2.solvePnP(
                            objectPoints=object_points,
                            imagePoints=image_points,
                            cameraMatrix=camera_matrix,
                            distCoeffs=None,
                            flags=cv2.SOLVEPNP_EPNP,
                        )
                    except cv2.error:
                        success = False

                if not success:
                    continue

                rotmat, _ = cv2.Rodrigues(rvec)
                proj_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
                proj_points = proj_points.reshape(-1, 2)
                reproj_err = np.linalg.norm(proj_points - image_points, axis=1).mean()
                trans_norm = float(np.linalg.norm(tvec.reshape(3)))
                if (
                    not np.isfinite(rotmat).all()
                    or not np.isfinite(tvec).all()
                    or float(tvec[2]) <= 1e-6
                    or trans_norm > self.obj_pnp_max_trans_m
                    or reproj_err > self.obj_pnp_max_reproj_px
                ):
                    continue

                rotmat = torch.from_numpy(rotmat).to(device=device, dtype=dtype).unsqueeze(0)
                pred_rot6d[b_idx, v_idx] = rotmat_to_rot6d(rotmat).squeeze(0)
                pred_trans[b_idx, v_idx] = torch.from_numpy(tvec.reshape(3)).to(device=device, dtype=dtype)
                valid_mask[b_idx, v_idx] = 1.0

        return pred_rot6d, pred_trans, valid_mask

    def _camera_pose_to_master(self, rot6d_cam, trans_cam, cam_to_master_extr):
        batch_size, num_views = rot6d_cam.shape[:2]
        cam_to_master_extr = cam_to_master_extr.to(device=rot6d_cam.device, dtype=rot6d_cam.dtype)
        cam_to_master_rot = cam_to_master_extr[..., :3, :3]
        cam_to_master_trans = cam_to_master_extr[..., :3, 3]
        rotmat_cam = rot6d_to_rotmat(rot6d_cam.reshape(-1, 6)).view(batch_size, num_views, 3, 3)
        rotmat_master = torch.matmul(cam_to_master_rot, rotmat_cam)
        trans_master = torch.matmul(cam_to_master_rot, trans_cam.unsqueeze(-1)).squeeze(-1) + cam_to_master_trans
        rot6d_master = rotmat_to_rot6d(rotmat_master.reshape(-1, 3, 3)).view(batch_size, num_views, 6)
        return rot6d_master, trans_master

    def _compute_object_pose_consistency_gate(self, rot6d_cam, trans_cam, valid_mask, cam_to_master_extr):
        batch_size, num_views = rot6d_cam.shape[:2]
        dtype = rot6d_cam.dtype
        device = rot6d_cam.device
        valid = valid_mask > 0.5
        zero = rot6d_cam.new_zeros(batch_size, num_views)
        if num_views <= 1:
            return valid_mask.to(dtype=dtype), zero, zero

        rot6d_master, trans_master = self._camera_pose_to_master(rot6d_cam, trans_cam, cam_to_master_extr)
        rotmat_master = rot6d_to_rotmat(rot6d_master.reshape(-1, 6)).view(batch_size, num_views, 3, 3)
        rel_rot = torch.matmul(
            rotmat_master.unsqueeze(2),
            rotmat_master.unsqueeze(1).transpose(-1, -2),
        )
        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        pair_rot_deg = torch.rad2deg(torch.acos(cos_theta))
        pair_trans_m = torch.linalg.norm(trans_master.unsqueeze(2) - trans_master.unsqueeze(1), dim=-1)

        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(num_views, device=device, dtype=torch.bool).unsqueeze(0)
        pair_valid = pair_valid & (~eye)
        pair_cost = pair_rot_deg / max(self.obj_pose_gate_rot_tau_deg, 1e-6) + pair_trans_m / max(self.obj_pose_gate_trans_tau_m, 1e-6)
        pair_cost = torch.where(pair_valid, pair_cost, torch.zeros_like(pair_cost))
        denom = pair_valid.to(dtype).sum(dim=-1).clamp_min(1.0)
        mean_cost = pair_cost.sum(dim=-1) / denom
        mean_cost = torch.where(valid, mean_cost, torch.full_like(mean_cost, 1e6))
        ref_idx = mean_cost.argmin(dim=-1)

        ref_rot = rot6d_master[torch.arange(batch_size, device=device), ref_idx]
        ref_trans = trans_master[torch.arange(batch_size, device=device), ref_idx]
        rot_err_deg = torch.rad2deg(
            self.rotation_geodesic(rot6d_master.reshape(-1, 6), ref_rot.unsqueeze(1).expand(-1, num_views, -1).reshape(-1, 6))
        ).view(batch_size, num_views)
        trans_err_m = torch.linalg.norm(trans_master - ref_trans.unsqueeze(1), dim=-1)
        pose_gate = torch.exp(-rot_err_deg / max(self.obj_pose_gate_rot_tau_deg, 1e-6))
        pose_gate = pose_gate * torch.exp(-trans_err_m / max(self.obj_pose_gate_trans_tau_m, 1e-6))
        pose_gate = torch.where(valid, pose_gate, torch.zeros_like(pose_gate))
        pose_gate = torch.nan_to_num(pose_gate, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return pose_gate, rot_err_deg, trans_err_m

    def _predict_object_pose_gate(self, obj_token_views, obj_id_embed, obj_pose_valid_base):
        batch_size, num_views, _ = obj_token_views.shape
        obj_id_feat = obj_id_embed.unsqueeze(1).expand(-1, num_views, -1)
        pose_feat = torch.cat((obj_token_views, obj_id_feat, obj_pose_valid_base.unsqueeze(-1)), dim=-1)
        return self.obj_pose_gate_head(pose_feat)

    def _get_gt_object_pose_abs(self, gt, dtype, device):
        gt_rotmat = gt.get("target_R_label", None)
        gt_transform = gt.get("target_obj_transform", None)
        if gt_rotmat is None or gt_transform is None:
            return None, None
        gt_rotmat = gt_rotmat.to(device=device, dtype=dtype)
        gt_rot6d = rotmat_to_rot6d(gt_rotmat.reshape(-1, 3, 3)).view(gt_rotmat.shape[0], gt_rotmat.shape[1], 6)
        gt_trans = gt_transform[..., :3, 3].to(device=device, dtype=dtype)
        return gt_rot6d, gt_trans

    @staticmethod
    def _stabilize_triangulation_confidence(conf, valid_mask):
        conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        valid_mask = valid_mask.to(dtype=conf.dtype)
        conf = conf * valid_mask
        conf_sum = conf.sum(dim=1, keepdim=True)
        uniform = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fallback_mask = conf_sum <= 1e-6
        conf = torch.where(fallback_mask.expand_as(conf), uniform, conf)
        return conf

    def _triangulate_with_reprojection_confidence(self, kp2d_pixel, cam_intr, target_cam_extr, tau_px, init_conf=None):
        kp2d_pixel = torch.nan_to_num(kp2d_pixel, nan=0.0, posinf=1e6, neginf=-1e6).float()
        cam_intr = cam_intr.detach().to(dtype=kp2d_pixel.dtype)
        extr_m2c = torch.linalg.inv(target_cam_extr.detach()).to(dtype=kp2d_pixel.dtype)

        valid_obs = torch.isfinite(kp2d_pixel).all(dim=-1)
        tri_master = batch_triangulate_dlt_torch(kp2d_pixel, cam_intr, extr_m2c)
        reproj_pixel = None
        reproj_error = None
        if init_conf is None:
            conf = valid_obs.to(dtype=kp2d_pixel.dtype)
        else:
            conf = torch.nan_to_num(init_conf, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=kp2d_pixel.dtype)
            conf = conf * valid_obs.to(dtype=kp2d_pixel.dtype)

        for _ in range(max(self.conf_tri_refine_iters, 1)):
            conf = self._stabilize_triangulation_confidence(conf, valid_obs)
            tri_master = batch_triangulate_dlt_torch_confidence(kp2d_pixel, cam_intr, extr_m2c, confidences=conf)
            tri_master_views = tri_master.unsqueeze(1).expand(-1, kp2d_pixel.shape[1], -1, -1)
            tri_cam = batch_cam_extr_transf(extr_m2c, tri_master_views)
            reproj_pixel = batch_cam_intr_projection(cam_intr, tri_cam)
            reproj_error = torch.linalg.norm(reproj_pixel - kp2d_pixel, dim=-1)
            positive_depth = tri_cam[..., 2] > 1e-6
            conf = torch.exp(-reproj_error / float(tau_px)) * positive_depth.to(dtype=kp2d_pixel.dtype)

        conf = self._stabilize_triangulation_confidence(conf, valid_obs)
        return tri_master, reproj_pixel, reproj_error, conf

    @staticmethod
    def loss_proj_to_multicam(pred_points, cam_to_view_extr, cam_intr, gt_points_2d, img_scale, visibility=None):
        pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=img_scale, neginf=-img_scale)
        gt_valid = torch.isfinite(gt_points_2d).all(dim=-1)
        gt_points_2d = torch.nan_to_num(gt_points_2d, nan=0.0, posinf=img_scale, neginf=-img_scale)
        pred_points = pred_points.unsqueeze(1).expand(-1, cam_intr.shape[1], -1, -1)
        pred_points_cam = batch_cam_extr_transf(cam_to_view_extr, pred_points)
        pred_points_2d = batch_cam_intr_projection(cam_intr, pred_points_cam)
        pred_points_2d = torch.nan_to_num(pred_points_2d, nan=0.0, posinf=img_scale, neginf=-img_scale)
        proj_error = torch.sum(torch.pow((pred_points_2d - gt_points_2d) / img_scale, 2), dim=-1)
        valid_mask = gt_valid.to(dtype=proj_error.dtype)
        if visibility is not None:
            valid_mask = valid_mask * visibility.to(dtype=proj_error.dtype)
        return (proj_error * valid_mask).sum() / (valid_mask.sum() + 1e-9)

    def _recover_master_object_pose_from_tri(self, obj_kp_rest, obj_kp_master, hand_joints_master):
        hand_root = hand_joints_master[:, self.center_idx, :]
        pose_dict = pose_from_keypoints(obj_kp_rest, obj_kp_master, hand_root=hand_root)
        return pose_dict["rot6d"], pose_dict["trans_rel"], pose_dict["trans_abs"]

    def _forward_impl(self, batch, **kwargs):
        img = batch["image"]
        batch_size, num_cams = img.shape[:2]
        img_h, img_w = img.shape[-2:]

        img_feats, global_feat = self.extract_img_feat(img)
        mlvl_feat = self.feat_decode(img_feats)
        bn = mlvl_feat.shape[0]

        shared_feat = self.shared_stem(mlvl_feat)
        master_id = self._extract_master_id(batch, batch_size, shared_feat.device).clamp(max=max(num_cams - 1, 0))
        obj_mv_view_weights = shared_feat.new_full((batch_size, num_cams), 1.0 / max(num_cams, 1))
        obj_mv_view_gate = shared_feat.new_ones(batch_size, num_cams, self.shared_dim)
        obj_mv_master_weight = shared_feat.new_zeros(batch_size)
        if self.shared_mv_fusion is not None and num_cams > 1:
            shared_feat_mv = shared_feat.view(batch_size, num_cams, self.shared_dim, shared_feat.shape[-2], shared_feat.shape[-1])
            shared_token_seed, _ = self.shared_pool(shared_feat)
            shared_token_views = shared_token_seed.view(batch_size, num_cams, -1)
            shared_feat_mv, shared_mv_aux = self.shared_mv_fusion(
                shared_feat_mv,
                shared_token_views,
                batch["target_cam_intr"].to(device=shared_feat.device, dtype=shared_feat.dtype),
                batch["target_cam_extr"].to(device=shared_feat.device, dtype=shared_feat.dtype),
                master_id,
            )
            shared_feat = shared_feat_mv.flatten(0, 1)
            obj_mv_view_weights = shared_mv_aux["view_weights"]
            obj_mv_view_gate = shared_mv_aux["view_gate"]
            batch_idx = torch.arange(batch_size, device=shared_feat.device)
            obj_mv_master_weight = obj_mv_view_weights[batch_idx, master_id]
        shared_feat_stage2 = shared_feat.view(batch_size, num_cams, self.shared_dim, shared_feat.shape[-2], shared_feat.shape[-1])
        hand_feat = self.hand_adapter(shared_feat)
        obj_feat = self.obj_adapter(shared_feat)
        obj_feat, obj_id_embed = self._apply_object_identity_conditioning(obj_feat, batch, batch_size, num_cams)
        hand_ref_memory = hand_feat.detach() if self.stage1_obj_hand_ref_detach else hand_feat
        obj_feat, obj_hand_ref_gate = self.obj_query_hand_ref(obj_feat, hand_ref_memory)

        hand_tokens = self.hot_pose(hand_feat)
        hand_att0 = self.att_0(hand_tokens)
        hand_att1 = self.att_1(hand_att0)
        hand_context = self.hand_context_proj(hand_att1.mean(dim=1))
        hand_fused = self.mano_fuse(hand_att1)
        hand_fused_flat = hand_fused.view(bn, -1)

        mano_params = self.mano_fc(hand_fused_flat)
        pred_hand_pose = mano_params[:, :96]
        pred_shape = mano_params[:, 96:106]
        pred_cam = mano_params[:, 106:]
        pred_cam = torch.cat((F.relu(pred_cam[:, 0:1]), pred_cam[:, 1:]), dim=1)
        coord_xyz_sv, coord_uv_sv, pose_euler_sv, shape_sv, cam_sv = self.mano_decoder(pred_hand_pose, pred_shape, pred_cam)
        coord_xyz_sv = coord_xyz_sv * (self.data_preset_cfg.BBOX_3D_SIZE / 2.0)
        pred_hand_jts = coord_uv_sv[:, self.num_hand_verts:]
        pred_hand_jts_pixel = self._normed_uv_to_pixel(pred_hand_jts, (img_h, img_w))

        obj_token, _ = self.obj_pool(obj_feat)
        obj_token_views = obj_token.view(batch_size, num_cams, -1)
        (
            obj_heatmap,
            pred_obj_kp2d_norm_raw_bn,
            pred_obj_kp2d_pixel_raw_bn,
            pred_obj_kp_heat_conf_bn,
            pred_obj_kp_vis_conf_bn,
            pred_obj_kp_base_conf_bn,
        ) = self._decode_object_keypoints(
            obj_feat,
            (img_h, img_w),
            obj_token,
        )
        pred_obj_kp2d_norm_raw = pred_obj_kp2d_norm_raw_bn.view(batch_size, num_cams, self.num_obj_joints, 2)
        pred_obj_kp2d_pixel_raw = pred_obj_kp2d_pixel_raw_bn.view(batch_size, num_cams, self.num_obj_joints, 2)
        pred_obj_kp_heat_conf = pred_obj_kp_heat_conf_bn.view(batch_size, num_cams, self.num_obj_joints, 1)
        pred_obj_kp_vis_conf = pred_obj_kp_vis_conf_bn.view(batch_size, num_cams, self.num_obj_joints, 1)
        pred_obj_kp_base_conf = pred_obj_kp_base_conf_bn.view(batch_size, num_cams, self.num_obj_joints, 1)
        obj_kp21_rest = batch["master_obj_kp21_rest"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype)
        pred_obj_rot6d_base_sv, pred_obj_trans_base_sv, pred_obj_pose_valid_base = self._solve_object_pnp(
            pred_obj_kp2d_pixel_raw,
            pred_obj_kp_base_conf,
            obj_kp21_rest,
            batch["target_cam_intr"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype),
        )
        obj_kp3d_tri_master, pred_obj_kp2d_reproj, pred_obj_kp_reproj_error, pred_obj_kp_tri_conf = self._triangulate_with_reprojection_confidence(
            pred_obj_kp2d_pixel_raw,
            batch["target_cam_intr"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype),
            batch["target_cam_extr"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype),
            tau_px=self.conf_obj_tri_tau_px,
            init_conf=pred_obj_kp_base_conf.squeeze(-1),
        )
        pred_obj_kp2d_reproj = torch.where(
            torch.isfinite(pred_obj_kp2d_reproj),
            pred_obj_kp2d_reproj,
            pred_obj_kp2d_pixel_raw,
        )
        obj_pose_gate, obj_pose_gate_rot_err, obj_pose_gate_trans_err = self._compute_object_pose_consistency_gate(
            pred_obj_rot6d_base_sv,
            pred_obj_trans_base_sv,
            pred_obj_pose_valid_base,
            batch["target_cam_extr"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype),
        )
        pred_obj_pose_gate_learned = self._predict_object_pose_gate(obj_token_views, obj_id_embed, pred_obj_pose_valid_base)
        obj_pose_gate = torch.sqrt(torch.clamp(pred_obj_pose_gate_learned.squeeze(-1) * obj_pose_gate, min=0.0, max=1.0))
        obj_pose_gate_kp = obj_pose_gate.unsqueeze(-1).unsqueeze(-1)
        pred_obj_kp_tri_conf = pred_obj_kp_tri_conf.unsqueeze(-1)
        pred_obj_kp_final_conf = torch.pow(
            torch.clamp(pred_obj_kp_heat_conf, min=1e-6)
            * torch.clamp(pred_obj_kp_vis_conf, min=1e-6)
            * torch.clamp(pred_obj_kp_tri_conf, min=1e-6)
            * torch.clamp(obj_pose_gate_kp, min=1e-6),
            0.25,
        )
        pred_obj_kp_final_conf = torch.nan_to_num(pred_obj_kp_final_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        pred_obj_kp_refine_gate = torch.sqrt(torch.clamp(pred_obj_kp_tri_conf * obj_pose_gate_kp, min=0.0, max=1.0))
        pred_obj_kp2d_pixel = pred_obj_kp2d_pixel_raw * (1.0 - pred_obj_kp_refine_gate) + pred_obj_kp2d_reproj * pred_obj_kp_refine_gate
        pred_obj_kp2d_norm = pred_obj_kp2d_pixel / pred_obj_kp2d_pixel.new_tensor([img_w, img_h]).view(1, 1, 1, 2) * 2.0 - 1.0
        pred_obj_rot6d_sv, pred_obj_trans_sv, pred_obj_pose_valid = self._solve_object_pnp(
            pred_obj_kp2d_pixel,
            pred_obj_kp_final_conf,
            obj_kp21_rest,
            batch["target_cam_intr"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype),
        )
        pred_obj_points_abs_sv = self._build_pred_object_points_for_eval(
            {
                "mano_3d_mesh_sv": coord_xyz_sv,
                "obj_pose_rot6d_cam": pred_obj_rot6d_sv,
                "obj_pose_trans_cam": pred_obj_trans_sv,
            },
            batch,
        )
        mano_mesh_sv = coord_xyz_sv.view(batch_size, num_cams, self.num_hand_verts + self.num_hand_joints, 3)
        pred_hand_jts_pixel_views = pred_hand_jts_pixel.view(batch_size, num_cams, self.num_hand_joints, 2)

        tri_hand_master, tri_hand_reproj_pixel, tri_hand_reproj_error, conf_hand_tri = self._triangulate_with_reprojection_confidence(
            pred_hand_jts_pixel_views,
            batch["target_cam_intr"].to(device=pred_hand_jts_pixel.device, dtype=pred_hand_jts_pixel.dtype),
            batch["target_cam_extr"].to(device=pred_hand_jts_pixel.device, dtype=pred_hand_jts_pixel.dtype),
            tau_px=self.conf_tri_hand_tau_px,
        )

        sv_preds = {
            "pred_hand": pred_hand_jts,
            "pred_hand_pixel": pred_hand_jts_pixel,
            "mano_3d_mesh_sv": coord_xyz_sv,
            "mano_2d_mesh_sv": coord_uv_sv,
            "mano_pose_euler_sv": pose_euler_sv,
            "mano_shape_sv": shape_sv,
            "mano_cam_sv": cam_sv,
            "pred_obj_kp2d_norm_raw": pred_obj_kp2d_norm_raw,
            "pred_obj_kp2d_pixel_raw": pred_obj_kp2d_pixel_raw,
            "pred_obj_kp2d_norm": pred_obj_kp2d_norm,
            "pred_obj_kp2d_pixel": pred_obj_kp2d_pixel,
            "pred_obj_kp_heat_conf": pred_obj_kp_heat_conf,
            "pred_obj_kp_vis_conf": pred_obj_kp_vis_conf,
            "pred_obj_kp_conf_base": pred_obj_kp_base_conf,
            "pred_obj_kp_conf_tri": pred_obj_kp_tri_conf,
            "pred_obj_kp_conf": pred_obj_kp_final_conf,
            "pred_obj_kp_refine_gate": pred_obj_kp_refine_gate,
            "pred_obj_kp2d_reproj": pred_obj_kp2d_reproj,
            "pred_obj_kp_reproj_error": pred_obj_kp_reproj_error,
            "pred_obj_heatmap": obj_heatmap.view(batch_size, num_cams, self.num_obj_joints, obj_heatmap.shape[-2], obj_heatmap.shape[-1]),
            "obj_pose_rot6d_cam_base": pred_obj_rot6d_base_sv,
            "obj_pose_trans_cam_base": pred_obj_trans_base_sv,
            "obj_pose_valid_base": pred_obj_pose_valid_base,
            "obj_pose_rot6d_cam": pred_obj_rot6d_sv,
            "obj_pose_trans_cam": pred_obj_trans_sv,
            "obj_pose_valid": pred_obj_pose_valid,
            "pred_obj_pose_gate_learned": pred_obj_pose_gate_learned,
            "obj_pose_gate": obj_pose_gate,
            "obj_pose_gate_rot_err": obj_pose_gate_rot_err,
            "obj_pose_gate_trans_err": obj_pose_gate_trans_err,
            "obj_tri_kp_master": obj_kp3d_tri_master,
            "obj_mv_view_weights": obj_mv_view_weights,
            "obj_mv_view_gate": obj_mv_view_gate,
            "obj_mv_master_weight": obj_mv_master_weight,
            "obj_points_cam": pred_obj_points_abs_sv,
            "pred_hand_pixel_views": pred_hand_jts_pixel_views,
            "tri_hand_joints_master": tri_hand_master,
            "tri_hand_reproj_pixel": tri_hand_reproj_pixel,
            "tri_hand_reproj_error": tri_hand_reproj_error,
            "conf_hand_tri": conf_hand_tri,
            "reference_hand": mano_mesh_sv.detach(),
            "reference_obj": pred_obj_points_abs_sv.detach(),
            "shared_feat": shared_feat,
            "hand_context": hand_context,
            "obj_token": obj_token,
            "obj_attn_map": obj_attn_map,
            "obj_hand_ref_gate": obj_hand_ref_gate.view(
                batch_size,
                num_cams,
                self.shared_dim,
                obj_hand_ref_gate.shape[-2],
                obj_hand_ref_gate.shape[-1],
            ),
            "obj_id_embed": obj_id_embed,
        }
        interaction_mode = kwargs.get("interaction_mode", self._stage_to_interaction_mode(self.current_stage_name))
        if interaction_mode not in {"hand", "ho"}:
            return sv_preds

        obj_view_rot6d_master_abs, obj_view_trans_master_abs = self._camera_pose_to_master(
            pred_obj_rot6d_sv,
            pred_obj_trans_sv,
            batch["target_cam_extr"].to(device=coord_xyz_sv.device, dtype=coord_xyz_sv.dtype),
        )
        obj_init_rot6d, obj_init_trans_rel, obj_init_trans_abs = self._recover_master_object_pose_from_tri(
            obj_kp21_rest,
            obj_kp3d_tri_master,
            tri_hand_master,
        )
        hand_root_master = tri_hand_master[:, self.center_idx, :]
        obj_init_rot6d = torch.nan_to_num(obj_init_rot6d, nan=0.0, posinf=10.0, neginf=-10.0)
        obj_init_trans_rel = torch.nan_to_num(obj_init_trans_rel, nan=0.0, posinf=10.0, neginf=-10.0)
        obj_init_trans_abs = torch.nan_to_num(obj_init_trans_abs, nan=0.0, posinf=10.0, neginf=-10.0)
        obj_view_rot6d_master_abs = torch.nan_to_num(obj_view_rot6d_master_abs, nan=0.0, posinf=10.0, neginf=-10.0)
        obj_view_trans_master_rel = torch.nan_to_num(
            obj_view_trans_master_abs - hand_root_master.unsqueeze(1),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        sv_rot_outputs = {
            "obj_view_rot6d_cam": pred_obj_rot6d_sv,
            "obj_view_rot6d_master": obj_view_rot6d_master_abs,
            "obj_view_trans": pred_obj_trans_sv,
            "obj_view_trans_master": obj_view_trans_master_rel,
            "obj_fused_rot6d_sv": obj_init_rot6d,
            "obj_fused_trans_sv": obj_init_trans_rel,
            "obj_init_rot6d": obj_init_rot6d,
            "obj_init_trans": obj_init_trans_rel,
            "obj_init_trans_abs": obj_init_trans_abs,
        }

        img_metas = {
            "inp_img_shape": img.shape[-2:],
            "cam_intr": batch["target_cam_intr"],
            "cam_extr": batch["target_cam_extr"],
            "master_id": batch["master_id"],
            "cam_view_num": num_cams,
        }
        pt_preds = self.ptEmb_head(
            mlvl_feat=shared_feat_stage2,
            mano_3d_sv=coord_xyz_sv,
            img_metas=img_metas,
            reference_hand=tri_hand_master.detach(),
            reference_obj=None,
            obj_template=batch["master_obj_sparse_rest"],
            hand_view_conf=conf_hand_tri.detach(),
            sv_rot_outputs=sv_rot_outputs,
            obj_init_rot6d=obj_init_rot6d.detach(),
            obj_init_trans=obj_init_trans_rel.detach(),
            interaction_mode=interaction_mode,
            stage2_warmup=self.current_stage2_warmup,
        )

        final_preds = dict(sv_preds)
        final_preds.update({
            "ref_hand": tri_hand_master,
            "interaction_mode": pt_preds.get("interaction_mode", interaction_mode),
            "obj_view_rot6d_master": sv_rot_outputs["obj_view_rot6d_master"],
            "obj_view_trans_master": sv_rot_outputs["obj_view_trans_master"],
            "obj_init_rot6d": pt_preds.get("obj_init_rot6d", obj_init_rot6d),
            "obj_init_trans": pt_preds.get("obj_init_trans", obj_init_trans_rel),
            "obj_fused_rot6d_sv": sv_rot_outputs["obj_fused_rot6d_sv"],
            "obj_fused_trans_sv": sv_rot_outputs["obj_fused_trans_sv"],
            "hand_mesh_xyz_master": pt_preds["all_hand_mesh_xyz_master"],
            "obj_xyz_master": pt_preds["all_obj_xyz_master"],
            "obj_rot6d": pt_preds["all_obj_rot6d"],
            "obj_trans": pt_preds["all_obj_trans"],
            "pred_obj_trans_master": pt_preds["pred_obj_trans_master"],
            "mano_3d_mesh_master": pt_preds["pred_kp_mano_mesh_xyz_master"],
            "mano_3d_mesh_kp_master": pt_preds["pred_kp_mano_mesh_xyz_master"],
            "pred_kp_pose": pt_preds["pred_kp_mano_params_master"]["pose_euler"],
            "pred_kp_shape": pt_preds["pred_kp_mano_params_master"]["shape"],
            "all_hand_joints_xyz_master": pt_preds.get("all_hand_joints_xyz_master", None),
        })
        return final_preds

    def compute_loss(self, preds, gt, stage_name="stage1", epoch_idx=None, **kwargs):
        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]
        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"]
        pred_pose_sv = preds["mano_pose_euler_sv"]
        pred_shape_sv = preds["mano_shape_sv"]
        obj_warmup = self._stage2_object_warmup(epoch_idx, stage_name) if epoch_idx is not None else float(stage_name == "stage2")

        hand_joints_sv_gt = gt["target_joints_uvd"].flatten(0, 1)
        gt_hand_joints_rel = gt.get("target_joints_3d_rel", None)
        joints_vis_mv = gt.get("target_joints_vis", None)
        joints_vis = joints_vis_mv.flatten(0, 1) if joints_vis_mv is not None else None

        pred_jts_2d = pred_mano_2d_mesh_sv[:, self.num_hand_verts:]
        gt_jts_2d = hand_joints_sv_gt[..., :2]
        if gt_hand_joints_rel is not None:
            gt_sv_ortho_cam = self._fit_ortho_camera(
                gt_hand_joints_rel[..., :2],
                gt["target_joints_uvd"][..., :2],
                visibility=joints_vis_mv,
            )
            gt_jts_2d = self._ortho_project_points(gt_hand_joints_rel, gt_sv_ortho_cam).flatten(0, 1)
        diff_2d = torch.abs(pred_jts_2d - gt_jts_2d)
        if joints_vis is not None:
            vis_mask = joints_vis.unsqueeze(-1)
            diff_2d = diff_2d * vis_mask
            valid_count = vis_mask.expand_as(diff_2d).sum()
            loss_hand_2d_sv = diff_2d.sum() / (valid_count + 1e-9)
        else:
            loss_hand_2d_sv = diff_2d.mean()

        loss_pose_reg_sv = self.coord_loss(pred_pose_sv[:, 3:], torch.zeros_like(pred_pose_sv[:, 3:])) * self.pose_reg_weight
        loss_shape_reg_sv = self.coord_loss(pred_shape_sv, torch.zeros_like(pred_shape_sv)) * self.shape_reg_weight

        gt_obj_kp21 = gt.get("target_obj_kp21", None)
        if gt_obj_kp21 is not None:
            gt_obj_kp21 = gt_obj_kp21.to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            gt_obj_kp2d_pixel = batch_cam_intr_projection(
                gt["target_cam_intr"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype),
                gt_obj_kp21,
            )
            img_scale = pred_mano_3d_mesh_sv.new_tensor([gt["image"].shape[-1], gt["image"].shape[-2]])
            gt_obj_kp2d_norm = gt_obj_kp2d_pixel / (img_scale.view(1, 1, 1, 2) / 2.0) - 1.0
            obj_valid = torch.isfinite(gt_obj_kp2d_norm).all(dim=-1) & (gt_obj_kp21[..., 2] > 1e-6)
            obj_diff = torch.abs(preds["pred_obj_kp2d_norm_raw"] - gt_obj_kp2d_norm)
            obj_valid_exp = obj_valid.unsqueeze(-1).to(dtype=obj_diff.dtype)
            loss_obj_2d = (obj_diff * obj_valid_exp).sum() / (obj_valid_exp.sum() + 1e-9)

            obj_err_px = torch.linalg.norm(preds["pred_obj_kp2d_pixel_raw"] - gt_obj_kp2d_pixel, dim=-1)
            obj_conf_target = torch.exp(-obj_err_px.detach() / max(self.conf_obj_tau_px, 1e-6))
            obj_conf_target = obj_conf_target * obj_valid.to(dtype=obj_conf_target.dtype)
            pred_obj_conf = preds["pred_obj_kp_vis_conf"].squeeze(-1)
            loss_obj_conf = (
                torch.abs(pred_obj_conf - obj_conf_target) * obj_valid.to(dtype=pred_obj_conf.dtype)
            ).sum() / (obj_valid.sum() + 1e-9)

            pred_obj_kp2d_reproj = preds["pred_obj_kp2d_reproj"]
            pred_obj_kp2d_reproj_norm = pred_obj_kp2d_reproj / (img_scale.view(1, 1, 1, 2) / 2.0) - 1.0
            tri_weight = preds["pred_obj_kp_conf_tri"].detach().squeeze(-1) * obj_valid.to(dtype=pred_mano_3d_mesh_sv.dtype)
            tri_diff = torch.abs(pred_obj_kp2d_reproj_norm - gt_obj_kp2d_norm)
            tri_weight_exp = tri_weight.unsqueeze(-1)
            loss_obj_tri_reproj = (tri_diff * tri_weight_exp).sum() / (tri_weight_exp.sum() + 1e-9)

            gt_obj_rot6d_gt, gt_obj_trans_gt = self._get_gt_object_pose_abs(
                gt,
                dtype=pred_mano_3d_mesh_sv.dtype,
                device=pred_mano_3d_mesh_sv.device,
            )
            if gt_obj_rot6d_gt is not None and gt_obj_trans_gt is not None:
                base_rot_err_deg = torch.rad2deg(
                    self.rotation_geodesic(
                        preds["obj_pose_rot6d_cam_base"].reshape(-1, 6),
                        gt_obj_rot6d_gt.reshape(-1, 6),
                    )
                ).view_as(preds["obj_pose_valid_base"])
                base_trans_err_m = torch.linalg.norm(
                    preds["obj_pose_trans_cam_base"] - gt_obj_trans_gt,
                    dim=-1,
                )
                obj_pose_gate_target = torch.exp(-base_rot_err_deg.detach() / max(self.obj_pose_loss_rot_tau_deg, 1e-6))
                obj_pose_gate_target = obj_pose_gate_target * torch.exp(-base_trans_err_m.detach() / max(self.obj_pose_loss_trans_tau_m, 1e-6))
                obj_pose_gate_target = obj_pose_gate_target * preds["obj_pose_valid_base"].detach()
                pred_obj_pose_gate = preds["pred_obj_pose_gate_learned"].squeeze(-1)
                loss_obj_pose_gate = torch.abs(pred_obj_pose_gate - obj_pose_gate_target).mean()
            else:
                loss_obj_pose_gate = pred_mano_3d_mesh_sv.new_tensor(0.0)
        else:
            loss_obj_2d = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_conf = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_tri_reproj = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_pose_gate = pred_mano_3d_mesh_sv.new_tensor(0.0)

        loss_obj_2d = loss_obj_2d * self.obj_2d_weight
        loss_obj_conf = loss_obj_conf * self.obj_conf_weight
        loss_obj_tri_reproj = loss_obj_tri_reproj * self.obj_tri_reproj_weight
        loss_obj_pose_gate = loss_obj_pose_gate * self.obj_pose_gate_weight
        loss_triang_hand = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_3d_jts = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_2d_proj = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_mano_proj = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_init_rot_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_init_rot_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_init_rot = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_init_trans = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_view_rot_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_view_rot_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_view_trans = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_view_points = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_view_rot = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_rot_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_rot_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_rot = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_trans = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_points = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_pose = pred_mano_3d_mesh_sv.new_tensor(0.0)

        if preds.get("ref_hand", None) is not None:
            master_hand_joints_gt = gt["master_joints_3d"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            loss_triang_hand = self.criterion_joints(preds["ref_hand"], master_hand_joints_gt) * self.triangulation_hand_weight

        if preds.get("hand_mesh_xyz_master", None) is not None:
            master_hand_joints_gt = gt["master_joints_3d"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            gt_joints_2d_mv = gt.get("target_joints_2d", None)
            if gt_joints_2d_mv is None:
                gt_joints_2d_mv = batch_cam_intr_projection(
                    gt["target_cam_intr"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype),
                    gt["target_joints_3d"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype),
                )
            gt_T_master_to_cam = torch.linalg.inv(
                gt["target_cam_extr"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            )
            img_scale = math.sqrt(float(gt["image"].shape[-1] ** 2 + gt["image"].shape[-2] ** 2))
            final_hand_mesh_master = preds["hand_mesh_xyz_master"][-1]
            final_hand_joints_master = final_hand_mesh_master[:, self.num_hand_verts:]
            loss_3d_jts = self.criterion_joints(final_hand_joints_master, master_hand_joints_gt) * self.decoder_hand_weight
            loss_2d_proj = self.loss_proj_to_multicam(
                final_hand_joints_master,
                gt_T_master_to_cam,
                gt["target_cam_intr"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype),
                gt_joints_2d_mv.to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype),
                img_scale,
                visibility=joints_vis_mv,
            ) * self.decoder_proj_weight
            if preds.get("mano_3d_mesh_kp_master", None) is not None:
                pred_kp_master = preds["mano_3d_mesh_kp_master"][:, self.num_hand_verts:]
                loss_mano_proj = self.criterion_joints(pred_kp_master, master_hand_joints_gt) * self.mano_proj_weight

        master_obj_rot6d_gt = gt.get("master_obj_rot6d_label", None)
        master_obj_trans_gt = gt.get("master_obj_t_label_rel", None)
        master_obj_sparse_gt = gt.get("master_obj_sparse", None)
        if master_obj_rot6d_gt is not None:
            master_obj_rot6d_gt = master_obj_rot6d_gt.to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
        if master_obj_trans_gt is not None:
            master_obj_trans_gt = master_obj_trans_gt.to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
        if master_obj_sparse_gt is not None:
            master_obj_sparse_gt = master_obj_sparse_gt.to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)

        pred_obj_init_rot6d = preds.get("obj_init_rot6d", None)
        pred_obj_init_trans = preds.get("obj_init_trans", None)
        if pred_obj_init_rot6d is not None and master_obj_rot6d_gt is not None:
            loss_obj_init_rot_geo = torch.mean(self.rotation_geodesic(pred_obj_init_rot6d, master_obj_rot6d_gt)) * self.obj_init_rot_weight
            loss_obj_init_rot_l1 = self.coord_loss(pred_obj_init_rot6d, master_obj_rot6d_gt) * self.obj_init_rot6d_weight
            loss_obj_init_rot = loss_obj_init_rot_geo + loss_obj_init_rot_l1
        if pred_obj_init_trans is not None and master_obj_trans_gt is not None:
            loss_obj_init_trans = self.coord_loss(pred_obj_init_trans, master_obj_trans_gt) * self.obj_init_trans_weight

        pred_obj_view_rot6d_master = preds.get("obj_view_rot6d_master", None)
        pred_obj_view_trans_master = preds.get("obj_view_trans_master", None)
        if pred_obj_view_rot6d_master is not None and master_obj_rot6d_gt is not None:
            target_view_rot = master_obj_rot6d_gt.unsqueeze(1).expand_as(pred_obj_view_rot6d_master)
            loss_obj_view_rot_geo = torch.mean(
                self.rotation_geodesic(pred_obj_view_rot6d_master.reshape(-1, 6), target_view_rot.reshape(-1, 6))
            ) * self.obj_view_rot_weight
        if pred_obj_view_trans_master is not None and master_obj_trans_gt is not None:
            target_view_trans = master_obj_trans_gt.unsqueeze(1).expand_as(pred_obj_view_trans_master)
            loss_obj_view_trans = self.coord_loss(pred_obj_view_trans_master, target_view_trans) * self.obj_view_trans_weight
        hand_root_master = None
        if preds.get("ref_hand", None) is not None:
            hand_root_master = preds["ref_hand"][:, self.center_idx, :]
        if (
            preds.get("obj_points_cam", None) is not None
            and gt.get("target_obj_pc_sparse", None) is not None
        ):
            pred_obj_points_cam = preds["obj_points_cam"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            gt_obj_points_cam = gt["target_obj_pc_sparse"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            view_cd = self._bidirectional_chamfer_distance(
                pred_obj_points_cam.flatten(0, 1),
                gt_obj_points_cam.flatten(0, 1),
            ).view(pred_obj_points_cam.shape[0], pred_obj_points_cam.shape[1])
            loss_obj_view_points = view_cd.mean() * self.obj_view_points_weight
            if pred_obj_view_rot6d_master is not None and master_obj_rot6d_gt is not None:
                target_view_rot = master_obj_rot6d_gt.unsqueeze(1).expand_as(pred_obj_view_rot6d_master)
                view_rot_l1 = torch.abs(pred_obj_view_rot6d_master - target_view_rot).mean(dim=-1)
                view_rot_l1_weight = self._stage1_rot6d_l1_weight(view_cd.detach())
                loss_obj_view_rot_l1 = (
                    (view_rot_l1 * view_rot_l1_weight).sum() / (view_rot_l1_weight.sum() + 1e-6)
                ) * self.obj_view_rot6d_weight * self.stage1_obj_rot6d_l1_scale
        elif pred_obj_view_rot6d_master is not None and master_obj_rot6d_gt is not None:
            target_view_rot = master_obj_rot6d_gt.unsqueeze(1).expand_as(pred_obj_view_rot6d_master)
            loss_obj_view_rot_l1 = (
                self.coord_loss(pred_obj_view_rot6d_master, target_view_rot)
                * self.obj_view_rot6d_weight
                * self.stage1_obj_rot6d_l1_scale
            )

        if (
            pred_obj_init_rot6d is not None
            and pred_obj_init_trans is not None
            and master_obj_rot6d_gt is not None
            and master_obj_trans_gt is not None
            and master_obj_sparse_gt is not None
            and hand_root_master is not None
        ):
            obj_rest_master = gt["master_obj_sparse_rest"].to(device=pred_mano_3d_mesh_sv.device, dtype=pred_mano_3d_mesh_sv.dtype)
            pred_obj_init_abs = hand_root_master + pred_obj_init_trans
            gt_obj_init_abs = hand_root_master + master_obj_trans_gt
            pred_init_points = self._build_object_points_from_pose(obj_rest_master, pred_obj_init_rot6d, pred_obj_init_abs)
            gt_init_points = self._build_object_points_from_pose(obj_rest_master, master_obj_rot6d_gt, gt_obj_init_abs)
            init_cd = self._bidirectional_chamfer_distance(pred_init_points, gt_init_points)
            init_rot_l1 = torch.abs(pred_obj_init_rot6d - master_obj_rot6d_gt).mean(dim=-1)
            init_rot_l1_weight = self._stage1_rot6d_l1_weight(init_cd.detach())
            loss_obj_init_rot_l1 = (
                (init_rot_l1 * init_rot_l1_weight).sum() / (init_rot_l1_weight.sum() + 1e-6)
            ) * self.obj_init_rot6d_weight * self.stage1_obj_init_rot6d_l1_scale
            loss_obj_init_rot = loss_obj_init_rot_geo + loss_obj_init_rot_l1

        loss_obj_view_rot = loss_obj_view_rot_geo + loss_obj_view_rot_l1 + loss_obj_view_trans + loss_obj_view_points

        if (
            stage_name == "stage2"
            and preds.get("obj_rot6d", None) is not None
            and preds.get("obj_trans", None) is not None
            and preds.get("obj_xyz_master", None) is not None
            and master_obj_rot6d_gt is not None
            and master_obj_trans_gt is not None
            and master_obj_sparse_gt is not None
        ):
            final_obj_rot6d = preds["obj_rot6d"][-1]
            final_obj_trans = preds["obj_trans"][-1]
            final_obj_points = preds["obj_xyz_master"][-1]
            loss_obj_rot_geo = torch.mean(self.rotation_geodesic(final_obj_rot6d, master_obj_rot6d_gt)) * self.obj_pose_rot_weight * obj_warmup
            loss_obj_rot_l1 = self.coord_loss(final_obj_rot6d, master_obj_rot6d_gt) * self.obj_pose_rot6d_weight * obj_warmup
            loss_obj_rot = loss_obj_rot_geo + loss_obj_rot_l1
            loss_obj_trans = self.coord_loss(final_obj_trans, master_obj_trans_gt) * self.obj_pose_trans_weight * obj_warmup
            final_cd = self._bidirectional_chamfer_distance(final_obj_points, master_obj_sparse_gt)
            loss_obj_points = final_cd.mean() * self.obj_pose_points_weight * obj_warmup
            loss_obj_pose = loss_obj_rot + loss_obj_trans + loss_obj_points

        total_loss = (
            loss_hand_2d_sv
            + loss_pose_reg_sv
            + loss_shape_reg_sv
            + loss_obj_2d
            + loss_obj_conf
            + loss_obj_tri_reproj
            + loss_obj_pose_gate
            + loss_triang_hand
            + loss_3d_jts
            + loss_2d_proj
            + loss_mano_proj
            + loss_obj_init_rot
            + loss_obj_init_trans
            + loss_obj_view_rot
            + loss_obj_pose
        )
        loss_dict = {
            "loss_2d_sv": loss_hand_2d_sv,
            "loss_pose_reg_sv": loss_pose_reg_sv,
            "loss_shape_reg_sv": loss_shape_reg_sv,
            "loss_obj_2d": loss_obj_2d,
            "loss_obj_conf": loss_obj_conf,
            "loss_obj_tri_reproj": loss_obj_tri_reproj,
            "loss_obj_pose_gate": loss_obj_pose_gate,
            "loss_triang_hand": loss_triang_hand,
            "loss_3d_jts": loss_3d_jts,
            "loss_2d_proj": loss_2d_proj,
            "loss_mano_proj": loss_mano_proj,
            "loss_obj_init_rot": loss_obj_init_rot,
            "loss_obj_init_rot_geo": loss_obj_init_rot_geo,
            "loss_obj_init_rot_l1": loss_obj_init_rot_l1,
            "loss_obj_init_trans": loss_obj_init_trans,
            "loss_obj_view_rot": loss_obj_view_rot,
            "loss_obj_view_rot_geo": loss_obj_view_rot_geo,
            "loss_obj_view_rot_l1": loss_obj_view_rot_l1,
            "loss_obj_view_trans": loss_obj_view_trans,
            "loss_obj_view_points": loss_obj_view_points,
            "loss_obj_rot": loss_obj_rot,
            "loss_obj_rot_geo": loss_obj_rot_geo,
            "loss_obj_rot_l1_aux": loss_obj_rot_l1,
            "loss_obj_trans": loss_obj_trans,
            "loss_obj_points": loss_obj_points,
            "loss_obj_pose": loss_obj_pose,
            "loss_obj_warmup": pred_mano_3d_mesh_sv.new_tensor(float(obj_warmup)),
            "loss": total_loss,
        }
        return total_loss, loss_dict

    def compute_sv_object_metrics(self, preds, gt):
        pred_obj_rot6d_sv = preds["obj_pose_rot6d_cam"]
        pred_obj_trans_sv = preds["obj_pose_trans_cam"]
        zero = pred_obj_rot6d_sv.new_tensor(0.0)
        target_obj_rot6d_gt, target_obj_trans_gt = self._get_gt_object_pose_abs(
            gt,
            dtype=pred_obj_rot6d_sv.dtype,
            device=pred_obj_rot6d_sv.device,
        )
        if target_obj_rot6d_gt is None or target_obj_trans_gt is None:
            return {
                "metric_sv_obj_rot_l1": zero,
                "metric_sv_obj_rot_deg": zero,
                "metric_sv_obj_trans_l1": zero,
                "metric_sv_obj_trans_epe": zero,
                "metric_sv_obj_add": zero,
                "metric_sv_obj_adds": zero,
                "metric_obj_pnp_valid": zero,
            }

        rot_deg = torch.rad2deg(self.rotation_geodesic(pred_obj_rot6d_sv.reshape(-1, 6), target_obj_rot6d_gt.reshape(-1, 6)))
        rot_l1 = torch.abs(pred_obj_rot6d_sv - target_obj_rot6d_gt).mean(dim=-1).reshape(-1)
        trans_l1 = torch.abs(pred_obj_trans_sv - target_obj_trans_gt).mean(dim=-1).reshape(-1)
        trans_epe = torch.linalg.norm(pred_obj_trans_sv - target_obj_trans_gt, dim=-1).reshape(-1)

        gt_obj_points = self._build_gt_object_points(gt, dtype=pred_obj_trans_sv.dtype, device=pred_obj_trans_sv.device)
        if gt_obj_points is not None and gt.get("target_joints_3d", None) is not None:
            pred_obj_points = self._build_pred_object_points_for_eval(preds, gt)
            add = torch.linalg.norm(pred_obj_points - gt_obj_points, dim=-1).mean(dim=-1).reshape(-1)
            pairwise = torch.cdist(pred_obj_points.float().flatten(0, 1), gt_obj_points.float().flatten(0, 1))
            adds = pairwise.min(dim=-1)[0].mean(dim=-1)
        else:
            add = zero.view(1)
            adds = zero.view(1)
        return {
            "metric_sv_obj_rot_l1": rot_l1.mean(),
            "metric_sv_obj_rot_deg": rot_deg.mean(),
            "metric_sv_obj_trans_l1": trans_l1.mean(),
            "metric_sv_obj_trans_epe": trans_epe.mean(),
            "metric_sv_obj_add": add.mean(),
            "metric_sv_obj_adds": adds.mean(),
            "metric_obj_pnp_valid": preds["obj_pose_valid"].mean(),
        }

    def compute_triangulation_confidence_metrics(self, preds):
        zero = preds["mano_3d_mesh_sv"].new_tensor(0.0)
        hand_conf = preds.get("conf_hand_tri", None)
        hand_err = preds.get("tri_hand_reproj_error", None)
        return {
            "metric_tri_hand_conf": hand_conf.mean() if hand_conf is not None else zero,
            "metric_tri_hand_px": hand_err.mean() if hand_err is not None else zero,
        }

    def compute_object_gate_metrics(self, preds):
        zero = preds["mano_3d_mesh_sv"].new_tensor(0.0)
        obj_tri_err = preds.get("pred_obj_kp_reproj_error", None)
        return {
            "metric_obj_heat_conf": preds.get("pred_obj_kp_heat_conf", zero).mean() if torch.is_tensor(preds.get("pred_obj_kp_heat_conf", None)) else zero,
            "metric_obj_vis_conf": preds.get("pred_obj_kp_vis_conf", zero).mean() if torch.is_tensor(preds.get("pred_obj_kp_vis_conf", None)) else zero,
            "metric_obj_tri_conf": preds.get("pred_obj_kp_conf_tri", zero).mean() if torch.is_tensor(preds.get("pred_obj_kp_conf_tri", None)) else zero,
            "metric_obj_pose_gate": preds.get("obj_pose_gate", zero).mean() if torch.is_tensor(preds.get("obj_pose_gate", None)) else zero,
            "metric_obj_refine_gate": preds.get("pred_obj_kp_refine_gate", zero).mean() if torch.is_tensor(preds.get("pred_obj_kp_refine_gate", None)) else zero,
            "metric_obj_tri_px": obj_tri_err.mean() if obj_tri_err is not None else zero,
        }

    def compute_object_mv_fusion_metrics(self, preds):
        zero = preds["mano_3d_mesh_sv"].new_tensor(0.0)
        view_weights = preds.get("obj_mv_view_weights", None)
        view_gate = preds.get("obj_mv_view_gate", None)
        master_weight = preds.get("obj_mv_master_weight", None)
        return {
            "metric_obj_mv_w": view_weights.max(dim=1).values.mean() if torch.is_tensor(view_weights) else zero,
            "metric_obj_mv_gate": view_gate.mean() if torch.is_tensor(view_gate) else zero,
            "metric_obj_mv_master_w": master_weight.mean() if torch.is_tensor(master_weight) else zero,
        }

    def compute_object_pose_metrics(self, preds, gt):
        pred_obj_rot6d = preds.get("obj_rot6d", None)
        pred_obj_trans = preds.get("obj_trans", None)
        if pred_obj_rot6d is None or pred_obj_trans is None:
            return self._zero_metric_dict(preds["mano_3d_mesh_sv"].device)

        pred_obj_rot6d = pred_obj_rot6d[-1]
        pred_obj_trans = pred_obj_trans[-1]
        gt_obj_rot6d = gt["master_obj_rot6d_label"].to(device=pred_obj_rot6d.device, dtype=pred_obj_rot6d.dtype)
        gt_obj_trans = gt["master_obj_t_label_rel"].to(device=pred_obj_trans.device, dtype=pred_obj_trans.dtype)
        rot_deg = torch.rad2deg(self.rotation_geodesic(pred_obj_rot6d, gt_obj_rot6d))
        rot_l1 = torch.abs(pred_obj_rot6d - gt_obj_rot6d).mean(dim=-1)
        trans_l1 = torch.abs(pred_obj_trans - gt_obj_trans).mean(dim=-1)
        trans_epe = torch.linalg.norm(pred_obj_trans - gt_obj_trans, dim=-1)

        pred_obj_points = preds["obj_xyz_master"][-1]
        gt_obj_points = gt["master_obj_sparse"].to(device=pred_obj_rot6d.device, dtype=pred_obj_rot6d.dtype)
        add = torch.linalg.norm(pred_obj_points - gt_obj_points, dim=-1).mean(dim=-1)
        adds = torch.cdist(pred_obj_points.float(), gt_obj_points.float()).min(dim=-1)[0].mean(dim=-1)
        return {
            "metric_obj_rot_l1": rot_l1.mean(),
            "metric_obj_rot_deg": rot_deg.mean(),
            "metric_obj_trans_l1": trans_l1.mean(),
            "metric_obj_trans_epe": trans_epe.mean(),
            "metric_obj_add": add.mean(),
            "metric_obj_adds": adds.mean(),
        }

    def _update_stage2_pose_metric(self, preds, gt):
        pred_obj_rot6d = preds.get("obj_rot6d", None)
        pred_obj_trans = preds.get("obj_trans", None)
        if pred_obj_rot6d is None or pred_obj_trans is None:
            return
        self.OBJ_POSE_VAL.feed(
            pred_obj_rot6d[-1],
            pred_obj_trans[-1],
            gt["master_obj_rot6d_label"].to(device=pred_obj_rot6d.device, dtype=pred_obj_rot6d.dtype),
            gt["master_obj_t_label_rel"].to(device=pred_obj_trans.device, dtype=pred_obj_trans.dtype),
            gt["master_obj_sparse_rest"].to(device=pred_obj_rot6d.device, dtype=pred_obj_rot6d.dtype),
        )

    def _update_sv_metrics(self, preds, batch, use_pa=False):
        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]
        pred_mano_3d_joints_sv = pred_mano_3d_mesh_sv[:, self.num_hand_verts:]
        pred_mano_3d_verts_sv = pred_mano_3d_mesh_sv[:, :self.num_hand_verts]
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)
        self.MPJPE_SV_3D.feed(pred_mano_3d_joints_sv, gt_kp=joints_3d_rel_gt)
        self.MPVPE_SV_3D.feed(pred_mano_3d_verts_sv, gt_kp=verts_3d_rel_gt)
        if use_pa:
            self.PA_SV.feed(pred_mano_3d_joints_sv, joints_3d_rel_gt, pred_mano_3d_verts_sv, verts_3d_rel_gt)

        gt_obj_points = self._build_gt_object_points(batch, dtype=pred_mano_3d_mesh_sv.dtype, device=pred_mano_3d_mesh_sv.device)
        if gt_obj_points is not None and batch.get("target_joints_3d", None) is not None:
            pred_obj_points = self._build_pred_object_points_for_eval(preds, batch)
            self.OBJ_RECON_SV.feed(pred_obj_points.flatten(0, 1), gt_obj_points.flatten(0, 1))

    def _write_scalars(self, prefix, scalar_dict, step_idx):
        if self.summary is None:
            return
        for key, value in scalar_dict.items():
            if torch.is_tensor(value):
                self.summary.add_scalar(f"{prefix}{key}", value.item(), step_idx)

    def training_step(self, batch, step_idx, **kwargs):
        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = self._stage2_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        _, loss_dict = self.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        metric_dict = {
            **self.compute_sv_object_metrics(preds, batch),
            **self.compute_triangulation_confidence_metrics(preds),
            **self.compute_object_gate_metrics(preds),
            **self.compute_object_mv_fusion_metrics(preds),
            **(self.compute_object_pose_metrics(preds, batch) if stage_name == "stage2" else self._zero_metric_dict(batch["image"].device)),
        }
        self.loss_metric.feed({**loss_dict, **metric_dict}, batch_size=batch["image"].size(0))
        self._update_sv_metrics(preds, batch, use_pa=False)
        if stage_name == "stage2":
            self._update_stage2_pose_metric(preds, batch)
        if step_idx % self.train_log_interval == 0:
            self._write_scalars("", {**loss_dict, **metric_dict}, step_idx)
            if self.summary is not None:
                self.summary.add_scalar("MPJPE_SV_3D", self.MPJPE_SV_3D.get_result(), step_idx)
                self.summary.add_scalar("MPVPE_SV_3D", self.MPVPE_SV_3D.get_result(), step_idx)
                self.summary.add_scalar("OBJREC_SV_CD", self.OBJ_RECON_SV.cd.avg, step_idx)
                self.summary.add_scalar("OBJREC_SV_FS_5", self.OBJ_RECON_SV.fs_5.avg, step_idx)
                self.summary.add_scalar("OBJREC_SV_FS_10", self.OBJ_RECON_SV.fs_10.avg, step_idx)
            if step_idx % (self.train_log_interval * 10) == 0:
                self._log_visualizations("train", batch, preds, step_idx)
        return preds, loss_dict

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        if mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        if mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        if mode == "draw":
            return self.draw_step(inputs, step_idx, **kwargs)
        raise ValueError(f"Unknown mode {mode}")

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.OBJ_RECON_SV, self.OBJ_POSE_VAL], epoch_idx, comment=comment, summary=self.format_metric("train"))
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_POSE_VAL.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = self._stage2_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        _, loss_dict = self.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        metric_dict = {
            **self.compute_sv_object_metrics(preds, batch),
            **self.compute_triangulation_confidence_metrics(preds),
            **self.compute_object_gate_metrics(preds),
            **self.compute_object_mv_fusion_metrics(preds),
            **(self.compute_object_pose_metrics(preds, batch) if stage_name == "stage2" else self._zero_metric_dict(batch["image"].device)),
        }
        self.loss_metric.feed({**loss_dict, **metric_dict}, batch_size=batch["image"].size(0))
        self._update_sv_metrics(preds, batch, use_pa=True)
        if stage_name == "stage2":
            self._update_stage2_pose_metric(preds, batch)
        if self.summary is not None:
            self._write_scalars("val_", {**loss_dict, **metric_dict}, step_idx)
            self.summary.add_scalar("MPJPE_SV_3D_val", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D_val", self.MPVPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("PA_SV_val", self.PA_SV.get_result(), step_idx)
            self.summary.add_scalar("OBJREC_SV_CD_val", self.OBJ_RECON_SV.cd.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_5_val", self.OBJ_RECON_SV.fs_5.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_10_val", self.OBJ_RECON_SV.fs_10.avg, step_idx)
        if step_idx % (self.train_log_interval * 5) == 0:
            self._log_visualizations("val", batch, preds, step_idx)
        return None

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_metric([self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.PA_SV, self.OBJ_RECON_SV, self.OBJ_POSE_VAL], epoch_idx, comment=comment, summary=self.format_metric("val"))
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.PA_SV.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_POSE_VAL.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        return self.validation_step(batch, step_idx, **kwargs)

    def draw_step(self, batch, step_idx, **kwargs):
        interaction_mode = self._stage_to_interaction_mode(self.current_stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._log_visualizations("draw", batch, preds, step_idx)
        return preds

    def _export_visualization_png(self, image_hwc, tag, mode, step_idx):
        if self.summary is None or mode not in {"val", "draw"}:
            return
        log_dir = getattr(self.summary, "log_dir", None)
        if log_dir is None:
            return
        exp_dir = os.path.dirname(log_dir.rstrip(os.sep))
        out_dir = os.path.join(exp_dir, "draws", tag, mode)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"step_{int(step_idx):07d}.png")
        image_bgr = cv2.cvtColor(np.asarray(image_hwc), cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, image_bgr)

    def _log_visualizations(self, mode, batch, preds, step_idx):
        if self.summary is None:
            return
        img = batch["image"]
        batch_size, num_views = img.shape[:2]
        img_h, img_w = img.shape[-2:]
        batch_id = 0
        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"].view(batch_size, num_views, self.num_hand_verts + self.num_hand_joints, 2)
        pred_mano_2d_joints_sv = pred_mano_2d_mesh_sv[:, :, self.num_hand_verts:]
        pred_mano_2d_verts_sv = pred_mano_2d_mesh_sv[:, :, :self.num_hand_verts]
        pred_sv_joints_2d = self._normed_uv_to_pixel(pred_mano_2d_joints_sv, (img_h, img_w))
        pred_sv_verts_2d = self._normed_uv_to_pixel(pred_mano_2d_verts_sv, (img_h, img_w))
        gt_joints_2d = self._normed_uv_to_pixel(batch["target_joints_uvd"][..., :2], (img_h, img_w))
        gt_verts_2d = self._normed_uv_to_pixel(batch["target_verts_uvd"][..., :2], (img_h, img_w))
        tri_hand_conf = preds["conf_hand_tri"]
        tri_hand_reproj_2d = preds["tri_hand_reproj_pixel"]
        pred_obj_points = self._build_pred_object_points_for_eval(preds, batch)
        gt_obj_points = self._build_gt_object_points(batch, dtype=pred_obj_points.dtype, device=pred_obj_points.device)
        pred_obj_2d = batch_cam_intr_projection(batch["target_cam_intr"], pred_obj_points)
        gt_obj_2d = batch_cam_intr_projection(batch["target_cam_intr"], gt_obj_points) if gt_obj_points is not None else pred_obj_2d
        pred_obj_kp2d = preds["pred_obj_kp2d_pixel"]
        pred_obj_kp2d_raw = preds["pred_obj_kp2d_pixel_raw"]
        pred_obj_kp2d_reproj = preds["pred_obj_kp2d_reproj"]
        pred_obj_kp_conf = preds["pred_obj_kp_conf"].squeeze(-1)
        pred_obj_kp_vis_conf = preds["pred_obj_kp_vis_conf"].squeeze(-1)
        pred_obj_kp_tri_conf = preds["pred_obj_kp_conf_tri"].squeeze(-1)
        gt_obj_kp2d = batch_cam_intr_projection(batch["target_cam_intr"], batch["target_obj_kp21"])

        pred_obj_rot_deg = None
        pred_obj_trans_mm = None
        gt_obj_rot6d_abs, gt_obj_trans_abs = self._get_gt_object_pose_abs(
            batch,
            dtype=preds["obj_pose_rot6d_cam"].dtype,
            device=preds["obj_pose_rot6d_cam"].device,
        )
        if gt_obj_rot6d_abs is not None:
            pred_obj_rot_deg = torch.rad2deg(
                self.rotation_geodesic(
                    preds["obj_pose_rot6d_cam"][batch_id],
                    gt_obj_rot6d_abs[batch_id],
                )
            ).unsqueeze(-1)
        if gt_obj_trans_abs is not None:
            pred_obj_trans_mm = torch.linalg.norm(
                preds["obj_pose_trans_cam"][batch_id] - gt_obj_trans_abs[batch_id],
                dim=-1,
            ).unsqueeze(-1) * 1000.0

        img_views = img[batch_id]
        hand_joint_tile = tile_batch_images(draw_batch_joint_images(pred_sv_joints_2d[batch_id], gt_joints_2d[batch_id], img_views, step_idx, n_sample=num_views))
        self.summary.add_image(f"vis/{mode}/j2d", hand_joint_tile, step_idx, dataformats="HWC")
        tri_hand_reproj_tile = tile_batch_images(
            draw_batch_joint_images(
                tri_hand_reproj_2d[batch_id],
                pred_sv_joints_2d[batch_id],
                img_views,
                step_idx,
                n_sample=num_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/tri", tri_hand_reproj_tile, step_idx, dataformats="HWC")
        hand_conf_tile = tile_batch_images(
            draw_batch_joint_confidence_images(
                pred_sv_joints_2d[batch_id],
                tri_hand_conf[batch_id],
                img_views,
                n_sample=num_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/conf", hand_conf_tile, step_idx, dataformats="HWC")
        hand_mesh_tile = tile_batch_images(
            draw_batch_hand_mesh_images_2d(
                gt_verts2d=gt_verts_2d[batch_id],
                pred_verts2d=pred_sv_verts_2d[batch_id],
                face=self.face,
                tensor_image=img_views,
                n_sample=num_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/mesh", hand_mesh_tile, step_idx, dataformats="HWC")
        hand_obj_tile = tile_batch_images(
            draw_batch_mesh_images_pred(
                gt_verts2d=gt_verts_2d[batch_id],
                pred_verts2d=pred_sv_verts_2d[batch_id],
                face=self.face,
                gt_obj2d=gt_obj_2d[batch_id],
                pred_obj2d=pred_obj_2d[batch_id],
                gt_objc2d=batch_cam_intr_projection(batch["target_cam_intr"], batch["target_obj_kp21"])[batch_id, :, -1:],
                pred_objc2d=pred_obj_kp2d[batch_id, :, -1:],
                intr=batch["target_cam_intr"][batch_id],
                tensor_image=img_views,
                pred_obj_rot_error=pred_obj_rot_deg,
                pred_obj_trans_error=pred_obj_trans_mm,
                n_sample=num_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/ho", hand_obj_tile, step_idx, dataformats="HWC")
        self._export_visualization_png(hand_obj_tile, "ho", mode, step_idx)
        obj_kp_pred_tile = tile_batch_images(
            draw_batch_object_kp_images(pred_obj_kp2d_raw[batch_id], img_views, n_sample=num_views, title="Pred Object KP21 Raw")
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_pred", obj_kp_pred_tile, step_idx, dataformats="HWC")
        obj_kp_reproj_tile = tile_batch_images(
            draw_batch_object_kp_images(pred_obj_kp2d_reproj[batch_id], img_views, n_sample=num_views, title="Tri Reproj Object KP21")
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_reproj", obj_kp_reproj_tile, step_idx, dataformats="HWC")
        obj_kp_refined_tile = tile_batch_images(
            draw_batch_object_kp_images(pred_obj_kp2d[batch_id], img_views, n_sample=num_views, title="Refined Object KP21")
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_refined", obj_kp_refined_tile, step_idx, dataformats="HWC")
        obj_kp_gt_tile = tile_batch_images(
            draw_batch_object_kp_images(gt_obj_kp2d[batch_id], img_views, n_sample=num_views, title="GT Object KP21")
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_gt", obj_kp_gt_tile, step_idx, dataformats="HWC")
        obj_kp_vis_conf_tile = tile_batch_images(
            draw_batch_object_kp_confidence_images(
                pred_obj_kp2d_raw[batch_id],
                pred_obj_kp_vis_conf[batch_id],
                img_views,
                n_sample=num_views,
                title="Pred Object Visibility Conf",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_vis_conf", obj_kp_vis_conf_tile, step_idx, dataformats="HWC")
        obj_kp_tri_conf_tile = tile_batch_images(
            draw_batch_object_kp_confidence_images(
                pred_obj_kp2d_reproj[batch_id],
                pred_obj_kp_tri_conf[batch_id],
                img_views,
                n_sample=num_views,
                title="Pred Object Tri Conf",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_tri_conf", obj_kp_tri_conf_tile, step_idx, dataformats="HWC")
        obj_kp_conf_tile = tile_batch_images(
            draw_batch_object_kp_confidence_images(
                pred_obj_kp2d[batch_id],
                pred_obj_kp_conf[batch_id],
                img_views,
                n_sample=num_views,
                title="Pred Object KP21 Conf",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_conf", obj_kp_conf_tile, step_idx, dataformats="HWC")

    def get_metric(self, preds, batch, **kwargs):
        return {
            **self.compute_sv_object_metrics(preds, batch),
            **self.compute_triangulation_confidence_metrics(preds),
            **self.compute_object_gate_metrics(preds),
            **self.compute_object_pose_metrics(preds, batch),
        }

    def format_metric(self, mode="train"):
        def _fmt_mm_value(value):
            return f"{float(value) * 1000.0:.1f}"
        def _fmt_mm_pair(v1, v2):
            return f"{float(v1) * 1000.0:.1f}/{float(v2) * 1000.0:.1f}"
        def _fmt_obj_recon(metric):
            return f"{metric.fs_5.avg:.3f}/{metric.fs_10.avg:.3f}/{metric.cd.avg:.3f}"
        def _get_loss(name, default=0.0):
            meter = self.loss_metric._losses.get(name, None)
            return float(meter.avg) if meter is not None else float(default)
        stage_prefix = "S2" if self.current_stage_name == "stage2" else "S1"

        if mode == "train":
            return (
                f"{stage_prefix} | L {_get_loss('loss'):.3f} | "
                f"H {_get_loss('loss_2d_sv'):.3f}/{_get_loss('loss_pose_reg_sv'):.3f}/{_get_loss('loss_shape_reg_sv'):.3f} | "
                f"O {_get_loss('loss_obj_2d'):.3f}/{_get_loss('loss_obj_conf'):.3f}/{_get_loss('loss_obj_tri_reproj'):.3f}/{_get_loss('loss_obj_pose_gate'):.3f}/{_get_loss('metric_obj_pnp_valid'):.2f} | "
                f"G {_get_loss('metric_obj_vis_conf'):.3f}/{_get_loss('metric_obj_tri_conf'):.3f}/{_get_loss('metric_obj_pose_gate'):.3f} | "
                f"M {_get_loss('metric_obj_mv_w'):.3f}/{_get_loss('metric_obj_mv_gate'):.3f}/{_get_loss('metric_obj_mv_master_w'):.3f} | "
                f"Tri {_get_loss('metric_tri_hand_conf'):.3f}/{_get_loss('metric_tri_hand_px'):.1f}px | "
                f"KP {_fmt_mm_pair(self.MPJPE_SV_3D.get_result(), self.MPVPE_SV_3D.get_result())} | "
                f"ObjSV {_get_loss('metric_sv_obj_rot_deg'):.1f}/{_fmt_mm_value(_get_loss('metric_sv_obj_trans_epe'))} | "
                f"ObjM {_get_loss('metric_obj_rot_deg'):.1f}/{_fmt_mm_value(_get_loss('metric_obj_trans_epe'))} | "
                f"RS {_fmt_obj_recon(self.OBJ_RECON_SV)}"
            )

        pa_sv = self.PA_SV.get_measures()
        return (
            f"{stage_prefix} | PA {_fmt_mm_pair(pa_sv.get('pa_mpjpe', 0.0), pa_sv.get('pa_mpvpe', 0.0))} | "
            f"KP {_fmt_mm_pair(self.MPJPE_SV_3D.get_result(), self.MPVPE_SV_3D.get_result())} | "
            f"Tri {_get_loss('metric_tri_hand_conf'):.3f}/{_get_loss('metric_tri_hand_px'):.1f}px | "
            f"PnP {_get_loss('metric_obj_pnp_valid'):.2f} | "
            f"G {_get_loss('metric_obj_vis_conf'):.3f}/{_get_loss('metric_obj_tri_conf'):.3f}/{_get_loss('metric_obj_pose_gate'):.3f} | "
            f"M {_get_loss('metric_obj_mv_w'):.3f}/{_get_loss('metric_obj_mv_gate'):.3f}/{_get_loss('metric_obj_mv_master_w'):.3f} | "
            f"Obj A/S {_fmt_mm_pair(_get_loss('metric_sv_obj_add'), _get_loss('metric_sv_obj_adds'))} | "
            f"Rot/Tr SV {_get_loss('metric_sv_obj_rot_deg'):.1f}/{_fmt_mm_value(_get_loss('metric_sv_obj_trans_epe'))} | "
            f"Rot/Tr M {_get_loss('metric_obj_rot_deg'):.1f}/{_fmt_mm_value(_get_loss('metric_obj_trans_epe'))} | "
            f"Rec {_fmt_obj_recon(self.OBJ_RECON_SV)}"
        )
