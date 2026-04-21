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
from ..utils.object_pose_utils import batched_kabsch_align, pose_from_keypoints
from ..utils.recorder import Recorder
from ..utils.transform import (
    batch_cam_extr_transf,
    batch_cam_intr_projection,
    batch_persp_project,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
)
from ..utils.triangulation import batch_triangulate_dlt_torch, batch_triangulate_dlt_torch_confidence
from ..viztools.draw import (
    draw_batch_hand_mesh_images_2d,
    draw_batch_joint_confidence_images,
    draw_batch_joint_images,
    draw_batch_mesh_images_pred,
    draw_batch_object_kp_confidence_images,
    draw_batch_object_kp_images,
    tile_batch_images,
)
from .HOR_sv_tri import AttentionPool2d, ResidualAttentionBlock, SharedHOStem
from .backbones import build_backbone
from .bricks.conv import ConvBlock
from .bricks.utils import ManoDecoder
from .model_abstraction import ModuleAbstract


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


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

        obj_tokens = self.obj_norm(obj_low.flatten(2).transpose(1, 2))
        hand_tokens = self.hand_norm(hand_low.flatten(2).transpose(1, 2))

        ref_tokens, _ = self.cross_attn(query=obj_tokens, key=hand_tokens, value=hand_tokens)
        ref_feat = ref_tokens.transpose(1, 2).reshape(batch_size, channels, self.pooled_hw, self.pooled_hw)
        ref_feat = F.interpolate(ref_feat, size=(height, width), mode="bilinear", align_corners=False)
        ref_feat = self.out_proj(ref_feat)
        ref_gate = self.gate(torch.cat((obj_feat, ref_feat), dim=1))
        fused_obj = obj_feat + ref_gate * ref_feat
        fused_obj = torch.nan_to_num(fused_obj, nan=0.0, posinf=1.0, neginf=-1.0)
        return fused_obj, ref_gate


@MODEL.register_module()
class POEM_SV_InitCPF(nn.Module, ModuleAbstract):
    """
    Stage1-only CPF-style initializer.

    Design goals:
    - Keep the current spatial backbone / feature pyramid.
    - Replace the old stage1 hand/object estimation path with two explicit
      single-view initialization branches:
        1) hand MANO init branch
        2) object canonical-pose init branch
    - Produce stable per-view 2D/3D predictions before DLT triangulation.
    - Leave clear teacher/self-distillation hooks for future pseudo-label use.

    Canonical teacher batch keys for future stage2/master distillation:
    - sv_teacher_hand_joints_2d
    - sv_teacher_obj_kp21_2d
    - sv_teacher_obj_rot6d_cam
    - sv_teacher_obj_trans_cam
    - sv_teacher_master_hand_joints_3d
    - sv_teacher_master_obj_kp21
    - sv_teacher_master_obj_rot6d
    - sv_teacher_master_obj_trans_rel

    Legacy teacher_* keys are still accepted for backward compatibility.
    """

    TEACHER_BATCH_KEY_ALIASES = {
        "sv_teacher_hand_joints_2d": ("sv_teacher_hand_joints_2d", "teacher_hand_joints_2d"),
        "sv_teacher_obj_kp21_2d": ("sv_teacher_obj_kp21_2d", "teacher_obj_kp2d"),
        "sv_teacher_obj_rot6d_cam": ("sv_teacher_obj_rot6d_cam", "teacher_obj_rot6d"),
        "sv_teacher_obj_trans_cam": ("sv_teacher_obj_trans_cam", "teacher_obj_trans"),
        "sv_teacher_master_hand_joints_3d": ("sv_teacher_master_hand_joints_3d", "teacher_master_joints_3d"),
        "sv_teacher_master_obj_kp21": ("sv_teacher_master_obj_kp21", "teacher_master_obj_kp21"),
        "sv_teacher_master_obj_rot6d": ("sv_teacher_master_obj_rot6d", "teacher_master_obj_rot6d"),
        "sv_teacher_master_obj_trans_rel": ("sv_teacher_master_obj_trans_rel", "teacher_master_obj_trans_rel"),
    }

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET
        self.summary = None

        self.center_idx = int(cfg.DATA_PRESET.CENTER_IDX)
        self.num_hand_joints = int(cfg.get("NUM_HAND_JOINTS", 21))
        self.num_hand_verts = int(cfg.get("NUM_HAND_VERTS", 778))
        self.num_obj_joints = int(cfg.DATA_PRESET.get("NUM_OBJ_JOINTS", cfg.get("NUM_OBJ_JOINTS", 21)))
        self.num_obj_classes = int(cfg.get("NUM_OBJ_CLASSES", 32))
        self.obj_id_embed_dim = int(cfg.get("OBJ_ID_EMBED_DIM", 32))
        self.shared_dim = int(cfg.get("SHARED_DIM", 128))
        self.branch_hidden_dim = int(cfg.get("INIT_BRANCH_HIDDEN_DIM", 512))
        self.obj_branch_style = str(cfg.get("OBJ_BRANCH_STYLE", "geom")).lower()

        loss_cfg = cfg.LOSS
        self.supervision_mode = str(loss_cfg.get("SUPERVISION_MODE", "full")).lower()
        self.hand_2d_weight = float(loss_cfg.get("HAND_2D_N", 1.0))
        self.hand_joints_3d_rel_weight = float(loss_cfg.get("HAND_JOINTS_3D_REL_N", 0.0))
        self.hand_verts_3d_rel_weight = float(loss_cfg.get("HAND_VERTS_3D_REL_N", 0.0))
        self.hand_joints_3d_cam_weight = float(loss_cfg.get("HAND_JOINTS_3D_CAM_N", 0.0))
        self.hand_verts_3d_cam_weight = float(loss_cfg.get("HAND_VERTS_3D_CAM_N", 0.0))
        self.hand_mano_pose_weight = float(loss_cfg.get("HAND_MANO_POSE_N", 0.0))
        self.hand_mano_shape_weight = float(loss_cfg.get("HAND_MANO_SHAPE_N", 0.0))
        self.pose_reg_weight = float(loss_cfg.get("POSE_N", 0.01))
        self.shape_reg_weight = float(loss_cfg.get("SHAPE_N", 1.0))
        self.obj_2d_weight = float(loss_cfg.get("OBJ_2D_N", loss_cfg.get("HFL_OBJ_REG_N", 1.0)))
        self.obj_rot_weight = float(loss_cfg.get("OBJ_POSE_ROT_N", 5.0))
        self.obj_rot6d_weight = float(loss_cfg.get("OBJ_POSE_ROT6D_N", 0.1))
        self.obj_trans_weight = float(loss_cfg.get("OBJ_POSE_TRANS_N", 10.0))
        self.obj_points_3d_weight = float(loss_cfg.get("OBJ_POINTS_3D_N", loss_cfg.get("OBJ_POSE_POINTS_N", 0.0)))
        self.obj_points_2d_weight = float(loss_cfg.get("OBJ_POINTS_2D_N", 0.0))
        self.obj_kp_rel_3d_weight = float(loss_cfg.get("OBJ_KP_REL_3D_N", 0.0))
        self.obj_points_rel_3d_weight = float(loss_cfg.get("OBJ_POINTS_REL_3D_N", 0.0))
        self.obj_points_use_kp21_fallback = bool(loss_cfg.get("OBJ_POINTS_USE_KP21_FALLBACK", True))
        self.obj_init_rot_weight = float(loss_cfg.get("OBJ_INIT_ROT_N", 2.0))
        self.obj_init_rot6d_weight = float(loss_cfg.get("OBJ_INIT_ROT6D_N", self.obj_rot6d_weight))
        self.obj_init_trans_weight = float(loss_cfg.get("OBJ_INIT_TRANS_N", self.obj_trans_weight))
        self.obj_view_rot_weight = float(loss_cfg.get("OBJ_VIEW_ROT_N", 1.0))
        self.obj_view_rot6d_weight = float(loss_cfg.get("OBJ_VIEW_ROT6D_N", 0.05))
        self.obj_view_trans_weight = float(loss_cfg.get("OBJ_VIEW_TRANS_N", self.obj_trans_weight))
        self.stage1_obj_rot6d_l1_scale = float(loss_cfg.get("STAGE1_OBJ_ROT6D_L1_SCALE", 0.2))
        self.stage1_obj_init_rot6d_l1_scale = float(
            loss_cfg.get("STAGE1_OBJ_INIT_ROT6D_L1_SCALE", self.stage1_obj_rot6d_l1_scale)
        )
        self.stage1_obj_view_rot6d_l1_scale = float(
            loss_cfg.get("STAGE1_OBJ_VIEW_ROT6D_L1_SCALE", self.stage1_obj_rot6d_l1_scale)
        )
        self.stage1_obj_rot6d_l1_cd_tau_m = float(loss_cfg.get("STAGE1_OBJ_ROT6D_L1_CD_TAU_M", 0.01))
        self.stage1_obj_rot6d_l1_mask_power = float(loss_cfg.get("STAGE1_OBJ_ROT6D_L1_MASK_POWER", 1.0))
        self.obj_direct_pose_supervision = bool(loss_cfg.get("OBJ_DIRECT_POSE_SUPERVISION", False))
        self.obj_master_supervision = bool(loss_cfg.get("OBJ_MASTER_SUPERVISION", False))
        self.tri_hand_weight = float(loss_cfg.get("TRIANGULATION_HAND_N", loss_cfg.get("TRIANGULATION_N", 10.0)))
        self.tri_obj_weight = float(loss_cfg.get("TRIANGULATION_OBJ_N", 1.0))

        self.teacher_hand_2d_weight = float(loss_cfg.get("DISTILL_HAND_2D_N", 0.0))
        self.teacher_obj_2d_weight = float(loss_cfg.get("DISTILL_OBJ_2D_N", 0.0))
        self.teacher_obj_rot_weight = float(loss_cfg.get("DISTILL_OBJ_ROT_N", 0.0))
        self.teacher_obj_trans_weight = float(loss_cfg.get("DISTILL_OBJ_TRANS_N", 0.0))
        self.teacher_tri_hand_weight = float(loss_cfg.get("DISTILL_TRI_HAND_N", 0.0))
        self.teacher_tri_obj_weight = float(loss_cfg.get("DISTILL_TRI_OBJ_N", 0.0))
        self._apply_supervision_mode()

        self.conf_tri_hand_tau_px = float(cfg.get("CONF_TRI_HAND_TAU_PX", 24.0))
        self.conf_tri_obj_tau_px = float(cfg.get("CONF_OBJ_TRI_TAU_PX", 24.0))
        self.conf_tri_refine_iters = int(cfg.get("CONF_TRI_REFINE_ITERS", 2))

        self.hand_trans_factor = float(cfg.get("HAND_TRANS_FACTOR", cfg.get("OBJ_TRANS_FACTOR", 100.0)))
        self.hand_scale_factor = float(cfg.get("HAND_SCALE_FACTOR", cfg.get("OBJ_SCALE_FACTOR", 0.0001)))
        self.obj_trans_factor = float(cfg.get("OBJ_TRANS_FACTOR", 100.0))
        self.obj_scale_factor = float(cfg.get("OBJ_SCALE_FACTOR", 0.0001))
        self.pose_recover_off_z = float(cfg.get("POSE_RECOVER_OFF_Z", 0.4))
        self.stage1_obj_hand_ref_detach = bool(cfg.get("STAGE1_OBJ_HAND_REF_DETACH", True))
        self.stage1_obj_hand_ref_heads = int(cfg.get("STAGE1_OBJ_HAND_REF_HEADS", 4))
        self.stage1_obj_hand_ref_pooled_hw = int(cfg.get("STAGE1_OBJ_HAND_REF_POOLED_HW", 8))

        self.coord_loss = nn.L1Loss()
        self.train_log_interval = int(cfg.TRAIN.LOG_INTERVAL)
        self.vis_log_interval = int(cfg.TRAIN.get("VIS_LOG_INTERVAL", max(self.train_log_interval * 20, 100)))
        self.val_vis_log_interval = int(cfg.TRAIN.get("VAL_VIS_LOG_INTERVAL", max(self.train_log_interval * 5, 25)))
        self.vis_num_views = int(cfg.TRAIN.get("VIS_NUM_VIEWS", 8))
        self.val_vis_num_views = int(cfg.TRAIN.get("VAL_VIS_NUM_VIEWS", min(4, self.vis_num_views)))
        self.vis_selection_log = bool(cfg.TRAIN.get("VIS_SELECTION_LOG", False))

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
        self.hand_pool = AttentionPool2d(self.shared_dim, out_dim=self.shared_dim)
        self.obj_pool = AttentionPool2d(self.shared_dim, out_dim=self.shared_dim)
        self.hand_avg = nn.AdaptiveAvgPool2d(1)
        self.obj_avg = nn.AdaptiveAvgPool2d(1)
        self.obj_query_hand_ref = ObjectQueryHandReference(
            channels=self.shared_dim,
            num_heads=self.stage1_obj_hand_ref_heads,
            pooled_hw=self.stage1_obj_hand_ref_pooled_hw,
        )
        self.global_proj = nn.Sequential(
            nn.Linear(self.feat_size[0], self.shared_dim),
            nn.GELU(),
            nn.Linear(self.shared_dim, self.shared_dim),
        )

        self.obj_id_embed = nn.Embedding(self.num_obj_classes, self.obj_id_embed_dim)
        self.obj_id_film = nn.Sequential(
            nn.Linear(self.obj_id_embed_dim, self.shared_dim * 2),
            nn.GELU(),
            nn.Linear(self.shared_dim * 2, self.shared_dim * 2),
        )

        hand_feat_dim = self.shared_dim * 3
        obj_feat_dim = self.shared_dim * 3 + self.obj_id_embed_dim

        self.hand_param_head = MLPHead(hand_feat_dim, self.branch_hidden_dim, 16 * 6 + 10)
        self.hand_camera_head = MLPHead(hand_feat_dim, max(self.branch_hidden_dim // 2, 128), 3)
        self.obj_rot_head = MLPHead(obj_feat_dim, self.branch_hidden_dim, 6)
        self.obj_camera_head = MLPHead(obj_feat_dim, max(self.branch_hidden_dim // 2, 128), 3)

        self.mano_decoder = ManoDecoder(
            self.data_preset_cfg.CENTER_IDX,
            self.data_preset_cfg.BBOX_3D_SIZE,
            self.data_preset_cfg.IMAGE_SIZE,
        )
        self.face = self.mano_decoder.face
        self.register_buffer("dummy_mano_cam", torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))
        self.register_buffer("identity_rot6d", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32))

        self.fc_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        self.loss_metric = LossMetric(cfg)
        self.MPJPE_SV_3D = MeanEPE(cfg, "SV_J")
        self.MPVPE_SV_3D = MeanEPE(cfg, "SV_V")
        self.PA_SV = PAEval(cfg, mesh_score=True)
        self.OBJ_RECON_SV = ObjectReconMetric(cfg, name="SVObjRec")
        self.OBJ_POSE_VAL = ObjectPoseMetric(cfg, name="ObjSV")
        self.OBJ_POSE_INIT = ObjectPoseMetric(cfg, name="ObjInit")

        self.init_weights()
        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} supervision mode: {self.supervision_mode}")
        logger.info(
            f"{self.name} object branch: style={self.obj_branch_style} "
            f"direct_pose_sup={self.obj_direct_pose_supervision} master_sup={self.obj_master_supervision}"
        )

    def _apply_supervision_mode(self):
        if self.supervision_mode == "full":
            return
        if self.supervision_mode == "pseudo_pose":
            self.hand_joints_3d_rel_weight = 0.0
            self.hand_verts_3d_rel_weight = 0.0
            self.hand_joints_3d_cam_weight = 0.0
            self.hand_verts_3d_cam_weight = 0.0
            self.hand_mano_pose_weight = 0.0
            self.hand_mano_shape_weight = 0.0

            self.tri_hand_weight = 0.0
            self.tri_obj_weight = 0.0
            return
        if self.supervision_mode != "pseudo":
            raise ValueError(f"Unsupported supervision mode: {self.supervision_mode}")

        # Pseudo-compatible mode keeps supervision that can later be replaced by
        # per-view pseudo labels or by template + pseudo pose synthesis, and
        # disables losses that require GT hand 3D / MANO / master 3D targets.
        self.hand_joints_3d_rel_weight = 0.0
        self.hand_verts_3d_rel_weight = 0.0
        self.hand_joints_3d_cam_weight = 0.0
        self.hand_verts_3d_cam_weight = 0.0
        self.hand_mano_pose_weight = 0.0
        self.hand_mano_shape_weight = 0.0

        self.obj_init_rot_weight = 0.0
        self.obj_init_rot6d_weight = 0.0
        self.obj_init_trans_weight = 0.0
        self.obj_view_rot_weight = 0.0
        self.obj_view_rot6d_weight = 0.0
        self.obj_view_trans_weight = 0.0
        self.obj_master_supervision = False

        self.tri_hand_weight = 0.0
        self.tri_obj_weight = 0.0

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
        for param in self.parameters():
            param.requires_grad = True
        if hasattr(self.img_backbone, "fc") and isinstance(self.img_backbone.fc, nn.Module):
            for param in self.img_backbone.fc.parameters():
                param.requires_grad = False

    def extract_img_feat(self, img):
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
    def _pixel_to_normed_uv(coords_uv, image_hw):
        img_h, img_w = image_hw
        scale = coords_uv.new_tensor([img_w, img_h]).view(1, 1, 2)
        return coords_uv / (scale / 2.0) - 1.0

    @staticmethod
    def _project_rotation_6d(rot6d):
        return rotmat_to_rot6d(rot6d_to_rotmat(rot6d))

    @staticmethod
    def rotation_geodesic(pred_rot6d, gt_rot6d):
        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return torch.acos(cos_theta)

    @staticmethod
    def recover_3d_proj(points3d, camintr, est_scale, est_trans, input_res, off_z=0.4):
        focal = camintr[:, :1, :1].reshape(-1, 1)
        est_scale = est_scale.reshape(-1, 1)
        est_trans = est_trans.reshape(-1, 2)
        est_z0 = focal * est_scale + off_z
        cam_centers = camintr[:, :2, 2]
        img_centers = (cam_centers.new_tensor(input_res) / 2.0).view(1, 2).repeat(points3d.shape[0], 1)
        est_xy0 = (est_trans + img_centers - cam_centers) * est_z0 / focal
        center3d = torch.cat([est_xy0, est_z0], dim=-1).unsqueeze(1)
        points3d_abs = center3d + points3d
        return points3d_abs, center3d

    @staticmethod
    def _build_object_points_from_pose(obj_points_rest, rot6d, trans_abs):
        obj_points_rest = torch.nan_to_num(obj_points_rest, nan=0.0, posinf=10.0, neginf=-10.0).float()
        rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=10.0, neginf=-10.0).float()
        trans_abs = torch.nan_to_num(trans_abs, nan=0.0, posinf=10.0, neginf=-10.0).float()
        rotmat = rot6d_to_rotmat(rot6d.reshape(-1, 6))
        if obj_points_rest.dim() == 4:
            obj_points_rest = obj_points_rest.reshape(-1, obj_points_rest.shape[-2], obj_points_rest.shape[-1])
        posed = torch.matmul(obj_points_rest, rotmat.transpose(1, 2))
        return posed + trans_abs.unsqueeze(1)

    def _extract_object_id(self, batch, device):
        obj_id = batch.get("obj_id", None)
        if obj_id is None:
            return None
        if not torch.is_tensor(obj_id):
            obj_id = torch.as_tensor(obj_id, device=device)
        obj_id = obj_id.to(device=device, dtype=torch.long)
        while obj_id.dim() > 1:
            obj_id = obj_id[..., 0]
        return obj_id.clamp_(min=0, max=max(self.num_obj_classes - 1, 0))

    def _apply_object_identity_conditioning(self, obj_feat, batch, batch_size, num_views):
        obj_id = self._extract_object_id(batch, obj_feat.device)
        if obj_id is None:
            cond_embed = obj_feat.new_zeros(batch_size, self.obj_id_embed_dim)
        else:
            cond_embed = self.obj_id_embed(obj_id)

        gamma_beta = self.obj_id_film(cond_embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma).repeat_interleave(num_views, dim=0).view(-1, self.shared_dim, 1, 1)
        beta = 0.1 * torch.tanh(beta).repeat_interleave(num_views, dim=0).view(-1, self.shared_dim, 1, 1)
        obj_feat = obj_feat * (1.0 + gamma) + beta
        return obj_feat, cond_embed

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
        if init_conf is None:
            conf = valid_obs.to(dtype=kp2d_pixel.dtype)
        else:
            conf = torch.nan_to_num(init_conf, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=kp2d_pixel.dtype)
            conf = conf * valid_obs.to(dtype=kp2d_pixel.dtype)

        reproj_pixel = None
        reproj_error = None
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

    @staticmethod
    def _recover_rigid_object_rotation_from_keypoints(rest_points, pred_points, eps=1e-6):
        rotmat, _, fit_error = batched_kabsch_align(rest_points, pred_points, eps=eps)
        rot6d = rotmat_to_rot6d(rotmat.reshape(-1, 3, 3)).view(rest_points.shape[0], 6)
        rigid_points = torch.matmul(rest_points, rotmat.transpose(1, 2))
        return rotmat, rot6d, rigid_points, fit_error

    def _recover_master_object_pose_from_tri(self, obj_kp_rest, obj_kp_master, hand_joints_master):
        hand_root = hand_joints_master[:, self.center_idx, :]
        pose_dict = pose_from_keypoints(obj_kp_rest, obj_kp_master, hand_root=hand_root)
        return pose_dict["rot6d"], pose_dict["trans_rel"], pose_dict["trans_abs"]

    @staticmethod
    def _masked_l1(pred, target, mask=None):
        diff = torch.abs(pred - target)
        if mask is None:
            return diff.mean()
        while mask.dim() < diff.dim():
            mask = mask.unsqueeze(-1)
        diff = diff * mask.to(dtype=diff.dtype)
        denom = mask.expand_as(diff).sum().clamp_min(1e-9)
        return diff.sum() / denom

    @staticmethod
    def _masked_epe(pred, target, mask=None):
        dist = torch.linalg.norm(pred - target, dim=-1)
        if mask is None:
            return dist.mean()
        mask = mask.to(dtype=dist.dtype)
        while mask.dim() > dist.dim():
            mask = mask.squeeze(-1)
        dist = dist * mask
        return dist.sum() / mask.sum().clamp_min(1e-9)

    @staticmethod
    def _masked_bidirectional_chamfer_distance(pred_points, target_points, pred_mask=None, target_mask=None):
        pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=10.0, neginf=-10.0).float()
        target_points = torch.nan_to_num(target_points, nan=0.0, posinf=10.0, neginf=-10.0).float()
        if pred_mask is None:
            pred_mask = torch.ones(pred_points.shape[:2], device=pred_points.device, dtype=torch.bool)
        if target_mask is None:
            target_mask = torch.ones(target_points.shape[:2], device=target_points.device, dtype=torch.bool)
        pred_mask = pred_mask.to(device=pred_points.device, dtype=torch.bool)
        target_mask = target_mask.to(device=target_points.device, dtype=torch.bool)

        pairwise = torch.cdist(pred_points, target_points)
        invalid_pair = (~pred_mask).unsqueeze(-1) | (~target_mask).unsqueeze(-2)
        pairwise = pairwise.masked_fill(invalid_pair, float("inf"))

        pred_min = pairwise.min(dim=-1)[0]
        target_min = pairwise.min(dim=-2)[0]
        pred_has_target = target_mask.any(dim=-1, keepdim=True)
        target_has_pred = pred_mask.any(dim=-1, keepdim=True)
        pred_valid = pred_mask & pred_has_target
        target_valid = target_mask & target_has_pred

        pred_min = torch.where(torch.isfinite(pred_min), pred_min, torch.zeros_like(pred_min))
        target_min = torch.where(torch.isfinite(target_min), target_min, torch.zeros_like(target_min))
        pred_loss = (pred_min * pred_valid.to(dtype=pred_min.dtype)).sum(dim=-1) / pred_valid.sum(dim=-1).clamp_min(1)
        target_loss = (target_min * target_valid.to(dtype=target_min.dtype)).sum(dim=-1) / target_valid.sum(dim=-1).clamp_min(1)
        return 0.5 * (pred_loss + target_loss)

    def _stage1_rot6d_l1_weight(self, chamfer_m):
        tau = max(float(self.stage1_obj_rot6d_l1_cd_tau_m), 1e-6)
        weight = (chamfer_m / tau).clamp_(0.0, 1.0)
        power = max(float(self.stage1_obj_rot6d_l1_mask_power), 1e-6)
        if abs(power - 1.0) > 1e-6:
            weight = weight.pow(power)
        return weight

    def _build_gt_object_points(self, batch, dtype, device):
        gt_obj_points = batch.get("target_obj_pc_sparse", None)
        if gt_obj_points is not None:
            return gt_obj_points.to(device=device, dtype=dtype)
        gt_obj_transform = batch.get("target_obj_transform", None)
        gt_obj_rotmat = batch.get("target_R_label", None)
        if gt_obj_transform is None or gt_obj_rotmat is None:
            return None
        obj_rest = batch.get("master_obj_sparse_rest", None)
        if obj_rest is None:
            obj_rest = batch["master_obj_kp21_rest"]
        obj_rest = obj_rest.to(device=device, dtype=dtype)
        gt_rot6d = rotmat_to_rot6d(gt_obj_rotmat.reshape(-1, 3, 3)).view(gt_obj_rotmat.shape[0], gt_obj_rotmat.shape[1], 6)
        gt_trans = gt_obj_transform[..., :3, 3].to(device=device, dtype=dtype)
        rest = obj_rest.unsqueeze(1).expand(-1, gt_rot6d.shape[1], -1, -1).reshape(-1, obj_rest.shape[-2], 3)
        points = self._build_object_points_from_pose(rest, gt_rot6d.reshape(-1, 6).to(device=device, dtype=dtype), gt_trans.reshape(-1, 3))
        return points.view(gt_rot6d.shape[0], gt_rot6d.shape[1], -1, 3)

    def _build_gt_master_object_kp21(self, batch, dtype, device):
        master_rot6d = batch.get("master_obj_rot6d_label", None)
        master_trans_rel = batch.get("master_obj_t_label_rel", None)
        master_joints = batch.get("master_joints_3d", None)
        obj_rest = batch.get("master_obj_kp21_rest", None)
        if master_rot6d is None or master_trans_rel is None or master_joints is None or obj_rest is None:
            return None
        master_rot6d = master_rot6d.to(device=device, dtype=dtype)
        master_trans_rel = master_trans_rel.to(device=device, dtype=dtype)
        hand_root = master_joints.to(device=device, dtype=dtype)[:, self.center_idx, :]
        master_trans_abs = master_trans_rel + hand_root
        obj_rest = obj_rest.to(device=device, dtype=dtype)
        return self._build_object_points_from_pose(obj_rest, master_rot6d, master_trans_abs)

    def _resolve_gt_object_points_for_pred(self, preds, batch, dtype, device):
        pred_obj_points = preds["obj_points_cam"]
        gt_obj_points = self._build_gt_object_points(batch, dtype=dtype, device=device)
        if gt_obj_points is not None and gt_obj_points.shape == pred_obj_points.shape:
            return pred_obj_points, gt_obj_points

        if not self.obj_points_use_kp21_fallback:
            return None, None
        gt_obj_kp21 = batch.get("target_obj_kp21", None)
        if gt_obj_kp21 is not None:
            gt_obj_kp21 = gt_obj_kp21.to(device=device, dtype=dtype)
            if gt_obj_kp21.shape == preds["obj_kp21_cam"].shape:
                return preds["obj_kp21_cam"], gt_obj_kp21

        return None, None

    def _get_obj_rest_template(self, batch, dtype, device):
        obj_rest = batch.get("master_obj_sparse_rest", None)
        if obj_rest is None:
            obj_rest = batch.get("master_obj_kp21_rest", None)
        if obj_rest is None:
            return None
        return obj_rest.to(device=device, dtype=dtype)

    def _get_gt_object_pose_abs(self, batch, dtype, device):
        gt_rotmat = batch.get("target_R_label", None)
        gt_obj_transform = batch.get("target_obj_transform", None)
        if gt_rotmat is None or gt_obj_transform is None:
            return None, None
        gt_rotmat = gt_rotmat.to(device=device, dtype=dtype)
        gt_obj_transform = gt_obj_transform.to(device=device, dtype=dtype)
        gt_rot6d = rotmat_to_rot6d(gt_rotmat.reshape(-1, 3, 3)).view(gt_rotmat.shape[0], gt_rotmat.shape[1], 6)
        gt_trans = gt_obj_transform[..., :3, 3]
        return gt_rot6d, gt_trans

    def _get_gt_master_object_pose_abs(self, batch, dtype, device):
        master_rot6d = batch.get("master_obj_rot6d_label", None)
        master_trans_rel = batch.get("master_obj_t_label_rel", None)
        master_joints = batch.get("master_joints_3d", None)
        if master_rot6d is None or master_trans_rel is None or master_joints is None:
            return None, None
        master_rot6d = master_rot6d.to(device=device, dtype=dtype)
        master_trans_rel = master_trans_rel.to(device=device, dtype=dtype)
        master_joints = master_joints.to(device=device, dtype=dtype)
        master_trans_abs = master_trans_rel + master_joints[:, self.center_idx, :]
        return master_rot6d, master_trans_abs

    def _get_teacher_tensor(self, batch, canonical_key, dtype, device):
        for key in self.TEACHER_BATCH_KEY_ALIASES[canonical_key]:
            teacher = batch.get(key, None)
            if teacher is not None:
                return teacher.to(device=device, dtype=dtype)
        return None

    def _resolve_gt_mano_targets(self, batch, dtype, device):
        gt_pose = batch.get("target_mano_pose", None)
        gt_shape = batch.get("target_mano_shape", None)

        if gt_shape is None:
            gt_shape = batch.get("mano_shape", None)

        if gt_pose is not None:
            gt_pose = gt_pose.to(device=device, dtype=dtype)
            if gt_pose.dim() >= 2 and gt_pose.shape[-1] == 3:
                gt_pose = gt_pose.reshape(*gt_pose.shape[:-2], gt_pose.shape[-2] * gt_pose.shape[-1])
        if gt_shape is not None:
            gt_shape = gt_shape.to(device=device, dtype=dtype)
            if gt_shape.dim() > 2:
                gt_shape = gt_shape.reshape(*gt_shape.shape[:-1], gt_shape.shape[-1])
        return gt_pose, gt_shape

    def _resolve_teacher_hand_2d(self, batch, dtype, device):
        teacher = self._get_teacher_tensor(batch, "sv_teacher_hand_joints_2d", dtype=dtype, device=device)
        if teacher is not None:
            return teacher
        teacher_master = self._get_teacher_tensor(batch, "sv_teacher_master_hand_joints_3d", dtype=dtype, device=device)
        if teacher_master is None:
            return None
        master_to_cam = torch.linalg.inv(batch["target_cam_extr"].to(device=device, dtype=dtype))
        teacher_cam = batch_cam_extr_transf(master_to_cam, teacher_master.unsqueeze(1).expand(-1, master_to_cam.shape[1], -1, -1))
        return batch_cam_intr_projection(batch["target_cam_intr"].to(device=device, dtype=dtype), teacher_cam)

    def _resolve_teacher_obj_2d(self, batch, dtype, device):
        teacher = self._get_teacher_tensor(batch, "sv_teacher_obj_kp21_2d", dtype=dtype, device=device)
        if teacher is not None:
            return teacher
        teacher_master = self._get_teacher_tensor(batch, "sv_teacher_master_obj_kp21", dtype=dtype, device=device)
        if teacher_master is None:
            return None
        master_to_cam = torch.linalg.inv(batch["target_cam_extr"].to(device=device, dtype=dtype))
        teacher_cam = batch_cam_extr_transf(master_to_cam, teacher_master.unsqueeze(1).expand(-1, master_to_cam.shape[1], -1, -1))
        return batch_cam_intr_projection(batch["target_cam_intr"].to(device=device, dtype=dtype), teacher_cam)

    def _resolve_teacher_master_obj_kp21(self, batch, dtype, device):
        teacher_master = self._get_teacher_tensor(batch, "sv_teacher_master_obj_kp21", dtype=dtype, device=device)
        if teacher_master is not None:
            return teacher_master
        teacher_rot6d = self._get_teacher_tensor(batch, "sv_teacher_master_obj_rot6d", dtype=dtype, device=device)
        teacher_trans_rel = self._get_teacher_tensor(batch, "sv_teacher_master_obj_trans_rel", dtype=dtype, device=device)
        teacher_hand = self._get_teacher_tensor(batch, "sv_teacher_master_hand_joints_3d", dtype=dtype, device=device)
        obj_rest = batch.get("master_obj_kp21_rest", None)
        if teacher_rot6d is None or teacher_trans_rel is None or teacher_hand is None or obj_rest is None:
            return None
        obj_rest = obj_rest.to(device=device, dtype=dtype)
        trans_abs = teacher_trans_rel + teacher_hand[:, self.center_idx, :]
        return self._build_object_points_from_pose(obj_rest, teacher_rot6d, trans_abs)

    def _resolve_teacher_obj_pose_cam(self, batch, dtype, device):
        teacher_rot = self._get_teacher_tensor(batch, "sv_teacher_obj_rot6d_cam", dtype=dtype, device=device)
        teacher_trans = self._get_teacher_tensor(batch, "sv_teacher_obj_trans_cam", dtype=dtype, device=device)
        if teacher_rot is not None and teacher_trans is not None:
            return teacher_rot, teacher_trans

        teacher_master_rot = self._get_teacher_tensor(batch, "sv_teacher_master_obj_rot6d", dtype=dtype, device=device)
        teacher_master_trans_rel = self._get_teacher_tensor(batch, "sv_teacher_master_obj_trans_rel", dtype=dtype, device=device)
        teacher_master_hand = self._get_teacher_tensor(batch, "sv_teacher_master_hand_joints_3d", dtype=dtype, device=device)
        if teacher_master_rot is None or teacher_master_trans_rel is None or teacher_master_hand is None:
            return teacher_rot, teacher_trans

        teacher_master_trans_abs = teacher_master_trans_rel + teacher_master_hand[:, self.center_idx, :]
        master_to_cam = torch.linalg.inv(batch["target_cam_extr"].to(device=device, dtype=dtype))
        master_to_cam_rot = master_to_cam[..., :3, :3]
        master_to_cam_trans = master_to_cam[..., :3, 3]

        teacher_master_rotmat = rot6d_to_rotmat(teacher_master_rot).unsqueeze(1).expand(-1, master_to_cam.shape[1], -1, -1)
        teacher_cam_rotmat = torch.matmul(master_to_cam_rot, teacher_master_rotmat)
        teacher_cam_rot6d = rotmat_to_rot6d(teacher_cam_rotmat.reshape(-1, 3, 3)).view(master_to_cam.shape[0], master_to_cam.shape[1], 6)
        teacher_cam_trans = (
            torch.matmul(master_to_cam_rot, teacher_master_trans_abs.unsqueeze(1).unsqueeze(-1)).squeeze(-1) + master_to_cam_trans
        )

        if teacher_rot is None:
            teacher_rot = teacher_cam_rot6d
        if teacher_trans is None:
            teacher_trans = teacher_cam_trans
        return teacher_rot, teacher_trans

    def _resolve_teacher_object_points_for_pred(self, preds, batch, dtype, device):
        pred_obj_points = preds["obj_points_cam"]
        obj_rest = self._get_obj_rest_template(batch, dtype=dtype, device=device)
        teacher_rot, teacher_trans = self._resolve_teacher_obj_pose_cam(batch, dtype=dtype, device=device)
        if obj_rest is None or teacher_rot is None or teacher_trans is None:
            return None, None
        if teacher_rot.dim() != 3 or teacher_trans.dim() != 3:
            return None, None
        if pred_obj_points.shape[:3] != (teacher_rot.shape[0], teacher_rot.shape[1], obj_rest.shape[-2]):
            return None, None
        obj_rest_flat = obj_rest.unsqueeze(1).expand(-1, teacher_rot.shape[1], -1, -1).reshape(-1, obj_rest.shape[-2], 3)
        teacher_points = self._build_object_points_from_pose(
            obj_rest_flat,
            teacher_rot.reshape(-1, 6),
            teacher_trans.reshape(-1, 3),
        ).view(teacher_rot.shape[0], teacher_rot.shape[1], -1, 3)
        return pred_obj_points, teacher_points

    def _resolve_object_points_for_supervision(self, preds, batch, dtype, device):
        if self.supervision_mode == "pseudo_pose":
            pred_obj_points, teacher_obj_points = self._resolve_teacher_object_points_for_pred(
                preds, batch, dtype=dtype, device=device
            )
            if pred_obj_points is not None and teacher_obj_points is not None:
                return pred_obj_points, teacher_obj_points
        return self._resolve_gt_object_points_for_pred(preds, batch, dtype=dtype, device=device)

    def _resolve_object_pose_abs_for_supervision(self, batch, dtype, device):
        if self.supervision_mode == "pseudo_pose":
            teacher_rot, teacher_trans = self._resolve_teacher_obj_pose_cam(batch, dtype=dtype, device=device)
            if teacher_rot is not None and teacher_trans is not None:
                return teacher_rot, teacher_trans
        return self._get_gt_object_pose_abs(batch, dtype=dtype, device=device)

    def _resolve_master_object_pose_for_supervision(self, batch, dtype, device):
        if self.supervision_mode == "pseudo_pose":
            teacher_rot = self._get_teacher_tensor(batch, "sv_teacher_master_obj_rot6d", dtype=dtype, device=device)
            teacher_trans_rel = self._get_teacher_tensor(batch, "sv_teacher_master_obj_trans_rel", dtype=dtype, device=device)
            teacher_hand = self._get_teacher_tensor(batch, "sv_teacher_master_hand_joints_3d", dtype=dtype, device=device)
            if teacher_rot is not None and teacher_trans_rel is not None and teacher_hand is not None:
                return teacher_rot, teacher_trans_rel, teacher_hand

        master_rot6d = batch.get("master_obj_rot6d_label", None)
        master_trans_rel = batch.get("master_obj_t_label_rel", None)
        master_hand = batch.get("master_joints_3d", None)
        if master_rot6d is None or master_trans_rel is None or master_hand is None:
            return None, None, None
        return (
            master_rot6d.to(device=device, dtype=dtype),
            master_trans_rel.to(device=device, dtype=dtype),
            master_hand.to(device=device, dtype=dtype),
        )

    def _forward_impl(self, batch, **kwargs):
        img = batch["image"]
        batch_size, num_cams = img.shape[:2]
        img_h, img_w = img.shape[-2:]
        bn = batch_size * num_cams

        img_feats, global_feat = self.extract_img_feat(img)
        mlvl_feat = self.feat_decode(img_feats)

        shared_feat = self.shared_stem(mlvl_feat)
        hand_feat = self.hand_adapter(shared_feat)
        obj_feat = self.obj_adapter(shared_feat)
        global_token = self.global_proj(global_feat)

        hand_token, hand_attn_map = self.hand_pool(hand_feat)
        hand_avg_feat = self.hand_avg(hand_feat).flatten(1)
        hand_branch_feat = torch.cat((hand_token, hand_avg_feat, global_token), dim=-1)

        obj_feat, obj_id_embed = self._apply_object_identity_conditioning(obj_feat, batch, batch_size, num_cams)
        hand_ref_memory = hand_feat.detach() if self.stage1_obj_hand_ref_detach else hand_feat
        obj_feat, obj_hand_ref_gate = self.obj_query_hand_ref(obj_feat, hand_ref_memory)
        obj_token, obj_attn_map = self.obj_pool(obj_feat)
        obj_avg_feat = self.obj_avg(obj_feat).flatten(1)
        obj_id_feat = obj_id_embed.repeat_interleave(num_cams, dim=0)
        obj_branch_feat = torch.cat((obj_token, obj_avg_feat, global_token, obj_id_feat), dim=-1)

        pred_hand_params = self.hand_param_head(hand_branch_feat)
        pred_hand_pose6d = pred_hand_params[:, :16 * 6]
        pred_hand_shape = pred_hand_params[:, 16 * 6:]
        pred_hand_camera = self.hand_camera_head(hand_branch_feat)
        pred_hand_scale = F.softplus(pred_hand_camera[:, :1]) + 1e-4
        pred_hand_trans = pred_hand_camera[:, 1:]

        dummy_cam = self.dummy_mano_cam.expand(bn, -1)
        hand_xyz_rel_norm, _, hand_pose_euler, hand_shape_sanitized, _ = self.mano_decoder(
            pred_hand_pose6d,
            pred_hand_shape,
            dummy_cam,
        )
        hand_xyz_rel = hand_xyz_rel_norm * (self.data_preset_cfg.BBOX_3D_SIZE / 2.0)
        hand_verts_rel = hand_xyz_rel[:, :self.num_hand_verts]
        hand_joints_rel = hand_xyz_rel[:, self.num_hand_verts:]

        target_cam_intr = batch["target_cam_intr"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype).flatten(0, 1)
        hand_scale = pred_hand_scale.view(-1, 1, 1) * self.hand_scale_factor
        hand_trans = pred_hand_trans * self.hand_trans_factor
        hand_joints_cam, hand_center3d = self.recover_3d_proj(
            hand_joints_rel,
            target_cam_intr,
            hand_scale,
            hand_trans,
            input_res=(img_w, img_h),
            off_z=self.pose_recover_off_z,
        )
        hand_verts_cam = hand_verts_rel + hand_center3d
        hand_mesh_cam = torch.cat((hand_verts_cam, hand_joints_cam), dim=1)
        pred_hand_joints_pixel = batch_persp_project(hand_joints_cam, target_cam_intr)

        if self.obj_branch_style != "geom":
            raise ValueError(f"Unsupported obj branch style: {self.obj_branch_style}")

        obj_kp21_rest = batch["master_obj_kp21_rest"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype)
        obj_kp21_rest_flat = obj_kp21_rest.unsqueeze(1).expand(-1, num_cams, -1, -1).reshape(-1, self.num_obj_joints, 3)
        pred_obj_rot6d = self._project_rotation_6d(self.obj_rot_head(obj_branch_feat) + self.identity_rot6d.unsqueeze(0))
        pred_obj_rotmat = rot6d_to_rotmat(pred_obj_rot6d)
        obj_kp21_rel = torch.matmul(obj_kp21_rest_flat, pred_obj_rotmat.transpose(1, 2))

        obj_camera_raw = self.obj_camera_head(obj_branch_feat)
        pred_obj_scale = F.softplus(obj_camera_raw[:, :1]) + 1e-4
        pred_obj_trans = obj_camera_raw[:, 1:]
        obj_scale = pred_obj_scale.view(-1, 1, 1) * self.obj_scale_factor
        obj_trans = pred_obj_trans * self.obj_trans_factor
        obj_kp21_cam, obj_center3d = self.recover_3d_proj(
            obj_kp21_rel,
            target_cam_intr,
            obj_scale,
            obj_trans,
            input_res=(img_w, img_h),
            off_z=self.pose_recover_off_z,
        )
        pred_obj_kp2d_pixel = batch_persp_project(obj_kp21_cam, target_cam_intr)
        pred_obj_kp2d_norm = self._pixel_to_normed_uv(pred_obj_kp2d_pixel, (img_h, img_w))

        obj_sparse_rest = batch.get("master_obj_sparse_rest", None)
        if obj_sparse_rest is not None:
            obj_sparse_rest = obj_sparse_rest.to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype)
            obj_sparse_rest_flat = obj_sparse_rest.unsqueeze(1).expand(-1, num_cams, -1, -1).reshape(-1, obj_sparse_rest.shape[-2], 3)
            obj_sparse_rel = torch.matmul(obj_sparse_rest_flat, pred_obj_rotmat.transpose(1, 2))
            obj_sparse_cam = obj_sparse_rel + obj_center3d
        else:
            obj_sparse_rel = obj_kp21_rel
            obj_sparse_cam = obj_kp21_cam

        obj_geom_fit_error = obj_kp21_rel.new_zeros(obj_kp21_rel.shape[0])

        pred_hand_joints_pixel_views = pred_hand_joints_pixel.view(batch_size, num_cams, self.num_hand_joints, 2)
        pred_obj_kp2d_pixel_views = pred_obj_kp2d_pixel.view(batch_size, num_cams, self.num_obj_joints, 2)

        tri_hand_master, tri_hand_reproj_pixel, tri_hand_reproj_error, conf_hand_tri = self._triangulate_with_reprojection_confidence(
            pred_hand_joints_pixel_views,
            batch["target_cam_intr"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype),
            batch["target_cam_extr"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype),
            tau_px=self.conf_tri_hand_tau_px,
        )
        tri_obj_master, tri_obj_reproj_pixel, tri_obj_reproj_error, conf_obj_tri = self._triangulate_with_reprojection_confidence(
            pred_obj_kp2d_pixel_views,
            batch["target_cam_intr"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype),
            batch["target_cam_extr"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype),
            tau_px=self.conf_tri_obj_tau_px,
        )

        obj_pose_rot6d_cam = pred_obj_rot6d.view(batch_size, num_cams, 6)
        obj_pose_trans_cam = obj_center3d.view(batch_size, num_cams, 3)
        obj_view_rot6d_master, obj_view_trans_master_abs = self._camera_pose_to_master(
            obj_pose_rot6d_cam,
            obj_pose_trans_cam,
            batch["target_cam_extr"].to(device=hand_xyz_rel.device, dtype=hand_xyz_rel.dtype),
        )
        obj_init_rot6d, obj_init_trans_rel, obj_init_trans_abs = self._recover_master_object_pose_from_tri(
            obj_kp21_rest,
            tri_obj_master,
            tri_hand_master,
        )

        obj_view_trans_master_rel = obj_view_trans_master_abs - tri_hand_master[:, self.center_idx].unsqueeze(1)
        obj_pose_valid = (obj_kp21_cam[..., 2].mean(dim=-1) > 1e-6).to(dtype=hand_xyz_rel.dtype).view(batch_size, num_cams)

        preds = {
            "mano_3d_mesh_sv": hand_xyz_rel,
            "mano_3d_mesh_cam": hand_mesh_cam,
            "mano_pose_euler_sv": hand_pose_euler,
            "mano_shape_sv": hand_shape_sanitized,
            "hand_scale_sv": pred_hand_scale,
            "hand_trans_sv": pred_hand_trans,
            "hand_center3d": hand_center3d.squeeze(1),
            "pred_hand_pixel": pred_hand_joints_pixel,
            "pred_hand_pixel_views": pred_hand_joints_pixel_views,
            "pred_obj_kp2d_pixel": pred_obj_kp2d_pixel_views,
            "pred_obj_kp2d_norm": pred_obj_kp2d_norm.view(batch_size, num_cams, self.num_obj_joints, 2),
            "obj_kp21_rel_raw": obj_kp21_rel.view(batch_size, num_cams, self.num_obj_joints, 3),
            "obj_geom_fit_error": obj_geom_fit_error.view(batch_size, num_cams),
            "obj_pose_rot6d_cam": obj_pose_rot6d_cam,
            "obj_pose_trans_cam": obj_pose_trans_cam,
            "obj_pose_valid": obj_pose_valid,
            "obj_points_cam": obj_sparse_cam.view(batch_size, num_cams, -1, 3),
            "obj_points_rel": obj_sparse_rel.view(batch_size, num_cams, -1, 3),
            "obj_kp21_cam": obj_kp21_cam.view(batch_size, num_cams, self.num_obj_joints, 3),
            "obj_kp21_rel": obj_kp21_rel.view(batch_size, num_cams, self.num_obj_joints, 3),
            "obj_view_rot6d_master": obj_view_rot6d_master,
            "obj_view_trans_master": obj_view_trans_master_rel,
            "obj_init_rot6d": obj_init_rot6d,
            "obj_init_trans": obj_init_trans_rel,
            "obj_init_trans_abs": obj_init_trans_abs,
            "tri_hand_joints_master": tri_hand_master,
            "tri_hand_reproj_pixel": tri_hand_reproj_pixel,
            "tri_hand_reproj_error": tri_hand_reproj_error,
            "conf_hand_tri": conf_hand_tri,
            "tri_obj_kp_master": tri_obj_master,
            "tri_obj_reproj_pixel": tri_obj_reproj_pixel,
            "tri_obj_reproj_error": tri_obj_reproj_error,
            "conf_obj_tri": conf_obj_tri,
            "reference_hand": hand_mesh_cam.view(batch_size, num_cams, self.num_hand_verts + self.num_hand_joints, 3).detach(),
            "reference_obj": obj_sparse_cam.view(batch_size, num_cams, -1, 3).detach(),
            "hand_attn_map": hand_attn_map.view(batch_size, num_cams, 1, hand_attn_map.shape[-2], hand_attn_map.shape[-1]),
            "obj_attn_map": obj_attn_map.view(batch_size, num_cams, 1, obj_attn_map.shape[-2], obj_attn_map.shape[-1]),
            "obj_hand_ref_gate": obj_hand_ref_gate.view(
                batch_size,
                num_cams,
                self.shared_dim,
                obj_hand_ref_gate.shape[-2],
                obj_hand_ref_gate.shape[-1],
            ),
        }
        return preds

    def compute_loss(self, preds, gt, **kwargs):
        device = preds["mano_3d_mesh_sv"].device
        dtype = preds["mano_3d_mesh_sv"].dtype
        zero = preds["mano_3d_mesh_sv"].new_tensor(0.0)

        loss_hand_2d = zero
        loss_hand_joints_3d_rel = zero
        loss_hand_verts_3d_rel = zero
        loss_hand_joints_3d_cam = zero
        loss_hand_verts_3d_cam = zero
        loss_hand_mano_pose = zero
        loss_hand_mano_shape = zero
        gt_hand_2d = gt.get("target_joints_2d", None)
        gt_hand_vis = gt.get("target_joints_vis", None)
        if gt_hand_2d is not None:
            gt_hand_2d = gt_hand_2d.to(device=device, dtype=dtype)
            if gt_hand_vis is not None:
                gt_hand_vis = gt_hand_vis.to(device=device, dtype=dtype)
            loss_hand_2d = self._masked_l1(preds["pred_hand_pixel_views"], gt_hand_2d, gt_hand_vis) * self.hand_2d_weight

        gt_hand_joints_rel = gt.get("target_joints_3d_rel", None)
        gt_hand_verts_rel = gt.get("target_verts_3d_rel", None)
        gt_hand_joints_cam = gt.get("target_joints_3d", None)
        gt_hand_verts_cam = gt.get("target_verts_3d", None)
        hand_vis_flat = None
        if gt_hand_vis is not None:
            hand_vis_flat = gt_hand_vis.flatten(0, 1)
        if gt_hand_joints_rel is not None and self.hand_joints_3d_rel_weight > 0.0:
            gt_hand_joints_rel = gt_hand_joints_rel.to(device=device, dtype=dtype).flatten(0, 1)
            pred_hand_joints_rel = preds["mano_3d_mesh_sv"][:, self.num_hand_verts:]
            loss_hand_joints_3d_rel = (
                self._masked_l1(pred_hand_joints_rel, gt_hand_joints_rel, hand_vis_flat) * self.hand_joints_3d_rel_weight
            )
        if gt_hand_verts_rel is not None and self.hand_verts_3d_rel_weight > 0.0:
            gt_hand_verts_rel = gt_hand_verts_rel.to(device=device, dtype=dtype).flatten(0, 1)
            pred_hand_verts_rel = preds["mano_3d_mesh_sv"][:, :self.num_hand_verts]
            loss_hand_verts_3d_rel = self.coord_loss(pred_hand_verts_rel, gt_hand_verts_rel) * self.hand_verts_3d_rel_weight
        pred_hand_mesh_cam = preds["mano_3d_mesh_cam"].view(
            gt["image"].shape[0],
            gt["image"].shape[1],
            self.num_hand_verts + self.num_hand_joints,
            3,
        )
        if gt_hand_joints_cam is not None and self.hand_joints_3d_cam_weight > 0.0:
            gt_hand_joints_cam = gt_hand_joints_cam.to(device=device, dtype=dtype)
            loss_hand_joints_3d_cam = (
                self._masked_l1(pred_hand_mesh_cam[..., self.num_hand_verts:, :], gt_hand_joints_cam, gt_hand_vis)
                * self.hand_joints_3d_cam_weight
            )
        if gt_hand_verts_cam is not None and self.hand_verts_3d_cam_weight > 0.0:
            gt_hand_verts_cam = gt_hand_verts_cam.to(device=device, dtype=dtype)
            loss_hand_verts_3d_cam = (
                self.coord_loss(pred_hand_mesh_cam[..., :self.num_hand_verts, :], gt_hand_verts_cam) * self.hand_verts_3d_cam_weight
            )

        gt_mano_pose, gt_mano_shape = self._resolve_gt_mano_targets(gt, dtype=dtype, device=device)
        if gt_mano_pose is not None and self.hand_mano_pose_weight > 0.0:
            gt_mano_pose = gt_mano_pose.flatten(0, 1)
            if gt_mano_pose.shape == preds["mano_pose_euler_sv"].shape:
                loss_hand_mano_pose = self.coord_loss(preds["mano_pose_euler_sv"], gt_mano_pose) * self.hand_mano_pose_weight
        if gt_mano_shape is not None and self.hand_mano_shape_weight > 0.0:
            gt_mano_shape = gt_mano_shape.flatten(0, 1)
            if gt_mano_shape.shape == preds["mano_shape_sv"].shape:
                loss_hand_mano_shape = self.coord_loss(preds["mano_shape_sv"], gt_mano_shape) * self.hand_mano_shape_weight

        loss_pose_reg = self.coord_loss(preds["mano_pose_euler_sv"][:, 3:], torch.zeros_like(preds["mano_pose_euler_sv"][:, 3:])) * self.pose_reg_weight
        loss_shape_reg = self.coord_loss(preds["mano_shape_sv"], torch.zeros_like(preds["mano_shape_sv"])) * self.shape_reg_weight

        loss_obj_2d = zero
        gt_obj_kp21 = gt.get("target_obj_kp21", None)
        if gt_obj_kp21 is not None:
            gt_obj_kp21 = gt_obj_kp21.to(device=device, dtype=dtype)
            gt_obj_kp2d = batch_cam_intr_projection(gt["target_cam_intr"].to(device=device, dtype=dtype), gt_obj_kp21)
            obj_valid = torch.isfinite(gt_obj_kp2d).all(dim=-1) & (gt_obj_kp21[..., 2] > 1e-6)
            loss_obj_2d = self._masked_l1(preds["pred_obj_kp2d_pixel"], gt_obj_kp2d, obj_valid) * self.obj_2d_weight

        loss_obj_rot_geo = zero
        loss_obj_rot_l1 = zero
        loss_obj_trans = zero
        loss_obj_init_rot_geo = zero
        loss_obj_init_rot_l1 = zero
        loss_obj_init_rot = zero
        loss_obj_init_trans = zero
        loss_obj_view_rot_geo = zero
        loss_obj_view_rot_l1 = zero
        loss_obj_view_rot = zero
        loss_obj_view_trans = zero
        view_cd = None
        gt_rot6d, gt_trans = self._resolve_object_pose_abs_for_supervision(gt, dtype=dtype, device=device)
        if self.obj_direct_pose_supervision and gt_rot6d is not None and gt_trans is not None:
            pred_rot = preds["obj_pose_rot6d_cam"]
            pred_trans = preds["obj_pose_trans_cam"]
            loss_obj_rot_geo = torch.mean(self.rotation_geodesic(pred_rot.reshape(-1, 6), gt_rot6d.reshape(-1, 6))) * self.obj_rot_weight
            loss_obj_trans = self.coord_loss(pred_trans, gt_trans) * self.obj_trans_weight

        gt_master_rot6d, gt_master_trans_rel, gt_master_hand = self._resolve_master_object_pose_for_supervision(
            gt, dtype=dtype, device=device
        )

        pred_init_rot = preds.get("obj_init_rot6d", None)
        pred_init_trans = preds.get("obj_init_trans", None)
        if self.obj_master_supervision and pred_init_rot is not None and gt_master_rot6d is not None:
            loss_obj_init_rot_geo = torch.mean(self.rotation_geodesic(pred_init_rot, gt_master_rot6d)) * self.obj_init_rot_weight
            loss_obj_init_rot = loss_obj_init_rot_geo + loss_obj_init_rot_l1
        if self.obj_master_supervision and pred_init_trans is not None and gt_master_trans_rel is not None:
            loss_obj_init_trans = self.coord_loss(pred_init_trans, gt_master_trans_rel) * self.obj_init_trans_weight

        pred_view_rot_master = preds.get("obj_view_rot6d_master", None)
        pred_view_trans_master = preds.get("obj_view_trans_master", None)
        if self.obj_master_supervision and pred_view_rot_master is not None and gt_master_rot6d is not None:
            target_view_rot = gt_master_rot6d.unsqueeze(1).expand_as(pred_view_rot_master)
            loss_obj_view_rot_geo = torch.mean(
                self.rotation_geodesic(pred_view_rot_master.reshape(-1, 6), target_view_rot.reshape(-1, 6))
            ) * self.obj_view_rot_weight
            loss_obj_view_rot = loss_obj_view_rot_geo + loss_obj_view_rot_l1
        if self.obj_master_supervision and pred_view_trans_master is not None and gt_master_trans_rel is not None:
            target_view_trans = gt_master_trans_rel.unsqueeze(1).expand_as(pred_view_trans_master)
            loss_obj_view_trans = self.coord_loss(pred_view_trans_master, target_view_trans) * self.obj_view_trans_weight

        loss_obj_points_3d = zero
        loss_obj_points_2d = zero
        loss_obj_kp_rel_3d = zero
        loss_obj_points_rel_3d = zero
        pred_obj_points_sup, gt_obj_points_sup = self._resolve_object_points_for_supervision(
            preds, gt, dtype=dtype, device=device
        )
        if pred_obj_points_sup is not None and gt_obj_points_sup is not None:
            pred_obj_points_valid = torch.isfinite(pred_obj_points_sup).all(dim=-1)
            obj_points_valid = torch.isfinite(gt_obj_points_sup).all(dim=-1) & (gt_obj_points_sup[..., 2] > 1e-6)
            view_cd = self._masked_bidirectional_chamfer_distance(
                pred_obj_points_sup.flatten(0, 1),
                gt_obj_points_sup.flatten(0, 1),
                pred_mask=pred_obj_points_valid.flatten(0, 1),
                target_mask=obj_points_valid.flatten(0, 1),
            ).view(pred_obj_points_sup.shape[0], pred_obj_points_sup.shape[1])
            if self.obj_direct_pose_supervision and gt_rot6d is not None and gt_trans is not None:
                rot_l1 = torch.abs(preds["obj_pose_rot6d_cam"] - gt_rot6d).mean(dim=-1)
                rot_l1_weight = self._stage1_rot6d_l1_weight(view_cd.detach())
                loss_obj_rot_l1 = (
                    (rot_l1 * rot_l1_weight).sum() / (rot_l1_weight.sum() + 1e-6)
                ) * self.obj_rot6d_weight * self.stage1_obj_rot6d_l1_scale
            if self.obj_points_3d_weight > 0.0:
                loss_obj_points_3d = view_cd.mean() * self.obj_points_3d_weight
            if self.obj_points_rel_3d_weight > 0.0 and gt_trans is not None:
                pred_obj_points_rel = pred_obj_points_sup - preds["obj_pose_trans_cam"].unsqueeze(2)
                gt_obj_points_rel = gt_obj_points_sup - gt_trans.unsqueeze(2)
                rel_cd = self._masked_bidirectional_chamfer_distance(
                    pred_obj_points_rel.flatten(0, 1),
                    gt_obj_points_rel.flatten(0, 1),
                    pred_mask=pred_obj_points_valid.flatten(0, 1),
                    target_mask=obj_points_valid.flatten(0, 1),
                ).view(pred_obj_points_rel.shape[0], pred_obj_points_rel.shape[1])
                loss_obj_points_rel_3d = rel_cd.mean() * self.obj_points_rel_3d_weight
            if self.obj_points_2d_weight > 0.0:
                gt_obj_points_2d = batch_cam_intr_projection(gt["target_cam_intr"].to(device=device, dtype=dtype), gt_obj_points_sup)
                pred_obj_points_2d = batch_cam_intr_projection(gt["target_cam_intr"].to(device=device, dtype=dtype), pred_obj_points_sup)
                obj_points_2d_valid = obj_points_valid & torch.isfinite(gt_obj_points_2d).all(dim=-1)
                pred_obj_points_2d_valid = torch.isfinite(pred_obj_points_2d).all(dim=-1) & pred_obj_points_valid
                points_2d_cd = self._masked_bidirectional_chamfer_distance(
                    pred_obj_points_2d.flatten(0, 1),
                    gt_obj_points_2d.flatten(0, 1),
                    pred_mask=pred_obj_points_2d_valid.flatten(0, 1),
                    target_mask=obj_points_2d_valid.flatten(0, 1),
                ).view(pred_obj_points_2d.shape[0], pred_obj_points_2d.shape[1])
                loss_obj_points_2d = points_2d_cd.mean() * self.obj_points_2d_weight
        elif self.obj_direct_pose_supervision and gt_rot6d is not None and gt_trans is not None:
            loss_obj_rot_l1 = self.coord_loss(preds["obj_pose_rot6d_cam"], gt_rot6d) * self.obj_rot6d_weight * self.stage1_obj_rot6d_l1_scale

        if self.obj_master_supervision and pred_view_rot_master is not None and gt_master_rot6d is not None:
            target_view_rot = gt_master_rot6d.unsqueeze(1).expand_as(pred_view_rot_master)
            if view_cd is not None:
                view_rot_l1 = torch.abs(pred_view_rot_master - target_view_rot).mean(dim=-1)
                view_rot_l1_weight = self._stage1_rot6d_l1_weight(view_cd.detach())
                loss_obj_view_rot_l1 = (
                    (view_rot_l1 * view_rot_l1_weight).sum() / (view_rot_l1_weight.sum() + 1e-6)
                ) * self.obj_view_rot6d_weight * self.stage1_obj_view_rot6d_l1_scale
            else:
                loss_obj_view_rot_l1 = (
                    self.coord_loss(pred_view_rot_master, target_view_rot)
                    * self.obj_view_rot6d_weight
                    * self.stage1_obj_view_rot6d_l1_scale
                )
            loss_obj_view_rot = loss_obj_view_rot_geo + loss_obj_view_rot_l1

        if (
            self.obj_master_supervision
            and pred_init_rot is not None
            and pred_init_trans is not None
            and gt_master_rot6d is not None
            and gt_master_trans_rel is not None
            and gt_master_hand is not None
        ):
            obj_rest_master = self._get_obj_rest_template(gt, dtype=dtype, device=device)
            if obj_rest_master is not None:
                hand_root_master = gt_master_hand[:, self.center_idx, :]
                pred_init_abs = hand_root_master + pred_init_trans
                gt_init_abs = hand_root_master + gt_master_trans_rel
                pred_init_points = self._build_object_points_from_pose(obj_rest_master, pred_init_rot, pred_init_abs)
                gt_init_points = self._build_object_points_from_pose(obj_rest_master, gt_master_rot6d, gt_init_abs)
                init_cd = self._masked_bidirectional_chamfer_distance(pred_init_points, gt_init_points)
                init_rot_l1 = torch.abs(pred_init_rot - gt_master_rot6d).mean(dim=-1)
                init_rot_l1_weight = self._stage1_rot6d_l1_weight(init_cd.detach())
                loss_obj_init_rot_l1 = (
                    (init_rot_l1 * init_rot_l1_weight).sum() / (init_rot_l1_weight.sum() + 1e-6)
                ) * self.obj_init_rot6d_weight * self.stage1_obj_init_rot6d_l1_scale
                loss_obj_init_rot = loss_obj_init_rot_geo + loss_obj_init_rot_l1
        elif self.obj_master_supervision and pred_init_rot is not None and gt_master_rot6d is not None:
            loss_obj_init_rot_l1 = (
                self.coord_loss(pred_init_rot, gt_master_rot6d)
                * self.obj_init_rot6d_weight
                * self.stage1_obj_init_rot6d_l1_scale
            )
            loss_obj_init_rot = loss_obj_init_rot_geo + loss_obj_init_rot_l1

        if self.obj_kp_rel_3d_weight > 0.0 and gt_obj_kp21 is not None and gt_trans is not None:
            obj_kp_valid = torch.isfinite(gt_obj_kp21).all(dim=-1) & (gt_obj_kp21[..., 2] > 1e-6)
            gt_obj_kp21_rel = gt_obj_kp21 - gt_trans.unsqueeze(2)
            loss_obj_kp_rel_3d = self._masked_l1(preds["obj_kp21_rel"], gt_obj_kp21_rel, obj_kp_valid) * self.obj_kp_rel_3d_weight

        loss_tri_hand = zero
        gt_master_hand = gt.get("master_joints_3d", None)
        if gt_master_hand is not None:
            gt_master_hand = gt_master_hand.to(device=device, dtype=dtype)
            loss_tri_hand = self.coord_loss(preds["tri_hand_joints_master"], gt_master_hand) * self.tri_hand_weight

        loss_tri_obj = zero
        gt_master_obj_kp21 = self._build_gt_master_object_kp21(gt, dtype=dtype, device=device)
        if self.obj_master_supervision and gt_master_obj_kp21 is not None:
            loss_tri_obj = self.coord_loss(preds["tri_obj_kp_master"], gt_master_obj_kp21) * self.tri_obj_weight

        loss_teacher_hand_2d = zero
        teacher_hand_2d = self._resolve_teacher_hand_2d(gt, dtype=dtype, device=device)
        if teacher_hand_2d is not None and self.teacher_hand_2d_weight > 0.0:
            loss_teacher_hand_2d = self.coord_loss(preds["pred_hand_pixel_views"], teacher_hand_2d) * self.teacher_hand_2d_weight

        loss_teacher_obj_2d = zero
        teacher_obj_2d = self._resolve_teacher_obj_2d(gt, dtype=dtype, device=device)
        if teacher_obj_2d is not None and self.teacher_obj_2d_weight > 0.0:
            loss_teacher_obj_2d = self.coord_loss(preds["pred_obj_kp2d_pixel"], teacher_obj_2d) * self.teacher_obj_2d_weight

        teacher_obj_rot, teacher_obj_trans = self._resolve_teacher_obj_pose_cam(gt, dtype=dtype, device=device)

        loss_teacher_obj_rot = zero
        if teacher_obj_rot is not None and self.teacher_obj_rot_weight > 0.0:
            loss_teacher_obj_rot = torch.mean(
                self.rotation_geodesic(preds["obj_pose_rot6d_cam"].reshape(-1, 6), teacher_obj_rot.reshape(-1, 6))
            ) * self.teacher_obj_rot_weight

        loss_teacher_obj_trans = zero
        if teacher_obj_trans is not None and self.teacher_obj_trans_weight > 0.0:
            loss_teacher_obj_trans = self.coord_loss(preds["obj_pose_trans_cam"], teacher_obj_trans) * self.teacher_obj_trans_weight

        loss_teacher_tri_hand = zero
        teacher_master_hand = self._get_teacher_tensor(gt, "sv_teacher_master_hand_joints_3d", dtype=dtype, device=device)
        if teacher_master_hand is not None and self.teacher_tri_hand_weight > 0.0:
            loss_teacher_tri_hand = self.coord_loss(preds["tri_hand_joints_master"], teacher_master_hand) * self.teacher_tri_hand_weight

        loss_teacher_tri_obj = zero
        teacher_master_obj = self._resolve_teacher_master_obj_kp21(gt, dtype=dtype, device=device)
        if teacher_master_obj is not None and self.teacher_tri_obj_weight > 0.0:
            loss_teacher_tri_obj = self.coord_loss(preds["tri_obj_kp_master"], teacher_master_obj) * self.teacher_tri_obj_weight

        total_loss = (
            loss_hand_2d
            + loss_hand_joints_3d_rel
            + loss_hand_verts_3d_rel
            + loss_hand_joints_3d_cam
            + loss_hand_verts_3d_cam
            + loss_hand_mano_pose
            + loss_hand_mano_shape
            + loss_pose_reg
            + loss_shape_reg
            + loss_obj_2d
            + loss_obj_rot_geo
            + loss_obj_rot_l1
            + loss_obj_trans
            + loss_obj_init_rot
            + loss_obj_init_trans
            + loss_obj_view_rot
            + loss_obj_view_trans
            + loss_obj_points_3d
            + loss_obj_kp_rel_3d
            + loss_obj_points_rel_3d
            + loss_obj_points_2d
            + loss_tri_hand
            + loss_tri_obj
            + loss_teacher_hand_2d
            + loss_teacher_obj_2d
            + loss_teacher_obj_rot
            + loss_teacher_obj_trans
            + loss_teacher_tri_hand
            + loss_teacher_tri_obj
        )
        loss_dict = {
            "loss_hand_2d": loss_hand_2d,
            "loss_hand_joints_3d_rel": loss_hand_joints_3d_rel,
            "loss_hand_verts_3d_rel": loss_hand_verts_3d_rel,
            "loss_hand_joints_3d_cam": loss_hand_joints_3d_cam,
            "loss_hand_verts_3d_cam": loss_hand_verts_3d_cam,
            "loss_hand_mano_pose": loss_hand_mano_pose,
            "loss_hand_mano_shape": loss_hand_mano_shape,
            "loss_pose_reg": loss_pose_reg,
            "loss_shape_reg": loss_shape_reg,
            "loss_obj_2d": loss_obj_2d,
            "loss_obj_rot_geo": loss_obj_rot_geo,
            "loss_obj_rot_l1": loss_obj_rot_l1,
            "loss_obj_trans": loss_obj_trans,
            "loss_obj_init_rot": loss_obj_init_rot,
            "loss_obj_init_rot_geo": loss_obj_init_rot_geo,
            "loss_obj_init_rot_l1": loss_obj_init_rot_l1,
            "loss_obj_init_trans": loss_obj_init_trans,
            "loss_obj_view_rot": loss_obj_view_rot,
            "loss_obj_view_rot_geo": loss_obj_view_rot_geo,
            "loss_obj_view_rot_l1": loss_obj_view_rot_l1,
            "loss_obj_view_trans": loss_obj_view_trans,
            "loss_obj_points_3d": loss_obj_points_3d,
            "loss_obj_kp_rel_3d": loss_obj_kp_rel_3d,
            "loss_obj_points_rel_3d": loss_obj_points_rel_3d,
            "loss_obj_points_2d": loss_obj_points_2d,
            "loss_tri_hand": loss_tri_hand,
            "loss_tri_obj": loss_tri_obj,
            "loss_teacher_hand_2d": loss_teacher_hand_2d,
            "loss_teacher_obj_2d": loss_teacher_obj_2d,
            "loss_teacher_obj_rot": loss_teacher_obj_rot,
            "loss_teacher_obj_trans": loss_teacher_obj_trans,
            "loss_teacher_tri_hand": loss_teacher_tri_hand,
            "loss_teacher_tri_obj": loss_teacher_tri_obj,
            "loss": total_loss,
        }
        return total_loss, loss_dict

    def compute_sv_object_metrics(self, preds, gt):
        zero = preds["mano_3d_mesh_sv"].new_tensor(0.0)
        dtype = preds["mano_3d_mesh_sv"].dtype
        device = preds["mano_3d_mesh_sv"].device
        metric_dict = {
            "metric_hand_2d_px": zero,
            "metric_hand_cam_j3d_epe": zero,
            "metric_hand_cam_v3d_epe": zero,
            "metric_obj_kp2d_px": zero,
            "metric_obj_kp_rel_3d_epe": zero,
            "metric_obj_points_3d_epe": zero,
            "metric_obj_points_rel_3d_epe": zero,
            "metric_obj_points_2d_px": zero,
            "metric_sv_obj_rot_deg": zero,
            "metric_sv_obj_trans_epe": zero,
            "metric_init_obj_rot_deg": zero,
            "metric_init_obj_trans_epe": zero,
            "metric_tri_hand_conf": preds["conf_hand_tri"].mean(),
            "metric_tri_hand_px": preds["tri_hand_reproj_error"].mean(),
            "metric_tri_obj_conf": preds["conf_obj_tri"].mean(),
            "metric_tri_obj_px": preds["tri_obj_reproj_error"].mean(),
        }

        gt_hand_2d = gt.get("target_joints_2d", None)
        gt_hand_vis = gt.get("target_joints_vis", None)
        if gt_hand_2d is not None:
            gt_hand_2d = gt_hand_2d.to(device=device, dtype=dtype)
            if gt_hand_vis is not None:
                gt_hand_vis = gt_hand_vis.to(device=device, dtype=dtype)
            metric_dict["metric_hand_2d_px"] = self._masked_epe(preds["pred_hand_pixel_views"], gt_hand_2d, gt_hand_vis)
        pred_hand_mesh_cam = preds["mano_3d_mesh_cam"].view(
            gt["image"].shape[0],
            gt["image"].shape[1],
            self.num_hand_verts + self.num_hand_joints,
            3,
        )
        gt_hand_joints_cam = gt.get("target_joints_3d", None)
        if gt_hand_joints_cam is not None:
            gt_hand_joints_cam = gt_hand_joints_cam.to(device=device, dtype=dtype)
            metric_dict["metric_hand_cam_j3d_epe"] = self._masked_epe(
                pred_hand_mesh_cam[..., self.num_hand_verts:, :],
                gt_hand_joints_cam,
                gt_hand_vis,
            )
        gt_hand_verts_cam = gt.get("target_verts_3d", None)
        if gt_hand_verts_cam is not None:
            gt_hand_verts_cam = gt_hand_verts_cam.to(device=device, dtype=dtype)
            metric_dict["metric_hand_cam_v3d_epe"] = self._masked_epe(
                pred_hand_mesh_cam[..., :self.num_hand_verts, :],
                gt_hand_verts_cam,
            )

        gt_obj_kp21 = gt.get("target_obj_kp21", None)
        if gt_obj_kp21 is not None:
            gt_obj_kp21 = gt_obj_kp21.to(device=device, dtype=dtype)
            gt_obj_kp2d = batch_cam_intr_projection(gt["target_cam_intr"].to(device=device, dtype=dtype), gt_obj_kp21)
            obj_valid = torch.isfinite(gt_obj_kp2d).all(dim=-1) & (gt_obj_kp21[..., 2] > 1e-6)
            metric_dict["metric_obj_kp2d_px"] = self._masked_epe(preds["pred_obj_kp2d_pixel"], gt_obj_kp2d, obj_valid)
            gt_obj_transform = gt.get("target_obj_transform", None)
            if gt_obj_transform is not None:
                gt_obj_trans = gt_obj_transform.to(device=device, dtype=dtype)[..., :3, 3]
                gt_obj_kp21_rel = gt_obj_kp21 - gt_obj_trans.unsqueeze(2)
                metric_dict["metric_obj_kp_rel_3d_epe"] = self._masked_epe(
                    preds["obj_kp21_rel"],
                    gt_obj_kp21_rel,
                    obj_valid,
                )

        pred_obj_points_sup, gt_obj_points_sup = self._resolve_gt_object_points_for_pred(preds, gt, dtype=dtype, device=device)
        if pred_obj_points_sup is not None and gt_obj_points_sup is not None:
            obj_points_valid = torch.isfinite(gt_obj_points_sup).all(dim=-1) & (gt_obj_points_sup[..., 2] > 1e-6)
            metric_dict["metric_obj_points_3d_epe"] = self._masked_epe(pred_obj_points_sup, gt_obj_points_sup, obj_points_valid)
            gt_obj_transform = gt.get("target_obj_transform", None)
            if gt_obj_transform is not None:
                gt_obj_trans = gt_obj_transform.to(device=device, dtype=dtype)[..., :3, 3]
                pred_obj_points_rel = pred_obj_points_sup - preds["obj_pose_trans_cam"].unsqueeze(2)
                gt_obj_points_rel = gt_obj_points_sup - gt_obj_trans.unsqueeze(2)
                metric_dict["metric_obj_points_rel_3d_epe"] = self._masked_epe(
                    pred_obj_points_rel,
                    gt_obj_points_rel,
                    obj_points_valid,
                )
            gt_obj_points_2d = batch_cam_intr_projection(gt["target_cam_intr"].to(device=device, dtype=dtype), gt_obj_points_sup)
            pred_obj_points_2d = batch_cam_intr_projection(gt["target_cam_intr"].to(device=device, dtype=dtype), pred_obj_points_sup)
            obj_points_2d_valid = obj_points_valid & torch.isfinite(gt_obj_points_2d).all(dim=-1)
            metric_dict["metric_obj_points_2d_px"] = self._masked_epe(pred_obj_points_2d, gt_obj_points_2d, obj_points_2d_valid)

        gt_rot6d, gt_trans = self._get_gt_object_pose_abs(gt, dtype=dtype, device=device)
        if gt_rot6d is not None and gt_trans is not None:
            rot_deg = torch.rad2deg(
                self.rotation_geodesic(preds["obj_pose_rot6d_cam"].reshape(-1, 6), gt_rot6d.reshape(-1, 6))
            )
            trans_epe = torch.linalg.norm(preds["obj_pose_trans_cam"] - gt_trans, dim=-1)
            metric_dict["metric_sv_obj_rot_deg"] = rot_deg.mean()
            metric_dict["metric_sv_obj_trans_epe"] = trans_epe.mean()

        gt_master_rot6d, gt_master_trans_abs = self._get_gt_master_object_pose_abs(gt, dtype=dtype, device=device)
        if gt_master_rot6d is not None and gt_master_trans_abs is not None:
            metric_dict["metric_init_obj_rot_deg"] = torch.rad2deg(
                self.rotation_geodesic(preds["obj_init_rot6d"], gt_master_rot6d)
            ).mean()
            metric_dict["metric_init_obj_trans_epe"] = torch.linalg.norm(
                preds["obj_init_trans_abs"] - gt_master_trans_abs,
                dim=-1,
            ).mean()

        return metric_dict

    def _update_sv_metrics(self, preds, batch, use_pa=False):
        gt_joints_rel = batch.get("target_joints_3d_rel", None)
        gt_verts_rel = batch.get("target_verts_3d_rel", None)
        if gt_joints_rel is not None and gt_verts_rel is not None:
            gt_joints_rel = gt_joints_rel.flatten(0, 1)
            gt_verts_rel = gt_verts_rel.flatten(0, 1)
            pred_joints_rel = preds["mano_3d_mesh_sv"][:, self.num_hand_verts:]
            pred_verts_rel = preds["mano_3d_mesh_sv"][:, :self.num_hand_verts]
            self.MPJPE_SV_3D.feed(pred_joints_rel, gt_joints_rel)
            self.MPVPE_SV_3D.feed(pred_verts_rel, gt_verts_rel)
            if use_pa:
                self.PA_SV.feed(pred_joints_rel, gt_joints_rel, pred_verts_rel, gt_verts_rel)

        gt_obj_points = self._build_gt_object_points(batch, dtype=preds["mano_3d_mesh_sv"].dtype, device=preds["mano_3d_mesh_sv"].device)
        if gt_obj_points is not None:
            self.OBJ_RECON_SV.feed(preds["obj_points_cam"].flatten(0, 1), gt_obj_points.flatten(0, 1))

        dtype = preds["mano_3d_mesh_sv"].dtype
        device = preds["mano_3d_mesh_sv"].device
        gt_rot6d_views, gt_trans_views = self._get_gt_object_pose_abs(batch, dtype=dtype, device=device)
        obj_rest = self._get_obj_rest_template(batch, dtype=dtype, device=device)
        if gt_rot6d_views is not None and gt_trans_views is not None and obj_rest is not None:
            obj_rest_views = obj_rest.unsqueeze(1).expand(-1, batch["image"].shape[1], -1, -1).reshape(-1, obj_rest.shape[-2], 3)
            self.OBJ_POSE_VAL.feed(
                preds["obj_pose_rot6d_cam"].reshape(-1, 6),
                preds["obj_pose_trans_cam"].reshape(-1, 3),
                gt_rot6d_views.reshape(-1, 6),
                gt_trans_views.reshape(-1, 3),
                obj_rest_views,
            )

        gt_master_rot6d, gt_master_trans_abs = self._get_gt_master_object_pose_abs(batch, dtype=dtype, device=device)
        if gt_master_rot6d is not None and gt_master_trans_abs is not None and obj_rest is not None:
            self.OBJ_POSE_INIT.feed(
                preds["obj_init_rot6d"],
                preds["obj_init_trans_abs"],
                gt_master_rot6d,
                gt_master_trans_abs,
                obj_rest,
            )

    def _write_scalars(self, prefix, scalar_dict, step_idx):
        if self.summary is None:
            return
        for key, value in scalar_dict.items():
            if torch.is_tensor(value):
                self.summary.add_scalar(f"{prefix}{key}", value.item(), step_idx)

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

    @staticmethod
    def _to_uint8_hwc(image_hwc):
        image_hwc = np.asarray(image_hwc)
        if image_hwc.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape={image_hwc.shape}")
        if image_hwc.shape[-1] == 1:
            image_hwc = np.repeat(image_hwc, 3, axis=-1)
        elif image_hwc.shape[-1] == 4:
            image_hwc = cv2.cvtColor(image_hwc, cv2.COLOR_RGBA2RGB)
        if image_hwc.dtype != np.uint8:
            image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)
        return image_hwc

    @classmethod
    def _caption_visualization_panel(cls, image_hwc, title):
        image_hwc = cls._to_uint8_hwc(image_hwc)
        title_h = 28
        panel = np.full((image_hwc.shape[0] + title_h, image_hwc.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            panel,
            title,
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (32, 32, 32),
            1,
            cv2.LINE_AA,
        )
        panel[title_h:] = image_hwc
        return panel

    @staticmethod
    def _pad_visualization_panel(image_hwc, target_h, target_w, pad_value=255):
        image_hwc = np.asarray(image_hwc)
        pad_h = max(target_h - image_hwc.shape[0], 0)
        pad_w = max(target_w - image_hwc.shape[1], 0)
        if pad_h == 0 and pad_w == 0:
            return image_hwc
        return cv2.copyMakeBorder(
            image_hwc,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )

    @classmethod
    def _build_visualization_overview(cls, named_panels, n_cols=2, max_long_side=2400):
        panels = [
            cls._caption_visualization_panel(image_hwc, title)
            for title, image_hwc in named_panels
            if image_hwc is not None
        ]
        if not panels:
            return None

        target_h = max(panel.shape[0] for panel in panels)
        target_w = max(panel.shape[1] for panel in panels)
        padded_panels = [cls._pad_visualization_panel(panel, target_h, target_w) for panel in panels]

        blank_panel = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        rows = []
        for start in range(0, len(padded_panels), n_cols):
            row_panels = padded_panels[start:start + n_cols]
            if len(row_panels) < n_cols:
                row_panels = row_panels + [blank_panel] * (n_cols - len(row_panels))
            rows.append(np.concatenate(row_panels, axis=1))

        overview = np.concatenate(rows, axis=0)
        long_side = max(overview.shape[0], overview.shape[1])
        if long_side > max_long_side:
            scale = float(max_long_side) / float(long_side)
            new_w = max(1, int(round(overview.shape[1] * scale)))
            new_h = max(1, int(round(overview.shape[0] * scale)))
            overview = cv2.resize(overview, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return overview

    @staticmethod
    def _mean_pixel_error(pred_points, gt_points):
        if pred_points is None or gt_points is None:
            return None
        return torch.linalg.norm(pred_points - gt_points, dim=-1).mean(dim=-1)

    @staticmethod
    def _normalize_metric_per_sample(metric_2d):
        finite_mask = torch.isfinite(metric_2d)
        safe_metric = torch.nan_to_num(metric_2d, nan=0.0, posinf=0.0, neginf=0.0)
        min_metric = safe_metric.masked_fill(~finite_mask, float("inf")).min(dim=1, keepdim=True).values
        max_metric = safe_metric.masked_fill(~finite_mask, float("-inf")).max(dim=1, keepdim=True).values
        has_finite = finite_mask.any(dim=1, keepdim=True)
        min_metric = torch.where(has_finite, min_metric, torch.zeros_like(min_metric))
        max_metric = torch.where(has_finite, max_metric, min_metric + 1.0)
        denom = torch.clamp(max_metric - min_metric, min=1e-6)
        norm_metric = (safe_metric - min_metric) / denom
        norm_metric = norm_metric * finite_mask.to(norm_metric.dtype)
        return norm_metric, finite_mask

    def _select_visualization_batch_views(
        self,
        mode,
        num_views,
        hand_err_px,
        hand_tri_err_px,
        obj_kp_err_px,
        obj_tri_err_px,
        obj_point_err_mm,
        view_rot_err_deg,
        view_trans_err_mm,
    ):
        max_views = min(num_views, self.val_vis_num_views if mode == "val" else self.vis_num_views)
        device = None
        for metric in (hand_err_px, hand_tri_err_px, obj_kp_err_px, obj_tri_err_px, obj_point_err_mm, view_rot_err_deg, view_trans_err_mm):
            if metric is not None:
                device = metric.device
                break
        if device is None:
            device = torch.device("cpu")
        default_view_ids = torch.arange(max_views, device=device, dtype=torch.long)
        if mode != "val" or max_views >= num_views:
            return 0, default_view_ids, None

        metric_list = [
            hand_err_px,
            hand_tri_err_px,
            obj_kp_err_px,
            obj_tri_err_px,
            obj_point_err_mm,
            view_rot_err_deg,
            view_trans_err_mm,
        ]
        metric_list = [metric for metric in metric_list if metric is not None]
        if not metric_list:
            return 0, default_view_ids, None

        combined_score = torch.zeros_like(metric_list[0], dtype=torch.float32)
        valid_count = torch.zeros_like(metric_list[0], dtype=torch.float32)
        for metric in metric_list:
            norm_metric, finite_mask = self._normalize_metric_per_sample(metric.detach().to(dtype=torch.float32))
            combined_score = combined_score + norm_metric
            valid_count = valid_count + finite_mask.to(dtype=torch.float32)

        has_metric = valid_count > 0
        combined_score = combined_score / valid_count.clamp(min=1.0)
        batch_valid = has_metric.any(dim=1)
        if not bool(batch_valid.any().item()):
            return 0, default_view_ids, None

        batch_score = combined_score.masked_fill(~has_metric, 0.0).sum(dim=1) / has_metric.to(dtype=torch.float32).sum(dim=1).clamp(min=1.0)
        batch_score = batch_score.masked_fill(~batch_valid, float("-inf"))
        batch_id = int(torch.argmax(batch_score).item())

        view_score = combined_score[batch_id]
        view_valid = has_metric[batch_id]
        if not bool(view_valid.any().item()):
            return batch_id, default_view_ids, f"b{batch_id} first views"

        k = min(max_views, int(view_valid.sum().item()))
        ranked_view_ids = torch.topk(view_score.masked_fill(~view_valid, float("-inf")), k=k, largest=True).indices
        if k < max_views:
            pad_candidates = default_view_ids[~torch.isin(default_view_ids, ranked_view_ids)]
            ranked_view_ids = torch.cat([ranked_view_ids, pad_candidates[:max_views - k]], dim=0)

        selected_view_ids = ranked_view_ids[:max_views]
        selected_view_ids_cpu = selected_view_ids.detach().cpu().tolist()
        selection_desc = f"b{batch_id} worst v{selected_view_ids_cpu}"
        return batch_id, selected_view_ids, selection_desc

    def _log_visualizations(self, mode, batch, preds, step_idx):
        if self.summary is None:
            return

        img = batch["image"]
        batch_size, num_views = img.shape[:2]
        if batch_size == 0 or num_views == 0:
            return

        img_h, img_w = img.shape[-2:]
        num_vis_views = min(num_views, self.val_vis_num_views if mode == "val" else self.vis_num_views)

        pred_hand_mesh_cam = preds["mano_3d_mesh_cam"].view(batch_size, num_views, self.num_hand_verts + self.num_hand_joints, 3)
        pred_hand_verts_2d = batch_cam_intr_projection(
            batch["target_cam_intr"],
            pred_hand_mesh_cam[..., :self.num_hand_verts, :],
        )
        pred_hand_joints_2d = preds["pred_hand_pixel_views"]

        gt_hand_joints_2d = batch.get("target_joints_2d", None)
        if gt_hand_joints_2d is None and batch.get("target_joints_uvd", None) is not None:
            gt_hand_joints_2d = self._normed_uv_to_pixel(batch["target_joints_uvd"][..., :2], (img_h, img_w))
        if gt_hand_joints_2d is None:
            gt_hand_joints_2d = pred_hand_joints_2d.detach()

        gt_hand_verts_2d = None
        if batch.get("target_verts_uvd", None) is not None:
            gt_hand_verts_2d = self._normed_uv_to_pixel(batch["target_verts_uvd"][..., :2], (img_h, img_w))
        elif batch.get("target_verts_3d", None) is not None:
            gt_hand_verts_2d = batch_cam_intr_projection(batch["target_cam_intr"], batch["target_verts_3d"])
        if gt_hand_verts_2d is None:
            gt_hand_verts_2d = pred_hand_verts_2d.detach()

        tri_hand_reproj_2d = preds["tri_hand_reproj_pixel"]
        tri_hand_conf = preds["conf_hand_tri"]

        pred_obj_points = preds["obj_points_cam"]
        gt_obj_points = self._build_gt_object_points(batch, dtype=pred_obj_points.dtype, device=pred_obj_points.device)
        if gt_obj_points is None and batch.get("target_obj_kp21", None) is not None:
            gt_obj_points = batch["target_obj_kp21"].to(device=pred_obj_points.device, dtype=pred_obj_points.dtype)
        pred_obj_2d = batch_cam_intr_projection(batch["target_cam_intr"], pred_obj_points)
        gt_obj_2d = batch_cam_intr_projection(batch["target_cam_intr"], gt_obj_points) if gt_obj_points is not None else pred_obj_2d.detach()

        pred_obj_kp2d = preds["pred_obj_kp2d_pixel"]
        tri_obj_kp2d = preds["tri_obj_reproj_pixel"]
        obj_tri_conf = preds["conf_obj_tri"]
        if batch.get("target_obj_kp21", None) is not None:
            gt_obj_kp2d = batch_cam_intr_projection(
                batch["target_cam_intr"],
                batch["target_obj_kp21"].to(device=pred_obj_kp2d.device, dtype=pred_obj_kp2d.dtype),
            )
        else:
            gt_obj_kp2d = pred_obj_kp2d.detach()

        view_rot_err = None
        view_trans_err = None
        obj_point_err = None
        gt_obj_rot6d, gt_obj_trans = self._get_gt_object_pose_abs(batch, dtype=pred_obj_kp2d.dtype, device=pred_obj_kp2d.device)
        if gt_obj_rot6d is not None and gt_obj_trans is not None:
            view_rot_err = torch.rad2deg(
                self.rotation_geodesic(
                    preds["obj_pose_rot6d_cam"].reshape(-1, 6),
                    gt_obj_rot6d.reshape(-1, 6),
                )
            ).view(batch_size, num_views)
            view_trans_err = torch.linalg.norm(
                preds["obj_pose_trans_cam"] - gt_obj_trans,
                dim=-1,
            ) * 1000.0
        if gt_obj_points is not None:
            obj_point_err = torch.linalg.norm(
                pred_obj_points - gt_obj_points,
                dim=-1,
            ).mean(dim=-1) * 1000.0

        hand_err_px = self._mean_pixel_error(pred_hand_joints_2d, gt_hand_joints_2d)
        hand_tri_err_px = self._mean_pixel_error(tri_hand_reproj_2d, gt_hand_joints_2d)
        obj_kp_err_px = self._mean_pixel_error(pred_obj_kp2d, gt_obj_kp2d)
        obj_tri_err_px = self._mean_pixel_error(tri_obj_kp2d, gt_obj_kp2d)

        batch_id, view_ids, selection_desc = self._select_visualization_batch_views(
            mode=mode,
            num_views=num_views,
            hand_err_px=hand_err_px,
            hand_tri_err_px=hand_tri_err_px,
            obj_kp_err_px=obj_kp_err_px,
            obj_tri_err_px=obj_tri_err_px,
            obj_point_err_mm=obj_point_err,
            view_rot_err_deg=view_rot_err,
            view_trans_err_mm=view_trans_err,
        )
        if selection_desc is not None and self.vis_selection_log:
            logger.info(f"[{mode.upper()}Vis] step={step_idx} {selection_desc}")

        img_views = img[batch_id, view_ids]
        cam_intr = batch["target_cam_intr"][batch_id, view_ids]
        pred_hand_verts_2d_sel = pred_hand_verts_2d[batch_id, view_ids]
        pred_hand_joints_2d_sel = pred_hand_joints_2d[batch_id, view_ids]
        gt_hand_joints_2d_sel = gt_hand_joints_2d[batch_id, view_ids]
        gt_hand_verts_2d_sel = gt_hand_verts_2d[batch_id, view_ids]
        tri_hand_reproj_2d_sel = tri_hand_reproj_2d[batch_id, view_ids]
        tri_hand_conf_sel = tri_hand_conf[batch_id, view_ids]
        pred_obj_2d_sel = pred_obj_2d[batch_id, view_ids]
        gt_obj_2d_sel = gt_obj_2d[batch_id, view_ids]
        pred_obj_kp2d_sel = pred_obj_kp2d[batch_id, view_ids]
        tri_obj_kp2d_sel = tri_obj_kp2d[batch_id, view_ids]
        obj_tri_conf_sel = obj_tri_conf[batch_id, view_ids]
        gt_obj_kp2d_sel = gt_obj_kp2d[batch_id, view_ids]
        gt_obj_center_2d = gt_obj_kp2d_sel[:, -1:] if gt_obj_kp2d_sel is not None else None
        pred_obj_center_2d = pred_obj_kp2d_sel[:, -1:]
        view_rot_err_sel = view_rot_err[batch_id, view_ids].unsqueeze(-1) if view_rot_err is not None else None
        view_trans_err_sel = view_trans_err[batch_id, view_ids].unsqueeze(-1) if view_trans_err is not None else None
        obj_point_err_sel = obj_point_err[batch_id, view_ids].unsqueeze(-1) if obj_point_err is not None else None
        title_suffix = f" | {selection_desc}" if selection_desc is not None else ""

        hand_joint_tile = tile_batch_images(
            draw_batch_joint_images(
                pred_hand_joints_2d_sel,
                gt_hand_joints_2d_sel,
                img_views,
                step_idx,
                n_sample=num_vis_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/hand_j2d", hand_joint_tile, step_idx, dataformats="HWC")

        tri_hand_tile = tile_batch_images(
            draw_batch_joint_images(
                tri_hand_reproj_2d_sel,
                pred_hand_joints_2d_sel,
                img_views,
                step_idx,
                n_sample=num_vis_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/hand_tri", tri_hand_tile, step_idx, dataformats="HWC")

        hand_conf_tile = tile_batch_images(
            draw_batch_joint_confidence_images(
                pred_hand_joints_2d_sel,
                tri_hand_conf_sel,
                img_views,
                n_sample=num_vis_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/hand_conf", hand_conf_tile, step_idx, dataformats="HWC")

        hand_mesh_tile = tile_batch_images(
            draw_batch_hand_mesh_images_2d(
                gt_verts2d=gt_hand_verts_2d_sel,
                pred_verts2d=pred_hand_verts_2d_sel,
                face=self.face,
                tensor_image=img_views,
                n_sample=num_vis_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/hand_mesh", hand_mesh_tile, step_idx, dataformats="HWC")

        ho_tile = tile_batch_images(
            draw_batch_mesh_images_pred(
                gt_verts2d=gt_hand_verts_2d_sel,
                pred_verts2d=pred_hand_verts_2d_sel,
                face=self.face,
                gt_obj2d=gt_obj_2d_sel,
                pred_obj2d=pred_obj_2d_sel,
                gt_objc2d=gt_obj_center_2d,
                pred_objc2d=pred_obj_center_2d,
                intr=cam_intr,
                tensor_image=img_views,
                pred_obj_error=obj_point_err_sel,
                pred_obj_rot_error=view_rot_err_sel,
                pred_obj_trans_error=view_trans_err_sel,
                n_sample=num_vis_views,
            )
        )
        self.summary.add_image(f"vis/{mode}/ho", ho_tile, step_idx, dataformats="HWC")
        self._export_visualization_png(ho_tile, "ho", mode, step_idx)

        obj_pred_tile = tile_batch_images(
            draw_batch_object_kp_images(
                pred_obj_kp2d_sel,
                img_views,
                n_sample=num_vis_views,
                title="Pred Object KP21",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_pred", obj_pred_tile, step_idx, dataformats="HWC")

        obj_tri_tile = tile_batch_images(
            draw_batch_object_kp_images(
                tri_obj_kp2d_sel,
                img_views,
                n_sample=num_vis_views,
                title="Tri Reproj Object KP21",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_tri", obj_tri_tile, step_idx, dataformats="HWC")

        obj_gt_tile = tile_batch_images(
            draw_batch_object_kp_images(
                gt_obj_kp2d_sel,
                img_views,
                n_sample=num_vis_views,
                title="GT Object KP21",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_gt", obj_gt_tile, step_idx, dataformats="HWC")

        obj_conf_tile = tile_batch_images(
            draw_batch_object_kp_confidence_images(
                pred_obj_kp2d_sel,
                obj_tri_conf_sel,
                img_views,
                n_sample=num_vis_views,
                title="Object Tri Confidence",
            )
        )
        self.summary.add_image(f"vis/{mode}/obj_kp_conf", obj_conf_tile, step_idx, dataformats="HWC")
        self._export_visualization_png(obj_conf_tile, "obj_conf", mode, step_idx)

        overview_tile = self._build_visualization_overview(
            [
                (f"Hand J2D{title_suffix}", hand_joint_tile),
                ("Hand Tri", tri_hand_tile),
                ("Hand Conf", hand_conf_tile),
                ("Hand Mesh", hand_mesh_tile),
                ("Hand-Object", ho_tile),
                ("Obj Pred", obj_pred_tile),
                ("Obj Tri", obj_tri_tile),
                ("Obj GT", obj_gt_tile),
                ("Obj Conf", obj_conf_tile),
            ],
            n_cols=2,
            max_long_side=2400,
        )
        if overview_tile is not None:
            self.summary.add_image(f"vis/{mode}/overview", overview_tile, step_idx, dataformats="HWC")
            self._export_visualization_png(overview_tile, "overview", mode, step_idx)

        if hasattr(self.summary, "flush"):
            self.summary.flush()

    def training_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        _, loss_dict = self.compute_loss(preds, batch)
        metric_dict = self.compute_sv_object_metrics(preds, batch)
        self.loss_metric.feed({**loss_dict, **metric_dict}, batch_size=batch["image"].size(0))
        self._update_sv_metrics(preds, batch, use_pa=False)
        if step_idx % self.train_log_interval == 0:
            self._write_scalars("", {**loss_dict, **metric_dict}, step_idx)
            if self.summary is not None:
                self.summary.add_scalar("MPJPE_SV_3D", self.MPJPE_SV_3D.get_result(), step_idx)
                self.summary.add_scalar("MPVPE_SV_3D", self.MPVPE_SV_3D.get_result(), step_idx)
                self.summary.add_scalar("OBJREC_SV_CD", self.OBJ_RECON_SV.cd.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_SV_ADD", self.OBJ_POSE_VAL.add.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_SV_ADDS", self.OBJ_POSE_VAL.adds.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_SV_ROT", self.OBJ_POSE_VAL.rot_deg.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_SV_TRANS", self.OBJ_POSE_VAL.trans_epe.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_INIT_ADD", self.OBJ_POSE_INIT.add.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_INIT_ADDS", self.OBJ_POSE_INIT.adds.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_INIT_ROT", self.OBJ_POSE_INIT.rot_deg.avg, step_idx)
                self.summary.add_scalar("OBJPOSE_INIT_TRANS", self.OBJ_POSE_INIT.trans_epe.avg, step_idx)
        if step_idx % self.vis_log_interval == 0:
            self._log_visualizations("train", batch, preds, step_idx)
        return preds, loss_dict

    def validation_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        _, loss_dict = self.compute_loss(preds, batch)
        metric_dict = self.compute_sv_object_metrics(preds, batch)
        self.loss_metric.feed({**loss_dict, **metric_dict}, batch_size=batch["image"].size(0))
        self._update_sv_metrics(preds, batch, use_pa=True)
        if self.summary is not None:
            self._write_scalars("val_", {**loss_dict, **metric_dict}, step_idx)
            self.summary.add_scalar("MPJPE_SV_3D_val", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D_val", self.MPVPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("PA_SV_val", self.PA_SV.get_result(), step_idx)
            self.summary.add_scalar("OBJREC_SV_CD_val", self.OBJ_RECON_SV.cd.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_5_val", self.OBJ_RECON_SV.fs_5.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_10_val", self.OBJ_RECON_SV.fs_10.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_SV_ADD_val", self.OBJ_POSE_VAL.add.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_SV_ADDS_val", self.OBJ_POSE_VAL.adds.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_SV_ROT_val", self.OBJ_POSE_VAL.rot_deg.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_SV_TRANS_val", self.OBJ_POSE_VAL.trans_epe.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_INIT_ADD_val", self.OBJ_POSE_INIT.add.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_INIT_ADDS_val", self.OBJ_POSE_INIT.adds.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_INIT_ROT_val", self.OBJ_POSE_INIT.rot_deg.avg, step_idx)
            self.summary.add_scalar("OBJPOSE_INIT_TRANS_val", self.OBJ_POSE_INIT.trans_epe.avg, step_idx)
        if step_idx % self.val_vis_log_interval == 0:
            self._log_visualizations("val", batch, preds, step_idx)
        return preds, loss_dict

    def testing_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        return preds, {}

    def inference_step(self, batch, step_idx, **kwargs):
        return self._forward_impl(batch)

    def draw_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        self._log_visualizations("draw", batch, preds, step_idx)
        return preds

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        if mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        if mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        if mode == "draw":
            return self.draw_step(inputs, step_idx, **kwargs)
        if mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        raise ValueError(f"Unknown mode {mode}")

    def format_metric(self, mode, **kwargs):
        compact = bool(kwargs.get("compact", True))
        max_len = kwargs.get("max_len", 118 if compact else None)

        def _get_loss(name, default=0.0):
            meter = self.loss_metric._losses.get(name, None)
            return float(meter.avg) if meter is not None else float(default)

        def _mm(value):
            return float(value) * 1000.0

        def _pair_mm(v1, v2):
            return f"{_mm(v1):.1f}/{_mm(v2):.1f}"

        def _pair_raw(v1, v2):
            return f"{float(v1):.3f}/{float(v2):.3f}"

        def _clip(text):
            if max_len is None or len(text) <= int(max_len):
                return text
            return text[: max(0, int(max_len) - 3)] + "..."

        if compact:
            if mode == "train":
                text = (
                    f"Init L {_get_loss('loss'):.3f} | "
                    f"Px H/O {_get_loss('metric_hand_2d_px'):.1f}/{_get_loss('metric_obj_kp2d_px'):.1f} | "
                    f"SV R/T {self.OBJ_POSE_VAL.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_VAL.trans_epe.avg):.1f} | "
                    f"M R/T {self.OBJ_POSE_INIT.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_INIT.trans_epe.avg):.1f} | "
                    f"Tri H/O {_get_loss('metric_tri_hand_px'):.1f}/{_get_loss('metric_tri_obj_px'):.1f}"
                )
                return _clip(text)

            text = (
                f"PA {_pair_mm(self.PA_SV.pa_mpjpe.avg, self.PA_SV.pa_mpvpe.avg)} | "
                f"KP {_pair_mm(self.MPJPE_SV_3D.get_result(), self.MPVPE_SV_3D.get_result())} | "
                f"Px H/O {_get_loss('metric_hand_2d_px'):.1f}/{_get_loss('metric_obj_kp2d_px'):.1f} | "
                f"SV R/T {self.OBJ_POSE_VAL.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_VAL.trans_epe.avg):.1f} | "
                f"M R/T {self.OBJ_POSE_INIT.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_INIT.trans_epe.avg):.1f}"
            )
            return _clip(text)

        if mode == "train":
            return (
                f"Init | L {_get_loss('loss'):.3f} | "
                f"H {_get_loss('loss_hand_2d'):.3f}/{_get_loss('loss_pose_reg'):.3f}/{_get_loss('loss_shape_reg'):.3f} | "
                f"O {_get_loss('loss_obj_2d'):.3f}/{_get_loss('loss_obj_rot_geo'):.3f}/{_get_loss('loss_obj_rot_l1'):.3f}/"
                f"{_get_loss('loss_obj_trans'):.3f}/{_get_loss('loss_obj_points_3d'):.3f}/{_get_loss('loss_obj_points_2d'):.3f} | "
                f"Px {_get_loss('metric_hand_2d_px'):.1f}/{_get_loss('metric_obj_kp2d_px'):.1f}/{_get_loss('metric_obj_points_2d_px'):.1f} | "
                f"ObjSV A/S {_pair_mm(self.OBJ_POSE_VAL.add.avg, self.OBJ_POSE_VAL.adds.avg)} | "
                f"Rot/Tr {self.OBJ_POSE_VAL.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_VAL.trans_epe.avg):.1f} | "
                f"ObjM A/S {_pair_mm(self.OBJ_POSE_INIT.add.avg, self.OBJ_POSE_INIT.adds.avg)} | "
                f"Rot/Tr {self.OBJ_POSE_INIT.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_INIT.trans_epe.avg):.1f} | "
                f"Tri H {_get_loss('metric_tri_hand_conf'):.3f}/{_get_loss('metric_tri_hand_px'):.1f}px "
                f"O {_get_loss('metric_tri_obj_conf'):.3f}/{_get_loss('metric_tri_obj_px'):.1f}px | "
                f"Rec {_pair_raw(self.OBJ_RECON_SV.fs_5.avg, self.OBJ_RECON_SV.fs_10.avg)}/{self.OBJ_RECON_SV.cd.avg:.3f}"
            )

        return (
            f"PA {_pair_mm(self.PA_SV.pa_mpjpe.avg, self.PA_SV.pa_mpvpe.avg)} | "
            f"KP {_pair_mm(self.MPJPE_SV_3D.get_result(), self.MPVPE_SV_3D.get_result())} | "
            f"Px {_get_loss('metric_hand_2d_px'):.1f}/{_get_loss('metric_obj_kp2d_px'):.1f}/{_get_loss('metric_obj_points_2d_px'):.1f} | "
            f"ObjSV A/S {_pair_mm(self.OBJ_POSE_VAL.add.avg, self.OBJ_POSE_VAL.adds.avg)} | "
            f"Rot/Tr {self.OBJ_POSE_VAL.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_VAL.trans_epe.avg):.1f} | "
            f"ObjM A/S {_pair_mm(self.OBJ_POSE_INIT.add.avg, self.OBJ_POSE_INIT.adds.avg)} | "
            f"Rot/Tr {self.OBJ_POSE_INIT.rot_deg.avg:.1f}/{_mm(self.OBJ_POSE_INIT.trans_epe.avg):.1f} | "
            f"Tri H {_get_loss('metric_tri_hand_conf'):.3f}/{_get_loss('metric_tri_hand_px'):.1f}px "
            f"O {_get_loss('metric_tri_obj_conf'):.3f}/{_get_loss('metric_tri_obj_px'):.1f}px | "
            f"Rec {_pair_raw(self.OBJ_RECON_SV.fs_5.avg, self.OBJ_RECON_SV.fs_10.avg)}/{self.OBJ_RECON_SV.cd.avg:.3f}"
        )

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric(
            [self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.OBJ_RECON_SV, self.OBJ_POSE_VAL, self.OBJ_POSE_INIT],
            epoch_idx,
            comment=comment,
            summary=self.format_metric("train", compact=False),
        )
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.PA_SV.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_POSE_VAL.reset()
        self.OBJ_POSE_INIT.reset()

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric(
            [self.PA_SV, self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.OBJ_RECON_SV, self.OBJ_POSE_VAL, self.OBJ_POSE_INIT],
            epoch_idx,
            comment=comment,
            summary=self.format_metric("val", compact=False),
        )
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.PA_SV.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_POSE_VAL.reset()
        self.OBJ_POSE_INIT.reset()
