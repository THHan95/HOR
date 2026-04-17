import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
# import torchvision.models as models

from ..metrics.basic_metric import LossMetric
from ..metrics.mean_epe import MeanEPE
from ..metrics.object_pose_metric import ObjectPoseMetric
from ..metrics.object_recon_metric import ObjectReconMetric
from ..metrics.pa_eval import PAEval
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import param_size
from ..utils.net_utils import init_weights, constant_init
from ..utils.recorder import Recorder
from ..utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, batch_persp_project, mano_to_openpose,
                               rot6d_to_rotmat, rotmat_to_rot6d)
from ..viztools.draw import (
    draw_batch_hand_mesh_images_2d,
    draw_batch_joint_confidence_images,
    draw_batch_joint_images,
    draw_batch_mesh_images_pred,
    draw_batch_obj_view_rotation_images,
    tile_batch_images,
)
# from .common.networks.tgs.models.snowflake.model_spdpp import SnowflakeModelSPDPP
# from .common.networks.tgs.models.snowflake.model_spdpp import mask_generation
from .backbones import build_backbone
from .bricks.conv import ConvBlock
from .bricks.utils import ManoDecoder, SelfAttn, HOT, GraphRegression, Proj2World, orthgonalProj
from .model_abstraction import ModuleAbstract
from .heads import build_head
from .integal_pose import integral_heatmap2d
from pytorch3d.loss import chamfer_distance
from lib.criterions.emd.emd import earth_mover_distance


@MODEL.register_module()
class POEM_Heatmap(nn.Module, ModuleAbstract):
    def __init__(self, cfg):
        super(POEM_Heatmap, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET
        self.debug_logs = bool(self.train_cfg.get("DEBUG_LOGS", False))
        # self.num_joints = cfg.DATA_PRESET.NUM_JOINTS
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        # 从配置中读取参数 (这里给出默认值以防报错)
        self.mode = cfg.get("DATA_PRESET.MODE", "mv")
        self.num_hand_joints = cfg.get("NUM_HAND_JOINTS", 21)
        self.num_hand_verts = cfg.get("NUM_HAND_VERTS", 778)
        self.num_obj_joints = cfg.get("NUM_OBJ_JOINTS", 1)
        self.joints_loss_type = cfg.LOSS.get("JOINTS_LOSS_TYPE", "l2")
        self.verts_loss_type = cfg.LOSS.get("VERTICES_LOSS_TYPE", "l1")
        self.feat_dim = 512  # ResNet34 最终输出的通道数
        self.hot_dim = 384
        self.conf_heatmap_spread_tau_px = float(cfg.get("CONF_HEATMAP_SPREAD_TAU_PX", 24.0))

        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)
        assert self.img_backbone.name in ["resnet18", "resnet34", "resnet50"], "Wrong backbone for PETR"
        # The classification FC head from torchvision ResNet is never used in this model.
        # Freeze it explicitly so DDP does not expect gradients for dead parameters.
        if hasattr(self.img_backbone, "fc") and isinstance(self.img_backbone.fc, nn.Module):
            for param in self.img_backbone.fc.parameters():
                param.requires_grad = False
        if self.img_backbone.name == "resnet18":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet34":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet50":
            self.feat_size = (2048, 1024, 512, 256)

        self.fc_layers = []

        self.feat_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True, norm='bn'),
        ])
        self.feat_in = ConvBlock(self.feat_size[3], self.feat_size[2], kernel_size=1, padding=0, relu=False, norm=None)

        if self.mode == 'mv':
            self.project = Proj2World(self.num_hand_joints, self.num_obj_joints, self.data_preset_cfg.IMAGE_SIZE)

            self.ptEmb_head = build_head(cfg.HEAD, data_preset=self.data_preset_cfg)
            self.num_preds = self.ptEmb_head.num_preds

        self.hot_pose = HOT(self.feat_size[2], self.hot_dim, self.num_hand_joints, self.num_obj_joints)

        self.att_0 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0.1)
        self.att_1 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0.1)

        # mano parameters regression
        self.mano_fuse = GraphRegression(self.num_hand_joints + self.num_obj_joints, self.feat_dim, 32, last=False)
        self.mano_fc = nn.Sequential(
            nn.Linear(32 * (self.num_hand_joints + self.num_obj_joints), 32 * (self.num_hand_joints + self.num_obj_joints)),
            nn.LeakyReLU(0.1),
            nn.Linear(32 * (self.num_hand_joints + self.num_obj_joints), 16 * 6 + 10 + 3),
        )
        self.mano_decoder = ManoDecoder(
            self.data_preset_cfg.CENTER_IDX,
            self.data_preset_cfg.BBOX_3D_SIZE,
            self.data_preset_cfg.IMAGE_SIZE,
        )
        self.face = self.mano_decoder.face
        self.sv_obj_feat_abs_max = 1e4
        self.sv_obj_rot6d_abs_max = 10.0
        self.sv_obj_trans_abs_max = self.data_preset_cfg.BBOX_3D_SIZE * 8.0
        self.sv_obj_global_dim = self.feat_size[0]
        self.sv_obj_local_dim = 32 * (self.num_hand_joints + self.num_obj_joints)
        self.sv_obj_pose_input_dim = self.sv_obj_global_dim + self.sv_obj_local_dim
        self.sv_object_pose_feat = nn.Sequential(
            nn.Linear(self.sv_obj_pose_input_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
        )
        self.sv_object_rot_regressor = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 6),
        )
        self.sv_object_trans_regressor = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 3),
        )
        self.sv_pose_init_token_dim = self.feat_dim // 2
        self.sv_object_init_pose_token = nn.Sequential(
            nn.Linear(9, self.sv_pose_init_token_dim),
            nn.ReLU(),
            nn.Linear(self.sv_pose_init_token_dim, self.sv_pose_init_token_dim),
            nn.ReLU(),
        )
        self.sv_object_init_pose_head = nn.Sequential(
            nn.Linear(self.sv_pose_init_token_dim * 2 + 9, self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 9),
        )
        self.register_buffer("identity_rot6d", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32))

        # self.pointcloud_upsampler = SnowflakeModelSPDPP(
        #     input_channels=256,
        #     dim_feat=256,
        #     num_p0=2048,
        #     up_factors=[2, 4],
        #     SPD_type='SPD_PP',
        #     token_type='image_token'
        # )
        
        self._build_heads()

        if self.joints_loss_type == "l2":
            self.criterion_joints = torch.nn.MSELoss()
        else:
            self.criterion_joints = torch.nn.L1Loss()

        self.coord_loss = nn.L1Loss()
        self.heatmap_hand_weight = cfg.LOSS.get("HEATMAP_HAND_N", cfg.LOSS.get("HEATMAP_JOINTS_WEIGHT", 1.0))
        self.heatmap_obj_weight = cfg.LOSS.get("HEATMAP_OBJ_N", self.heatmap_hand_weight)
        self.heatmap_hand_map_weight = cfg.LOSS.get("HEATMAP_HAND_MAP_N", 1.0)
        self.heatmap_obj_map_weight = cfg.LOSS.get("HEATMAP_OBJ_MAP_N", self.heatmap_hand_map_weight)
        self.chamfer_loss = chamfer_distance
        self.earth_mover_loss = earth_mover_distance
        self.obj_pose_rot_weight = cfg.LOSS.get("OBJ_POSE_ROT_N", 1.0)
        self.obj_pose_rot6d_weight = cfg.LOSS.get("OBJ_POSE_ROT6D_N", 0.1)
        self.obj_init_rot_weight = cfg.LOSS.get("OBJ_INIT_ROT_N", 2.0)
        self.obj_init_rot6d_weight = cfg.LOSS.get("OBJ_INIT_ROT6D_N", self.obj_pose_rot6d_weight)
        self.obj_view_rot_weight = cfg.LOSS.get("OBJ_VIEW_ROT_N", 1.0)
        self.obj_view_rot6d_weight = cfg.LOSS.get("OBJ_VIEW_ROT6D_N", self.obj_pose_rot6d_weight)
        self.obj_pose_trans_weight = cfg.LOSS.get("OBJ_POSE_TRANS_N", 5.0)
        self.obj_init_trans_weight = cfg.LOSS.get("OBJ_INIT_TRANS_N", self.obj_pose_trans_weight)
        self.obj_view_trans_weight = cfg.LOSS.get("OBJ_VIEW_TRANS_N", self.obj_pose_trans_weight)
        self.obj_pose_points_weight = cfg.LOSS.get("OBJ_POSE_POINTS_N", 10.0)
        self.obj_chamfer_weight = cfg.LOSS.get("OBJ_CHAMFER_N", 5.0)
        self.obj_emd_weight = cfg.LOSS.get("OBJ_EMD_N", 1.0)
        self.obj_penetration_weight = cfg.LOSS.get("OBJ_PENETRATION_N", 5.0)
        self.obj_direct_pose_aux_scale = float(cfg.LOSS.get("OBJ_DIRECT_POSE_AUX_SCALE", 0.1))
        self.decoder_hand_weight = cfg.LOSS.get("DECODER_HAND_N", 10.0)
        self.decoder_proj_weight = cfg.LOSS.get("DECODER_PROJ_N", 10.0)
        self.triangulation_weight = cfg.LOSS.get("TRIANGULATION_N", 10.0)
        self.triangulation_hand_weight = cfg.LOSS.get("TRIANGULATION_HAND_N", self.triangulation_weight)
        self.triangulation_obj_weight = cfg.LOSS.get("TRIANGULATION_OBJ_N", 1.0)
        self.mano_consistency_weight = cfg.LOSS.get("MANO_CONSIST_N", 10.0)
        self.mano_proj_weight = cfg.LOSS.get("MANO_PROJ_N", 50.0)
        self.stage1_end_epoch = cfg.TRAIN.get("STAGE1_END_EPOCH", 4)
        # Keep the old key as a read-only fallback so existing experiment YAMLs still load.
        self.stage2_warmup_epochs = cfg.TRAIN.get(
            "STAGE2_WARMUP_EPOCHS",
            cfg.TRAIN.get("STAGE3_WARMUP_EPOCHS", 3),
        )
        self.sv_self_distill_enabled = bool(cfg.TRAIN.get("SV_SELF_DISTILL_ENABLED", True))
        self.sv_self_distill_start_epoch = int(
            cfg.TRAIN.get(
                "SV_SELF_DISTILL_START_EPOCH",
                self.stage1_end_epoch + max(int(self.stage2_warmup_epochs), 0),
            )
        )
        self.val_recon_full_only_final = bool(cfg.TRAIN.get("VAL_RECON_FULL_ONLY_FINAL", True))
        self.current_stage_name = "stage2"
        self.current_stage2_warmup = 1.0
        self.current_val_recon_mode = "fast"
        self._debug_stage_log_state = {"train": None, "val": None}

        self.loss_metric = LossMetric(cfg)
        self.PA_SV = PAEval(cfg, mesh_score=True)
        self.PA_KP_MASTER = PAEval(cfg, mesh_score=True)
        self.PA_MESH_MASTER = PAEval(cfg, mesh_score=True)
        self.OBJ_POSE_VAL = ObjectPoseMetric(cfg, name="Obj")
        self.OBJ_RECON_SV = ObjectReconMetric(cfg, name="SVObjRec")
        self.OBJ_RECON_MASTER = ObjectReconMetric(cfg, name="MasterObjRec")
        self.MPJPE_MASTER_3D = MeanEPE(cfg, "Master_J")
        self.MPVPE_MASTER_3D = MeanEPE(cfg, "Master_V")
        self.MPJPE_KP_MASTER_3D = MeanEPE(cfg, "KP_Master_J")
        self.MPVPE_KP_MASTER_3D = MeanEPE(cfg, "KP_Master_V")
        self.MPJPE_MESH_MASTER_3D = MeanEPE(cfg, "Mesh_Master_J")
        self.MPVPE_MESH_MASTER_3D = MeanEPE(cfg, "Mesh_Master_V")
        self.MPJPE_SV_3D = MeanEPE(cfg, "SV_J")
        self.MPVPE_SV_3D = MeanEPE(cfg, "SV_V")
        self.MPRPE_HAND_3D = MeanEPE(cfg, "Ref_J")
        # self.MPJPE_3D_REF = MeanEPE(cfg, "joints_3d_ref")
        # self.MPVPE_3D = MeanEPE(cfg, "vertices_3d")
        # self.MPJPE_3D_REL = MeanEPE(cfg, "joints_3d_rel")
        # self.MPVPE_3D_REL = MeanEPE(cfg, "vertices_3d_rel")
        # self.MPTPE_3D = MeanEPE(cfg, "triangulate_joints")

        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.init_weights()

        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} loss type: joint {self.joints_loss_type} verts {self.verts_loss_type}")

    def init_weights(self):
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        init_weights(self, pretrained=self.cfg.PRETRAINED)

    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def set_train_stage(self, stage_name: str):
        self.current_stage_name = stage_name

        # Default: enable everything, then selectively freeze by stage.
        self._set_module_requires_grad(self.img_backbone, True)
        if hasattr(self.img_backbone, "fc") and isinstance(self.img_backbone.fc, nn.Module):
            self._set_module_requires_grad(self.img_backbone.fc, False)
        self._set_module_requires_grad(self.feat_delayer, True)
        self._set_module_requires_grad(self.feat_in, True)
        self._set_module_requires_grad(self.uv_delayer, True)
        self._set_module_requires_grad(self.uv_out, True)
        self._set_module_requires_grad(self.hot_pose, True)
        self._set_module_requires_grad(self.att_0, True)
        self._set_module_requires_grad(self.att_1, True)
        self._set_module_requires_grad(self.mano_fuse, True)
        self._set_module_requires_grad(self.mano_fc, True)
        self._set_module_requires_grad(self.sv_object_pose_feat, True)
        self._set_module_requires_grad(self.sv_object_rot_regressor, True)
        self._set_module_requires_grad(self.sv_object_trans_regressor, True)
        self._set_module_requires_grad(self.ptEmb_head, True)
        self._set_module_requires_grad(getattr(self.ptEmb_head, "obj_feat_fuser", None), False)
        if hasattr(self.ptEmb_head, "center_shift_layer"):
            self._set_module_requires_grad(self.ptEmb_head.center_shift_layer, False)
        if hasattr(self.ptEmb_head, "position_encoder"):
            self._set_module_requires_grad(self.ptEmb_head.position_encoder, False)
        if hasattr(self.ptEmb_head, "hand_transformer"):
            for block in self.ptEmb_head.hand_transformer._iter_blocks():
                self._set_module_requires_grad(block.hand_anchor_update_layer, False)
                self._set_module_requires_grad(block.hand_refine_layer, False)
        if hasattr(self.ptEmb_head, "ho_transformer") and hasattr(self.ptEmb_head.ho_transformer, "_iter_blocks"):
            for block in self.ptEmb_head.ho_transformer._iter_blocks():
                self._set_module_requires_grad(block.obj_update_layer.vec_attn.reg_branch, False)

        if stage_name == "stage1":
            # Stage1 trains single-view branches + keypoint-master hand refinement,
            # but still learns per-view SV object 6D pose without HO interaction refinement.
            self._set_module_requires_grad(getattr(self.ptEmb_head, "ho_transformer", None), False)
            self._set_module_requires_grad(getattr(self.ptEmb_head, "obj_feat_fuser", None), False)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        if self.debug_logs:
            logger.warning(
                f"[StageFreeze] stage={stage_name} trainable={trainable_params / 1e6:.2f}M "
                f"frozen={frozen_params / 1e6:.2f}M"
            )

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def extract_img_feat(self, img):
        B = img.size(0)
        if img.dim() == 5:
            if img.size(0) == 1 and img.size(1) != 1:  # (1, N, C, H, W)
                img = img.squeeze()  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(image=img)
        global_feat = img_feats["res_layer4_mean"]  # [B*N,512]
        if isinstance(img_feats, dict):
            """img_feats for ResNet 34: 
                torch.Size([BN, 64, 64, 64])
                torch.Size([BN, 128, 32, 32])
                torch.Size([BN, 256, 16, 16])
                torch.Size([BN, 512, 8, 8])
            """
            img_feats = list([v for v in img_feats.values() if len(v.size()) == 4])

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))  # (B, N, C, H, W)

        return img_feats, img_feats_reshaped, global_feat
    
    def feat_decode(self, mlvl_feats):
        mlvl_feats_rev = list(reversed(mlvl_feats))
        x = mlvl_feats_rev[0]
        for i, fde in enumerate(self.feat_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
            x = fde(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (BxN, 64, 32, 32)
        x = self.feat_in(x)  # (BxN, 128, 32, 32)
        return x

    def heatmap_decode(self, mlvl_feats):
        mlvl_feats_rev = list(reversed(mlvl_feats))
        x = mlvl_feats_rev[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
            x = de(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        uv_hmap = torch.sigmoid(self.uv_out(x) * 0.5)
        uv_hmap = torch.nan_to_num(uv_hmap, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return uv_hmap

    def _build_heads(self):
        total_points = self.num_hand_joints + self.num_obj_joints
        self.uv_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True, norm='bn'),
        ])
        self.uv_out = ConvBlock(self.feat_size[3], total_points, kernel_size=1, padding=0, relu=False, norm=None)
        self.fc_layers = []

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
    def _stage_display_name(stage_name, short=False):
        if stage_name == "stage1":
            return "S1 Hand+SVObjPose" if short else "stage1_hand_sv_obj_pose"
        if stage_name == "stage2":
            return "S2 HO-Pose" if short else "stage2_ho_pose"
        return stage_name

    def _stage_tb_prefix(self, mode, stage_name):
        return f"img/{mode}_{self._stage_display_name(stage_name, short=False)}"

    def _use_full_val_recon(self, epoch_idx):
        if not self.val_recon_full_only_final:
            return True
        total_epochs = int(self.train_cfg.get("EPOCH", 1))
        return epoch_idx >= max(total_epochs - 1, 0)

    @staticmethod
    def _pose_metrics_for_logging(pose_metric_dict):
        return pose_metric_dict

    def _log_sv_view_debug_scalars(self, preds, batch, step_idx, suffix=""):
        return

    @staticmethod
    def _loss_dict_is_finite(loss_dict):
        for value in loss_dict.values():
            if torch.is_tensor(value):
                if not torch.isfinite(value).all():
                    return False
            else:
                value_f = float(value)
                if not math.isfinite(value_f):
                    return False
        return True

    @staticmethod
    def _metric_dict_is_finite(metric_dict):
        if metric_dict is None:
            return True
        for value in metric_dict.values():
            if torch.is_tensor(value):
                if not torch.isfinite(value).all():
                    return False
            else:
                value_f = float(value)
                if not math.isfinite(value_f):
                    return False
        return True

    def _maybe_log_stage_transition(self, mode, epoch_idx, stage_name, preds, batch):
        if not self.debug_logs:
            return
        state_key = f"{mode}:{epoch_idx}:{stage_name}"
        if self._debug_stage_log_state.get(mode) == state_key:
            return
        self._debug_stage_log_state[mode] = state_key

        hand_states = preds.get("hand_mesh_xyz_master", None)
        obj_states = preds.get("obj_xyz_master", None)
        obj_rot_states = preds.get("obj_rot6d", None)
        obj_trans_states = preds.get("obj_trans", None)
        anchor_states = preds.get("all_hand_joints_xyz_master", None)
        interaction_mode = preds.get("interaction_mode", "unknown")

        def _shape(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return [tuple(v.shape) for v in x]
            return tuple(x.shape)

        logger.warning(
            f"[{mode}][StageDebug] epoch={epoch_idx} stage={self._stage_display_name(stage_name, short=True)} "
            f"interaction={interaction_mode} warmup={self.current_stage2_warmup:.2f} "
            f"image={tuple(batch['image'].shape)} "
            f"master_obj_sparse_rest={tuple(batch['master_obj_sparse_rest'].shape) if 'master_obj_sparse_rest' in batch else None} "
            f"ref_hand={tuple(preds['ref_hand'].shape) if 'ref_hand' in preds else None} "
            f"mano_sv={tuple(preds['mano_3d_mesh_sv'].shape) if 'mano_3d_mesh_sv' in preds else None} "
            f"mano_master={tuple(preds['mano_3d_mesh_master'].shape) if 'mano_3d_mesh_master' in preds else None} "
            f"hand_states={_shape(hand_states)} "
            f"anchor_states={_shape(anchor_states)} "
            f"obj_states={_shape(obj_states)} "
            f"obj_rot6d={_shape(obj_rot_states)} "
            f"obj_trans={_shape(obj_trans_states)}"
        )

    def _maybe_log_stage_transition_loss(self, mode, epoch_idx, stage_name, loss_dict, pose_metric_dict=None):
        if not self.debug_logs:
            return
        state_key = f"{mode}:{epoch_idx}:{stage_name}:loss"
        if self._debug_stage_log_state.get(f"{mode}_loss") == state_key:
            return
        self._debug_stage_log_state[f"{mode}_loss"] = state_key

        def _to_float(name):
            value = loss_dict.get(name, None)
            if value is None:
                return None
            return float(value.detach().item()) if torch.is_tensor(value) else float(value)

        metric_obj_rot_deg = None
        metric_obj_trans_epe = None
        if pose_metric_dict is not None:
            if "metric_obj_rot_deg" in pose_metric_dict:
                metric_obj_rot_deg = float(pose_metric_dict["metric_obj_rot_deg"].detach().item())
            if "metric_obj_trans_epe" in pose_metric_dict:
                metric_obj_trans_epe = float(pose_metric_dict["metric_obj_trans_epe"].detach().item())

        logger.warning(
            f"[{mode}][StageDebugLoss] epoch={epoch_idx} stage={self._stage_display_name(stage_name, short=True)} "
            f"loss={(_to_float('loss') or 0.0):.4f} "
            f"hm={((_to_float('loss_heatmap_hand') or 0.0) + (_to_float('loss_heatmap_hand_map') or 0.0)):.4f} "
            f"triang={(_to_float('loss_triang') or 0.0):.4f} "
            f"triang_h={(_to_float('loss_triang_hand') or 0.0):.4f} "
            f"j3d={(_to_float('loss_3d_jts') or 0.0):.4f} "
            f"proj2d={(_to_float('loss_2d_proj') or 0.0):.4f} "
            f"mano_proj={(_to_float('loss_mano_proj') or 0.0):.4f} "
            f"obj_pose={(_to_float('loss_obj_pose') or 0.0):.4f} "
            f"obj_pts={(_to_float('loss_obj_points') or 0.0):.4f} "
            f"obj_chamfer={(_to_float('loss_obj_chamfer') or 0.0):.4f} "
            f"obj_emd={(_to_float('loss_obj_emd') or 0.0):.4f} "
            f"penetration={(_to_float('loss_penetration') or 0.0):.4f} "
            f"rot_deg={f'{metric_obj_rot_deg:.2f}' if metric_obj_rot_deg is not None else 'None'} "
            f"trans_epe={f'{metric_obj_trans_epe:.4f}' if metric_obj_trans_epe is not None else 'None'}"
        )

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
            "metric_sv_obj_rot_l1": zero,
            "metric_sv_obj_rot_deg": zero,
        }

    @staticmethod
    def _normed_uv_to_pixel(coords_uv, image_hw):
        img_h, img_w = image_hw
        scale = coords_uv.new_tensor([img_w, img_h])
        return (coords_uv + 1.0) * 0.5 * scale

    @staticmethod
    def _masked_heatmap_mse(pred_heatmap, gt_heatmap, visibility=None):
        pred_heatmap = torch.nan_to_num(pred_heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        gt_finite = torch.isfinite(gt_heatmap)
        gt_heatmap = torch.nan_to_num(gt_heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        heatmap_diff = (pred_heatmap - gt_heatmap).pow(2)
        valid_mask = gt_finite.to(dtype=pred_heatmap.dtype)
        if visibility is not None:
            vis_mask = visibility.unsqueeze(-1).unsqueeze(-1).to(dtype=pred_heatmap.dtype)
            valid_mask = valid_mask * vis_mask
            heatmap_diff = heatmap_diff * valid_mask
            valid = valid_mask.sum()
            return heatmap_diff.sum() / (valid + 1e-9)
        valid = valid_mask.sum()
        return heatmap_diff.sum() / (valid + 1e-9)

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
    def _sanitize_obj_tensor(tensor, abs_max, nan_value=0.0):
        return torch.nan_to_num(tensor, nan=nan_value, posinf=abs_max, neginf=-abs_max).clamp(-abs_max, abs_max)

    @staticmethod
    def _project_matrix_to_rotation(matrix):
        orig_dtype = matrix.dtype
        compute_dtype = torch.float32 if matrix.dtype in (torch.float16, torch.bfloat16) else matrix.dtype
        matrix_f = matrix.to(dtype=compute_dtype)
        projected = rot6d_to_rotmat(matrix_f[..., :3, :2].reshape(*matrix_f.shape[:-2], 6))
        return projected.to(dtype=orig_dtype)

    def _predict_singleview_object_pose(
        self,
        global_feat,
        hand_obj_fused_flat,
        cam_extr,
        master_id,
    ):
        batch_size, num_cams = cam_extr.shape[:2]
        shared_input = torch.cat((global_feat.float(), hand_obj_fused_flat.float()), dim=-1)
        shared_input = self._sanitize_obj_tensor(shared_input, self.sv_obj_feat_abs_max)
        shared_feat = self.sv_object_pose_feat(shared_input)
        shared_feat = self._sanitize_obj_tensor(shared_feat, self.sv_obj_feat_abs_max)

        pred_rot6d_cam = self._sanitize_obj_tensor(
            self.sv_object_rot_regressor(shared_feat),
            self.sv_obj_rot6d_abs_max,
        ).view(batch_size, num_cams, 6)
        pred_trans_cam = self._sanitize_obj_tensor(
            self.sv_object_trans_regressor(shared_feat),
            self.sv_obj_trans_abs_max,
        ).view(batch_size, num_cams, 3)
        pred_rotmat_cam = rot6d_to_rotmat(pred_rot6d_cam.reshape(-1, 6)).view(batch_size, num_cams, 3, 3)

        cam_to_master = cam_extr[..., :3, :3]
        pred_rotmat_master = torch.matmul(cam_to_master, pred_rotmat_cam)
        pred_rotmat_master = self._project_matrix_to_rotation(pred_rotmat_master.reshape(-1, 3, 3)).view(batch_size, num_cams, 3, 3)
        pred_rot6d_master = self._sanitize_obj_tensor(
            rotmat_to_rot6d(pred_rotmat_master.reshape(-1, 3, 3)).view(batch_size, num_cams, 6),
            self.sv_obj_rot6d_abs_max,
        )
        pred_trans_master = torch.matmul(cam_to_master, pred_trans_cam.unsqueeze(-1)).squeeze(-1)
        pred_trans_master = self._sanitize_obj_tensor(pred_trans_master, self.sv_obj_trans_abs_max)

        pose_master = torch.cat((pred_rot6d_master.detach(), pred_trans_master.detach()), dim=-1)
        pose_master = self._sanitize_obj_tensor(pose_master, self.sv_obj_feat_abs_max)
        pose_tokens = self.sv_object_init_pose_token(pose_master.reshape(-1, 9)).view(batch_size, num_cams, -1)
        pose_tokens = self._sanitize_obj_tensor(pose_tokens, self.sv_obj_feat_abs_max)
        pose_mean = pose_tokens.mean(dim=1)
        pose_max = pose_tokens.max(dim=1).values

        pred_rotmat_master = rot6d_to_rotmat(pred_rot6d_master.detach().reshape(-1, 6)).view(batch_size, num_cams, 3, 3)
        base_rotmat = self._project_matrix_to_rotation(pred_rotmat_master.mean(dim=1))
        base_rot6d = self._sanitize_obj_tensor(rotmat_to_rot6d(base_rotmat), self.sv_obj_rot6d_abs_max)
        base_trans = self._sanitize_obj_tensor(pred_trans_master.detach().mean(dim=1), self.sv_obj_trans_abs_max)

        init_input = torch.cat((pose_mean, pose_max, base_rot6d, base_trans), dim=-1)
        init_input = self._sanitize_obj_tensor(init_input, self.sv_obj_feat_abs_max)
        init_delta = self.sv_object_init_pose_head(init_input)
        init_delta = self._sanitize_obj_tensor(init_delta, self.sv_obj_feat_abs_max)
        delta_rot = torch.tanh(init_delta[:, :6]) * 0.5
        delta_trans = torch.tanh(init_delta[:, 6:]) * self.data_preset_cfg.BBOX_3D_SIZE

        fused_rot_seed = self._sanitize_obj_tensor(base_rot6d + delta_rot, self.sv_obj_rot6d_abs_max)
        fused_rot6d = self._sanitize_obj_tensor(
            rotmat_to_rot6d(rot6d_to_rotmat(fused_rot_seed)),
            self.sv_obj_rot6d_abs_max,
        )
        fused_trans = self._sanitize_obj_tensor(base_trans + delta_trans, self.sv_obj_trans_abs_max)

        return {
            "obj_view_rot6d_cam": pred_rot6d_cam,
            "obj_view_rot6d_master": pred_rot6d_master,
            "obj_view_trans": pred_trans_cam,
            "obj_view_trans_master": pred_trans_master,
            "obj_fused_rot6d_sv": fused_rot6d,
            "obj_fused_trans_sv": fused_trans,
            "obj_init_rot6d": fused_rot6d,
            "obj_init_trans": fused_trans,
        }

    @staticmethod
    def _project_master_points(points_master, T_c2m, K):
        points_mv = points_master.unsqueeze(1).repeat(1, T_c2m.shape[1], 1, 1)
        points_cam = batch_cam_extr_transf(T_c2m, points_mv)
        points_2d = batch_cam_intr_projection(K, points_cam)
        return points_cam, points_2d

    @staticmethod
    def _build_object_points_from_pose(obj_points_rest, rot6d, center):
        obj_points_rest = torch.nan_to_num(obj_points_rest.detach(), nan=0.0, posinf=10.0, neginf=-10.0).float()
        rot6d = torch.nan_to_num(rot6d.detach(), nan=0.0, posinf=10.0, neginf=-10.0).float()
        center = torch.nan_to_num(center.detach(), nan=0.0, posinf=10.0, neginf=-10.0).float()

        if rot6d.dim() == 2:
            rotmat = rot6d_to_rotmat(rot6d)
            return torch.matmul(obj_points_rest, rotmat.transpose(1, 2)) + center.unsqueeze(1)

        batch_size, num_cams = rot6d.shape[:2]
        rotmat = rot6d_to_rotmat(rot6d.reshape(-1, 6)).view(batch_size, num_cams, 3, 3)
        points = obj_points_rest.unsqueeze(1).expand(-1, num_cams, -1, -1)
        if center.dim() == 3:
            center = center.unsqueeze(1)
        return torch.matmul(points, rotmat.transpose(-1, -2)) + center

    @classmethod
    def _build_object_points_from_hand_pose(cls, obj_points_rest, rot6d, hand_root, trans_rel):
        hand_root = torch.nan_to_num(hand_root.detach(), nan=0.0, posinf=10.0, neginf=-10.0).float()
        trans_rel = torch.nan_to_num(trans_rel.detach(), nan=0.0, posinf=10.0, neginf=-10.0).float()
        if rot6d.dim() == 2:
            if hand_root.dim() == 3 and hand_root.shape[-2] == 1:
                hand_root = hand_root.squeeze(-2)
            center = hand_root + trans_rel
        else:
            if hand_root.dim() == 3:
                hand_root = hand_root.unsqueeze(1)
            center = hand_root + trans_rel.unsqueeze(-2)
        return cls._build_object_points_from_pose(obj_points_rest, rot6d, center)

    def _log_visualizations(self, mode, batch, preds, step_idx, stage_name):
        img = batch["image"]
        batch_size, n_views = img.shape[:2]
        img_h, img_w = img.shape[-2:]
        batch_id = 0

        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])

        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"].view(batch_size, n_views, 799, 2)
        pred_mano_2d_joints_sv = pred_mano_2d_mesh_sv[:, :, self.num_hand_verts:]
        pred_mano_2d_verts_sv = pred_mano_2d_mesh_sv[:, :, :self.num_hand_verts]
        pred_mano_cam_sv = preds.get("mano_cam_sv", None)

        pred_ref_joints_2d = preds["pred_hand"].view(batch_size, n_views, self.num_hand_joints, 2)
        pred_hand_conf = preds["conf_hand"].view(batch_size, n_views, self.num_hand_joints, -1).mean(dim=-1)

        gt_joints_2d = self._normed_uv_to_pixel(batch["target_joints_uvd"][..., :2], (img_h, img_w))
        gt_verts_2d_persp = self._normed_uv_to_pixel(batch["target_verts_uvd"][..., :2], (img_h, img_w))
        pred_sv_joints_2d = self._normed_uv_to_pixel(pred_mano_2d_joints_sv, (img_h, img_w))
        pred_sv_verts_2d = self._normed_uv_to_pixel(pred_mano_2d_verts_sv, (img_h, img_w))
        pred_ref_joints_2d = self._normed_uv_to_pixel(pred_ref_joints_2d, (img_h, img_w))

        pred_kp_master_mesh = preds.get("mano_3d_mesh_kp_master", None)
        if pred_kp_master_mesh is not None:
            _, pred_kp_master_mesh_2d = self._project_master_points(pred_kp_master_mesh, T_c2m_gt, K_gt)
            pred_kp_master_joints_2d = pred_kp_master_mesh_2d[:, :, self.num_hand_verts:]
            pred_kp_master_verts_2d = pred_kp_master_mesh_2d[:, :, :self.num_hand_verts]
        else:
            pred_kp_master_joints_2d = None
            pred_kp_master_verts_2d = None

        pred_obj_sparse = preds["obj_xyz_master"][-1]
        _, pred_obj_sparse_2d_proj = self._project_master_points(pred_obj_sparse, T_c2m_gt, K_gt)

        gt_obj_sparse = batch["master_obj_sparse"]
        _, gt_obj_sparse_2d_proj = self._project_master_points(gt_obj_sparse, T_c2m_gt, K_gt)

        gt_obj_sparse_cam = batch.get("target_obj_pc_sparse", None)

        pred_sv_obj_sparse_2d = None
        pred_obj_view_trans_mm = None
        pred_obj_view_rot_deg = None
        if preds.get("obj_view_rot6d_cam", None) is not None and preds.get("obj_view_trans", None) is not None and pred_mano_cam_sv is not None:
            obj_template = batch["master_obj_sparse_rest"].to(device=pred_mano_2d_mesh_sv.device, dtype=pred_mano_2d_mesh_sv.dtype)
            obj_template = obj_template.unsqueeze(1).expand(-1, n_views, -1, -1)
            pred_obj_rotmat_sv = rot6d_to_rotmat(
                preds["obj_view_rot6d_cam"].reshape(-1, 6)
            ).view(batch_size, n_views, 3, 3).to(dtype=obj_template.dtype)
            pred_sv_obj_xyz = torch.matmul(obj_template, pred_obj_rotmat_sv.transpose(-1, -2)) + preds["obj_view_trans"].unsqueeze(-2).to(dtype=obj_template.dtype)
            pred_mano_cam_sv = pred_mano_cam_sv.view(batch_size, n_views, 3).to(dtype=pred_sv_obj_xyz.dtype)
            pred_sv_obj_sparse_2d = orthgonalProj(
                pred_sv_obj_xyz[..., :2].clone(),
                pred_mano_cam_sv[..., 0:1].unsqueeze(-2),
                torch.zeros_like(pred_mano_cam_sv[..., 1:]).unsqueeze(-2),
                img_size=img_w,
            )
            pred_sv_obj_sparse_2d = torch.nan_to_num(pred_sv_obj_sparse_2d, nan=0.0, posinf=max(img_h, img_w), neginf=0.0)

        img_views = img[batch_id]
        gt_joints_2d_views = gt_joints_2d[batch_id]
        pred_sv_joints_2d_views = pred_sv_joints_2d[batch_id]
        pred_sv_verts_2d_views = pred_sv_verts_2d[batch_id]
        pred_ref_joints_2d_views = pred_ref_joints_2d[batch_id]
        pred_hand_conf_views = pred_hand_conf[batch_id]
        master_view_mask = torch.zeros(n_views, 1, dtype=img.dtype, device=img.device)
        master_view_mask[batch["master_id"][batch_id].item(), 0] = 1.0
        if preds.get("obj_view_rot6d_cam", None) is not None and batch.get("target_rot6d_label", None) is not None:
            gt_view_rot6d = batch["target_rot6d_label"][batch_id]
            pred_view_rot6d = preds["obj_view_rot6d_cam"][batch_id]
            pred_obj_view_rot_deg = torch.rad2deg(
                self.rotation_geodesic(pred_view_rot6d, gt_view_rot6d)
            ).unsqueeze(-1)
        if preds.get("obj_view_trans", None) is not None and batch.get("target_t_label_rel", None) is not None:
            pred_obj_view_trans_mm = torch.linalg.norm(
                preds["obj_view_trans"][batch_id] - batch["target_t_label_rel"][batch_id].to(device=preds["obj_view_trans"].device, dtype=preds["obj_view_trans"].dtype),
                dim=-1,
            ).unsqueeze(-1) * 1000.0
        pred_obj_master_rot_deg = None
        pred_obj_master_trans_mm = None
        if preds.get("obj_rot6d", None) is not None and batch.get("master_obj_rot6d_label", None) is not None:
            pred_obj_master_rot_deg = torch.rad2deg(
                self.rotation_geodesic(
                    preds["obj_rot6d"][-1][batch_id:batch_id + 1],
                    batch["master_obj_rot6d_label"][batch_id:batch_id + 1],
                )
            ).unsqueeze(-1).repeat(n_views, 1)
        if preds.get("obj_trans", None) is not None and batch.get("master_obj_t_label_rel", None) is not None:
            pred_obj_master_trans_mm = torch.linalg.norm(
                preds["obj_trans"][-1][batch_id] - batch["master_obj_t_label_rel"][batch_id].to(device=preds["obj_trans"][-1].device, dtype=preds["obj_trans"][-1].dtype),
                dim=-1,
            ).unsqueeze(-1).repeat(n_views, 1) * 1000.0
        pred_obj_init_sparse_2d_views = None
        pred_obj_init_rot_deg = None
        pred_obj_init_trans_mm = None
        if stage_name == "stage2" and preds.get("obj_init_rot6d", None) is not None and preds.get("obj_init_trans", None) is not None:
            obj_template_master = batch["master_obj_sparse_rest"].to(device=pred_mano_2d_mesh_sv.device, dtype=pred_mano_2d_mesh_sv.dtype)
            hand_master_mesh = preds.get("mano_3d_mesh_kp_master", None)
            if hand_master_mesh is None:
                hand_master_mesh = preds["mano_3d_mesh_master"]
            hand_master_root = hand_master_mesh[:, self.num_hand_verts + self.center_idx:self.num_hand_verts + self.center_idx + 1]
            pred_obj_init_xyz = self._build_object_points_from_hand_pose(
                obj_template_master,
                preds["obj_init_rot6d"],
                hand_master_root,
                preds["obj_init_trans"],
            )
            _, pred_obj_init_sparse_2d = self._project_master_points(pred_obj_init_xyz, T_c2m_gt, K_gt)
            pred_obj_init_sparse_2d_views = pred_obj_init_sparse_2d[batch_id]
            if batch.get("master_obj_rot6d_label", None) is not None:
                pred_obj_init_rot_deg = torch.rad2deg(
                    self.rotation_geodesic(
                        preds["obj_init_rot6d"][batch_id:batch_id + 1],
                        batch["master_obj_rot6d_label"][batch_id:batch_id + 1],
                    )
                ).unsqueeze(-1).repeat(n_views, 1)
            if batch.get("master_obj_t_label_rel", None) is not None:
                pred_obj_init_trans_mm = torch.linalg.norm(
                    preds["obj_init_trans"][batch_id] - batch["master_obj_t_label_rel"][batch_id].to(device=preds["obj_init_trans"].device, dtype=preds["obj_init_trans"].dtype),
                    dim=-1,
                ).unsqueeze(-1).repeat(n_views, 1) * 1000.0
        pred_kp_master_joints_2d_views = pred_kp_master_joints_2d[batch_id] if pred_kp_master_joints_2d is not None else None
        pred_kp_master_verts_2d_views = pred_kp_master_verts_2d[batch_id] if pred_kp_master_verts_2d is not None else None
        pred_obj_sparse_2d_views = pred_obj_sparse_2d_proj[batch_id]
        pred_sv_obj_sparse_2d_views = pred_sv_obj_sparse_2d[batch_id] if pred_sv_obj_sparse_2d is not None else None
        gt_obj_sparse_2d_views = batch_cam_intr_projection(K_gt, gt_obj_sparse_cam)[batch_id] if gt_obj_sparse_cam is not None else gt_obj_sparse_2d_proj[batch_id]
        gt_joints_2d_sv_views = gt_joints_2d_views
        gt_verts_2d_views = gt_verts_2d_persp[batch_id]
        gt_verts_2d_sv_views = gt_verts_2d_views

        tag_prefix = self._stage_tb_prefix(mode, stage_name)
        self.summary.add_image(
            f"{tag_prefix}/hand_confidence_2d",
            tile_batch_images(
                draw_batch_joint_confidence_images(
                    pred_ref_joints_2d_views,
                    pred_hand_conf_views,
                    img_views,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/sv_hand_joints_2d",
            tile_batch_images(
                draw_batch_joint_images(
                    pred_sv_joints_2d_views,
                    gt_joints_2d_sv_views,
                    img_views,
                    step_idx,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/sv_hand_mesh_2d",
            tile_batch_images(
                    draw_batch_hand_mesh_images_2d(
                    gt_verts2d=gt_verts_2d_sv_views,
                    pred_verts2d=pred_sv_verts_2d_views,
                    face=self.face,
                    tensor_image=img_views,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        if pred_sv_obj_sparse_2d_views is not None:
            self.summary.add_image(
                f"{tag_prefix}/sv_hand_obj_pose_2d",
                tile_batch_images(
                    draw_batch_mesh_images_pred(
                        gt_verts2d=gt_verts_2d_views,
                        pred_verts2d=pred_sv_verts_2d_views,
                        face=self.face,
                        gt_obj2d=gt_obj_sparse_2d_views,
                        pred_obj2d=pred_sv_obj_sparse_2d_views,
                        gt_objc2d=None,
                        pred_objc2d=None,
                        intr=K_gt[batch_id],
                        tensor_image=img_views,
                        pred_obj_rot_error=pred_obj_view_rot_deg,
                        pred_obj_trans_error=pred_obj_view_trans_mm,
                        n_sample=n_views,
                    )
                ),
                step_idx,
                dataformats="HWC",
            )

        if pred_kp_master_joints_2d_views is not None:
            self.summary.add_image(
                f"{tag_prefix}/kp_hand_joints_2d",
                tile_batch_images(
                    draw_batch_joint_images(
                        pred_kp_master_joints_2d_views,
                        gt_joints_2d_views,
                        img_views,
                        step_idx,
                        n_sample=n_views,
                    )
                ),
                step_idx,
                dataformats="HWC",
            )
        if pred_kp_master_verts_2d_views is not None:
            self.summary.add_image(
                f"{tag_prefix}/kp_hand_mesh_2d",
                tile_batch_images(
                    draw_batch_hand_mesh_images_2d(
                        gt_verts2d=gt_verts_2d_views,
                        pred_verts2d=pred_kp_master_verts_2d_views,
                        face=self.face,
                        tensor_image=img_views,
                        n_sample=n_views,
                    )
                ),
                step_idx,
                dataformats="HWC",
            )

        if stage_name == "stage2":
            final_hand_verts_2d_views = pred_kp_master_verts_2d_views if pred_kp_master_verts_2d_views is not None else pred_sv_verts_2d_views
            if pred_obj_init_sparse_2d_views is not None:
                self.summary.add_image(
                    f"{tag_prefix}/init_mesh_obj_2d",
                    tile_batch_images(
                        draw_batch_mesh_images_pred(
                            gt_verts2d=gt_verts_2d_views,
                            pred_verts2d=final_hand_verts_2d_views,
                            face=self.face,
                            gt_obj2d=gt_obj_sparse_2d_views,
                            pred_obj2d=pred_obj_init_sparse_2d_views,
                            gt_objc2d=None,
                            pred_objc2d=None,
                            intr=K_gt[batch_id],
                            tensor_image=img_views,
                            pred_obj_rot_error=pred_obj_init_rot_deg,
                            pred_obj_trans_error=pred_obj_init_trans_mm,
                            n_sample=n_views,
                        )
                    ),
                    step_idx,
                    dataformats="HWC",
                )
            self.summary.add_image(
                f"{tag_prefix}/final_mesh_obj_2d",
                tile_batch_images(
                    draw_batch_mesh_images_pred(
                        gt_verts2d=gt_verts_2d_views,
                        pred_verts2d=final_hand_verts_2d_views,
                        face=self.face,
                        gt_obj2d=gt_obj_sparse_2d_views,
                        pred_obj2d=pred_obj_sparse_2d_views,
                        gt_objc2d=None,
                        pred_objc2d=None,
                        intr=K_gt[batch_id],
                        tensor_image=img_views,
                        pred_obj_rot_error=pred_obj_master_rot_deg,
                        pred_obj_trans_error=pred_obj_master_trans_mm,
                        n_sample=n_views,
                    )
                ),
                step_idx,
                dataformats="HWC",
            )
        self.summary.flush()

    def _forward_impl(self, batch, **kwargs):
        interaction_mode = kwargs.get("interaction_mode", "ho")
        img = batch["image"]  # (B, N, 3, H, W) 5 dimensions
        K = batch['target_cam_intr']  # (B, N, 3, 3)
        T_c2m = torch.linalg.inv(batch['target_cam_extr'])  # (B, N, 4, 4)

        batch_size, num_cams = img.size(0), img.size(1)
        inp_img_shape = img.shape[-2:]  # H, W
        H, W = inp_img_shape
        img_feats, img_feats_reshaped, global_feat = self.extract_img_feat(img)  # [(B, N, C, H, W), ...]\

        # NOTE: @licheng  merging multi-level features in backbone.
        mlvl_feat = self.feat_decode(img_feats)  # (BxN, 128, 32, 32)
        mlvl_feat = mlvl_feat.view(batch_size, num_cams, *mlvl_feat.shape[1:])  # (B, N, 128, 32, 32)
        mlvl_feat = torch.nan_to_num(mlvl_feat, nan=0.0, posinf=1e4, neginf=-1e4)

        uv_hmap = self.heatmap_decode(img_feats)  # (B*N, J_hand + J_obj, 32, 32)
        uv_peak_conf = self._heatmap_peak_confidence(uv_hmap)
        uv_pdf = uv_hmap.reshape(*uv_hmap.shape[:2], -1)
        uv_pdf = uv_pdf / (uv_pdf.sum(dim=-1, keepdim=True) + 1e-6)
        uv_pdf = uv_pdf.contiguous().view(batch_size * num_cams, self.num_hand_joints + self.num_obj_joints, *uv_hmap.shape[-2:])
        uv_coord = integral_heatmap2d(uv_pdf)  # (B*N, J_hand + J_obj, 2), range 0~1
        uv_spread_conf = self._heatmap_spread_confidence(
            uv_pdf,
            uv_coord,
            (H, W),
            tau_px=self.conf_heatmap_spread_tau_px,
        )
        uv_conf_base = torch.sqrt(torch.clamp(uv_peak_conf * uv_spread_conf, min=0.0))
        uv_conf_base = torch.nan_to_num(uv_conf_base, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        uv_conf_base_xy = uv_conf_base.expand(-1, -1, 2).contiguous()
        uv_coord_im = torch.einsum(
            "bij,j->bij", uv_coord, torch.tensor([W, H], dtype=uv_coord.dtype, device=uv_coord.device)
        )
        uv_coord_norm = uv_coord * 2.0 - 1.0
        uv_coord_im = torch.nan_to_num(uv_coord_im, nan=0.0, posinf=max(H, W), neginf=0.0)
        uv_coord_norm = torch.nan_to_num(uv_coord_norm, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

        pred_hand_jts = uv_coord_norm[:, :self.num_hand_joints]
        pred_obj_jts = uv_coord_norm[:, self.num_hand_joints:]
        pred_hand_jts_pixel = uv_coord_im[:, :self.num_hand_joints]
        pred_obj_jts_pixel = uv_coord_im[:, self.num_hand_joints:]
        hand_score = uv_conf_base_xy[:, :self.num_hand_joints]
        obj_score = uv_conf_base_xy[:, self.num_hand_joints:]

        hand_obj_jts = torch.cat((pred_hand_jts, pred_obj_jts), dim=1)  # (B*V, J_hand + J_obj, 2)
        hand_obj_jts = hand_obj_jts.detach()
        hand_obj_jts_feats = F.grid_sample(mlvl_feat.flatten(0, 1), hand_obj_jts.unsqueeze(-2), align_corners=False).squeeze(-1).permute(0, 2, 1)  # (B*V, C, J_hand + J_obj) -> (B*V, J_hand + J_obj, C)
        hand_obj_img_feats = self.hot_pose(mlvl_feat.flatten(0, 1))
        hand_obj_comb_feats = torch.cat((hand_obj_jts_feats, hand_obj_img_feats), dim=2)  # (B*V, J_hand + J_obj, 512)

        hand_obj_att0 = self.att_0(hand_obj_comb_feats)  # (B*V, J_hand + J_obj, 512)
        hand_obj_att1 = self.att_1(hand_obj_att0)  # (B*V, J_hand + J_obj, 512)
        hand_obj_fused = self.mano_fuse(hand_obj_att1)  # (B*V, J_hand + J_obj, 32)
        hand_obj_fused_flat = hand_obj_fused.view(batch_size*num_cams, -1)  # (B*V, (J_hand + J_obj)*32)
        mano_params = self.mano_fc(hand_obj_fused_flat)  # (B*V, 16*6 + 10 + 3)
        pred_hand_pose = mano_params[:, :96]
        pred_shape = mano_params[:, 96:106]
        pred_cam = mano_params[:, 106:]
        pred_cam = torch.cat((F.relu(pred_cam[:, 0:1]), pred_cam[:, 1:]), dim=1).view(batch_size*num_cams, 3)
        coord_xyz_sv, coord_uv_sv, pose_euler_sv, shape_sv, cam_sv = self.mano_decoder(pred_hand_pose, pred_shape, pred_cam)
        coord_xyz_sv = coord_xyz_sv * (self.data_preset_cfg.BBOX_3D_SIZE / 2)

        uv_conf = torch.cat((hand_score, obj_score), dim=1)

        sv_preds = {
            "pred_hand": pred_hand_jts,
            "pred_obj": pred_obj_jts,
            "pred_hand_pixel": pred_hand_jts_pixel,
            "pred_obj_pixel": pred_obj_jts_pixel,
            "pred_uv_heatmap": uv_hmap,
            "pred_uv_conf_base": uv_conf_base_xy,
            "pred_uv_conf": uv_conf,
            "conf_hand": hand_score,
            "mano_3d_mesh_sv": coord_xyz_sv,
            "mano_2d_mesh_sv": coord_uv_sv,
            "mano_pose_euler_sv": pose_euler_sv,
            "mano_shape_sv": shape_sv,
            "mano_cam_sv": cam_sv,
        }

        # 🌟 增加判断逻辑：如果是 'sv' 模式，直接提前返回 sv_preds
        # 假设你传入的参数名叫 mode (需要在函数的 def 定义里加上 mode="mv" 或者 mode="sv")
        if self.mode == 'sv':
            return sv_preds

        # ==========================================
        # 如果不是 'sv'，继续执行后续的多视角/特征增强逻辑
        # ==========================================

        hand_score_ref = hand_score.detach()
        obj_score_ref = obj_score.detach()
        ref_hand, _ = self.project(pred_hand_jts, pred_obj_jts, hand_score_ref, obj_score_ref, K, T_c2m, batch_size, num_cams)
        ref_hand_anchor = ref_hand.detach()
        sv_rot_outputs = None
        if interaction_mode in ["hand", "ho"]:
            sv_rot_outputs = self._predict_singleview_object_pose(
                global_feat=global_feat,
                hand_obj_fused_flat=hand_obj_fused_flat,
                cam_extr=batch["target_cam_extr"],
                master_id=batch["master_id"],
            )

        proj_preds = {
            "ref_hand": ref_hand,
        }
        # prepare image_metas
        img_metas = {
            "inp_img_shape": inp_img_shape,  # h, w
            "cam_intr": batch["target_cam_intr"], 
            "cam_extr": batch["target_cam_extr"],  
            "master_id": batch["master_id"],  # lst (B, )
            # "ref_mesh_gt": gt_mesh,
            "cam_view_num": num_cams
        }

        pt_preds = self.ptEmb_head(
            mlvl_feat=mlvl_feat,
            mano_3d_sv = coord_xyz_sv,
            img_metas=img_metas,
            reference_hand=ref_hand_anchor,
            reference_obj=None,
            obj_template=batch["master_obj_sparse_rest"],
            hand_view_conf=hand_score_ref.mean(dim=-1).view(batch_size, num_cams, self.num_hand_joints),
            sv_rot_outputs=sv_rot_outputs,
            obj_init_rot6d=sv_rot_outputs["obj_init_rot6d"] if sv_rot_outputs is not None else None,
            obj_init_trans=sv_rot_outputs["obj_init_trans"] if sv_rot_outputs is not None else None,
            interaction_mode=interaction_mode,
            stage2_warmup=self.current_stage2_warmup,
            # **extra_kwargs,
        )

        # 🌟 合并三个字典并返回
        # 注意：如果你想把 pt_preds 里的特定内容按照原来 return 的格式改名（比如 "hand_joints_3d"），
        # 我们需要在合并前或者合并时做个映射。这里我按照你原来 return 的结构，把它塞进一个大字典。
        
        final_preds = {}

        # pred_hand_verts_master = pt_preds["pred_hand_verts"].unsqueeze(1).repeat(1, num_cams, 1, 1)  # (B, N, 778, 3)
        # pred_verts_in_cam = batch_cam_extr_transf(T_c2m, pred_hand_verts_master)  # (B, N, 778, 3)
        # pred_obj_sparse_master = pt_preds["all_obj_preds"][-1].unsqueeze(1).repeat(1, num_cams, 1, 1)  # (B, N, 2048, 3)
        # pred_obj_sparse_in_cam = batch_cam_extr_transf(T_c2m, pred_obj_sparse_master)  # (B, N, 2048, 3)
        # enhanced_mlvl_feat = pt_preds["enhanced_img_feat"]  # (B, N, C, H, W)

        # upsampling_input = {
        #         "points": pred_obj_sparse_in_cam.flatten(0, 1),  # (B*N, 2048, 3)
        #         "hand_points": pred_verts_in_cam.flatten(0, 1),  # (B*N, 778, 3)
        #         "mlvl_feat": enhanced_mlvl_feat.flatten(0, 1),  # (B*N, 256, H, W)
        #         "cam_intr": K.flatten(0, 1),  # (B*N, 3, 3)
        #     }

        # up_results = self.pointcloud_upsampler(upsampling_input)
        # pointclouds_up = up_results[-1]

        # 1. 放入额外的特征信息 (如果在你的函数作用域里有的话)
        # 2. 合并 sv_preds 和 proj_preds
        final_preds.update(sv_preds)
        final_preds.update(proj_preds)
        obj_init_rot6d = pt_preds.get("obj_init_rot6d", None)
        obj_fused_rot6d_sv = pt_preds.get("obj_fused_rot6d_sv", None)
        obj_view_rot6d_cam = pt_preds.get("obj_view_rot6d_cam", None)
        obj_view_rot6d_master = pt_preds.get("obj_view_rot6d_master", None)
        obj_view_trans = pt_preds.get("obj_view_trans", None)
        obj_view_trans_master = pt_preds.get("obj_view_trans_master", None)
        obj_init_trans = pt_preds.get("obj_init_trans", None)
        obj_fused_trans_sv = pt_preds.get("obj_fused_trans_sv", None)
        if sv_rot_outputs is not None:
            obj_init_rot6d = sv_rot_outputs.get("obj_init_rot6d", obj_init_rot6d)
            obj_fused_rot6d_sv = sv_rot_outputs.get("obj_fused_rot6d_sv", obj_fused_rot6d_sv)
            obj_view_rot6d_cam = sv_rot_outputs.get("obj_view_rot6d_cam", obj_view_rot6d_cam)
            obj_view_rot6d_master = sv_rot_outputs.get("obj_view_rot6d_master", obj_view_rot6d_master)
            obj_view_trans = sv_rot_outputs.get("obj_view_trans", obj_view_trans)
            obj_view_trans_master = sv_rot_outputs.get("obj_view_trans_master", obj_view_trans_master)
            obj_init_trans = sv_rot_outputs.get("obj_init_trans", obj_init_trans)
            obj_fused_trans_sv = sv_rot_outputs.get("obj_fused_trans_sv", obj_fused_trans_sv)
        
        # 3. 提取并重命名 pt_preds 中的关键信息放入最终字典
        final_preds.update({
            "hand_mesh_xyz_master": pt_preds["all_hand_mesh_xyz_master"],
            "obj_xyz_master": pt_preds["all_obj_xyz_master"],
            "obj_rot6d": pt_preds["all_obj_rot6d"],
            "obj_trans": pt_preds["all_obj_trans"],
            "obj_init_rot6d": obj_init_rot6d,
            "obj_init_trans": obj_init_trans,
            "obj_fused_rot6d_sv": obj_fused_rot6d_sv,
            "obj_fused_trans_sv": obj_fused_trans_sv,
            "obj_view_rot6d_cam": obj_view_rot6d_cam,
            "obj_view_rot6d_master": obj_view_rot6d_master,
            "obj_view_trans": obj_view_trans,
            "obj_view_trans_master": obj_view_trans_master,
            "mano_3d_mesh_master": pt_preds["pred_kp_mano_mesh_xyz_master"],
            "mano_3d_mesh_kp_master": pt_preds["pred_kp_mano_mesh_xyz_master"],
            "mano_3d_mesh_mesh_master": None,
            "pred_pose": pt_preds["pred_kp_mano_params_master"]['pose_euler'],
            "pred_shape": pt_preds["pred_kp_mano_params_master"]['shape'],
            "pred_kp_pose": pt_preds["pred_kp_mano_params_master"]['pose_euler'],
            "pred_kp_shape": pt_preds["pred_kp_mano_params_master"]['shape'],
            "pred_mesh_pose": None,
            "pred_mesh_shape": None,
            "pred_obj_trans_master": pt_preds["pred_obj_trans_master"],
            "interaction_mode": pt_preds.get("interaction_mode", interaction_mode),
            "all_hand_joints_xyz_master": pt_preds.get("all_hand_joints_xyz_master", None),
            # "ball_points": pt_preds["obj_xyz"],
        })

        return final_preds
    
    @staticmethod
    def loss_proj_to_multicam(pred_points, T_c2m, K, gt_joints_2d, n_views, img_scale, visibility=None):
        pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=1e3, neginf=-1e3)
        gt_joints_2d_finite = torch.isfinite(gt_joints_2d).all(dim=-1)
        gt_joints_2d = torch.nan_to_num(gt_joints_2d, nan=0.0, posinf=img_scale, neginf=-img_scale)
        pred_points = pred_points.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)
        pred_points_in_cam = batch_cam_extr_transf(T_c2m, pred_points)
        pred_points_2d = batch_cam_intr_projection(K, pred_points_in_cam)  # (B, N, 21, 2)
        pred_points_2d = torch.nan_to_num(pred_points_2d, nan=0.0, posinf=img_scale, neginf=-img_scale)
        multicam_proj_offset = torch.clamp(pred_points_2d - gt_joints_2d, min=-.5 * img_scale,
                                           max=.5 * img_scale) / img_scale
        loss_2d_points = torch.sum(torch.pow(multicam_proj_offset, 2), dim=3)  # (B, N, 21, 2)
        valid_mask = gt_joints_2d_finite.to(dtype=loss_2d_points.dtype)
        if visibility is not None:
            valid_mask = valid_mask * visibility.to(dtype=loss_2d_points.dtype)
        return (loss_2d_points * valid_mask).sum() / (valid_mask.sum() + 1e-9)

    @staticmethod
    def compute_object_pose_metrics(preds, gt):
        pred_obj_rot6d = torch.nan_to_num(preds["obj_rot6d"][-1].detach(), nan=0.0, posinf=10.0, neginf=-10.0)
        pred_obj_trans = torch.nan_to_num(preds["obj_trans"][-1].detach(), nan=0.0, posinf=10.0, neginf=-10.0)

        gt_obj_rot6d = torch.nan_to_num(gt["master_obj_rot6d_label"].detach(), nan=0.0, posinf=10.0, neginf=-10.0)
        gt_obj_trans = torch.nan_to_num(gt["master_obj_t_label_rel"].detach(), nan=0.0, posinf=10.0, neginf=-10.0)

        pred_rotmat = rot6d_to_rotmat(pred_obj_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_obj_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        rot_angle_deg = torch.rad2deg(torch.acos(cos_theta))

        metric_dict = {
            "metric_obj_rot_l1": torch.mean(torch.abs(pred_obj_rot6d - gt_obj_rot6d)),
            "metric_obj_rot_deg": torch.mean(rot_angle_deg),
            "metric_obj_trans_l1": torch.mean(torch.abs(pred_obj_trans - gt_obj_trans)),
            "metric_obj_trans_epe": torch.mean(torch.norm(pred_obj_trans - gt_obj_trans, dim=-1)),
        }
        obj_points_rest = torch.nan_to_num(
            gt["master_obj_sparse_rest"].detach(),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        pred_obj_points_pose = (
            torch.matmul(rot6d_to_rotmat(pred_obj_rot6d), obj_points_rest.transpose(1, 2)).transpose(1, 2)
            + pred_obj_trans.unsqueeze(1)
        )
        gt_obj_points_pose = (
            torch.matmul(rot6d_to_rotmat(gt_obj_rot6d), obj_points_rest.transpose(1, 2)).transpose(1, 2)
            + gt_obj_trans.unsqueeze(1)
        )
        metric_dict["metric_obj_add"] = torch.mean(torch.norm(pred_obj_points_pose - gt_obj_points_pose, dim=-1).mean(dim=-1))
        metric_dict["metric_obj_adds"] = torch.mean(torch.cdist(pred_obj_points_pose.float(), gt_obj_points_pose.float()).min(dim=-1)[0].mean(dim=-1))
        return metric_dict

    def compute_sv_object_pose_metrics(self, preds, gt):
        pred_obj_view_rot6d_cam = preds.get("obj_view_rot6d_cam", None)
        pred_obj_view_trans = preds.get("obj_view_trans", None)
        target_obj_rot6d_gt = gt.get("target_rot6d_label", None)
        target_obj_trans_gt = gt.get("target_t_label_rel", None)
        if pred_obj_view_rot6d_cam is None or target_obj_rot6d_gt is None or pred_obj_view_trans is None or target_obj_trans_gt is None:
            return {
                "metric_sv_obj_rot_l1": preds["mano_3d_mesh_sv"].new_tensor(0.0),
                "metric_sv_obj_rot_deg": preds["mano_3d_mesh_sv"].new_tensor(0.0),
                "metric_sv_obj_trans_l1": preds["mano_3d_mesh_sv"].new_tensor(0.0),
                "metric_sv_obj_trans_epe": preds["mano_3d_mesh_sv"].new_tensor(0.0),
            }

        pred_obj_view_rot6d_cam = torch.nan_to_num(pred_obj_view_rot6d_cam.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
        pred_obj_view_trans = torch.nan_to_num(pred_obj_view_trans.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
        target_obj_rot6d_gt = torch.nan_to_num(target_obj_rot6d_gt.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
        target_obj_trans_gt = torch.nan_to_num(target_obj_trans_gt.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
        rot_deg = torch.rad2deg(
            self.rotation_geodesic(
                pred_obj_view_rot6d_cam.reshape(-1, 6),
                target_obj_rot6d_gt.reshape(-1, 6),
            )
        ).view(pred_obj_view_rot6d_cam.shape[0], pred_obj_view_rot6d_cam.shape[1])
        rot_l1 = torch.abs(pred_obj_view_rot6d_cam - target_obj_rot6d_gt).mean(dim=-1)
        trans_l1 = torch.abs(pred_obj_view_trans - target_obj_trans_gt).mean(dim=-1)
        trans_epe = torch.linalg.norm(pred_obj_view_trans - target_obj_trans_gt, dim=-1)

        valid_weight = torch.ones_like(rot_deg)
        denom = valid_weight.sum() + 1e-6
        return {
            "metric_sv_obj_rot_l1": (rot_l1 * valid_weight).sum() / denom,
            "metric_sv_obj_rot_deg": (rot_deg * valid_weight).sum() / denom,
            "metric_sv_obj_trans_l1": (trans_l1 * valid_weight).sum() / denom,
            "metric_sv_obj_trans_epe": (trans_epe * valid_weight).sum() / denom,
        }

    @staticmethod
    def rotation_geodesic(pred_rot6d, gt_rot6d):
        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return torch.acos(cos_theta)

    # Object rotation frames used in training:
    # 1) stage1 single-view rotation `obj_view_rot6d_cam` is supervised by
    #    `target_rot6d_label`, which is the per-view object rotation in the
    #    current camera frame after data augmentation.
    # 2) stage2 master rotation `obj_rot6d` is supervised by
    #    `master_obj_rot6d_label`, which is the object rotation in the master
    #    camera frame.
    # 3) stage2 also distills the final master rotation back to each view by
    #    applying the master-to-camera extrinsic rotation, producing a view-frame
    #    target for `obj_view_rot6d_cam`.

    def compute_loss(self, preds, gt, stage_name="stage2", epoch_idx=None, **kwargs):
        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]  # (BN, 799, 3)
        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"]  # (BN, 799, 2)
        pred_mano_mesh_master = preds["mano_3d_mesh_master"]  # (B, 799, 3)
        pred_mano_mesh_kp_master = preds["mano_3d_mesh_kp_master"]  # (B, 799, 3)
        pred_mano_mesh_mesh_master = preds.get("mano_3d_mesh_mesh_master", None)
        pred_hand_mesh_master = preds["hand_mesh_xyz_master"]  # (N_preds, B, 799, 3)
        pred_obj_points_3d = preds["obj_xyz_master"]  # (N_preds, B, 2048, 3)
        pred_obj_rot6d = preds["obj_rot6d"]  # (N_preds, B, 6)
        pred_obj_trans = preds["obj_trans"]  # (N_preds, B, 3)
        pred_obj_init_rot6d = preds.get("obj_init_rot6d", None)
        pred_obj_init_trans = preds.get("obj_init_trans", None)
        pred_obj_view_rot6d_cam = preds.get("obj_view_rot6d_cam", None)
        pred_obj_view_rot6d_master = preds.get("obj_view_rot6d_master", None)
        pred_obj_view_trans = preds.get("obj_view_trans", None)
        pred_pose_sv = preds["mano_pose_euler_sv"]  # (BN, 48)
        pred_shape_sv = preds["mano_shape_sv"]  # (BN, 10)
        pred_pose_kp_master = preds["pred_kp_pose"]  # (B, 48)
        pred_shape_kp_master = preds["pred_kp_shape"]  # (B, 10)
        pred_pose_mesh_master = preds.get("pred_mesh_pose", None)
        pred_shape_mesh_master = preds.get("pred_mesh_shape", None)

        batch_size = gt["image"].size(0)
        n_views = gt["image"].size(1)
        H, W = gt["image"].size(-2), gt["image"].size(-1)
        img_scale = math.sqrt(float(W**2 + H**2))

        # --- GT Data Unpacking ---
        master_hand_joints_gt = gt["master_joints_3d"]  # (B, 21, 3)
        master_obj_sparse_gt = gt["master_obj_sparse"]  # (B, 2048, 3)
        master_obj_rot6d_gt = gt["master_obj_rot6d_label"]  # (B, 6)
        master_obj_trans_gt = gt["master_obj_t_label_rel"]  # (B, 3)
        hand_joints_2d_gt = gt["target_joints_2d"]  # (B, N, 21, 2)
        hand_joints_sv_gt = gt["target_joints_uvd"].flatten(0, 1)  # (B*N, 21, 3)
        gt_T_c2m = torch.linalg.inv(gt["target_cam_extr"])  # (B, N, 4, 4)
        gt_K = gt["target_cam_intr"]  # (B, N, 3, 3)
        target_obj_rot6d_gt = gt.get("target_rot6d_label", None)
        target_obj_trans_gt = gt.get("target_t_label_rel", None)
        gt_hand_joints_rel = gt.get("target_joints_3d_rel", None)
        joints_vis_mv = gt.get("target_joints_vis", None)
        joints_vis = gt.get("target_joints_vis", None)
        joints_vis = joints_vis.flatten(0, 1) if joints_vis is not None else None  # (B*N, 21)
        enable_hand_refine = stage_name in ["stage1", "stage2"]
        enable_object_center_prior = False
        enable_sv_object_rotation = stage_name in ["stage1", "stage2"]
        enable_master_object_rotation = stage_name == "stage2"
        enable_object_refine = stage_name == "stage2"
        enable_sv_self_distill = (
            enable_master_object_rotation
            and self.sv_self_distill_enabled
            and epoch_idx is not None
            and epoch_idx >= self.sv_self_distill_start_epoch
        )
        obj_warmup = self._stage2_object_warmup(epoch_idx, stage_name) if epoch_idx is not None else float(enable_object_refine)
        master_obj_rot_geo_weight = self.obj_view_rot_weight if enable_master_object_rotation else self.obj_pose_rot_weight
        master_obj_rot_l1_weight = self.obj_view_rot6d_weight if enable_master_object_rotation else self.obj_pose_rot6d_weight
        direct_pose_aux_scale = self.obj_direct_pose_aux_scale

        loss_dict = {}

        # ====================================================================
        # 1. Base 2D & Heatmap Loss (早期视角的初步感知)
        # ====================================================================
        pred_jts_2d = pred_mano_2d_mesh_sv[:, self.num_hand_verts:] # (BN, 21, 2)
        gt_jts_2d = hand_joints_sv_gt[..., :2]                      # (BN, 21, 2)
        if gt_hand_joints_rel is not None:
            gt_sv_ortho_cam = self._fit_ortho_camera(
                gt_hand_joints_rel[..., :2],
                gt["target_joints_uvd"][..., :2],
                visibility=joints_vis_mv,
            )
            gt_jts_2d = self._ortho_project_points(gt_hand_joints_rel, gt_sv_ortho_cam).flatten(0, 1)

        # 1. 计算所有点的绝对误差 (L1)
        diff_2d = torch.abs(pred_jts_2d - gt_jts_2d) # (BN, 21, 2)
        
        if joints_vis is not None:
            # joints_vis: (BN, 21) -> (BN, 21, 1)
            vis_mask = joints_vis.unsqueeze(-1)
            
            # 2. 将不可见点的误差清零
            diff_2d = diff_2d * vis_mask
            
            # 3. 统计真正可见的“坐标通道”数量
            # vis_mask.expand_as(diff_2d) 使得 1 个可见点等于 2 个有效坐标 (x 和 y)
            valid_count = vis_mask.expand_as(diff_2d).sum()
            
            # 4. 手动求均值：真实误差总和 / 真实可见坐标数
            loss_hand_2d_sv = diff_2d.sum() / (valid_count + 1e-9)
        else:
            # 如果没有 Mask，就走正常的求均值
            loss_hand_2d_sv = diff_2d.mean()
        pred_hand_pixel = preds["pred_hand_pixel"].view(batch_size, n_views, self.num_hand_joints, 2)
        gt_hand_pixel = gt["target_joints_2d"]
        gt_hand_pixel_finite = torch.isfinite(gt_hand_pixel).all(dim=-1)
        gt_hand_pixel = torch.nan_to_num(gt_hand_pixel, nan=0.0, posinf=img_scale, neginf=-img_scale)
        hand_heatmap_diff = (pred_hand_pixel - gt_hand_pixel) / img_scale
        hand_heatmap_diff = torch.sum(torch.pow(hand_heatmap_diff, 2), dim=3)
        if joints_vis is not None:
            hand_valid_mask = joints_vis.view(batch_size, n_views, self.num_hand_joints) * gt_hand_pixel_finite.to(dtype=hand_heatmap_diff.dtype)
            hand_heatmap_diff = hand_heatmap_diff * hand_valid_mask
            hand_valid = hand_valid_mask.sum()
            loss_heatmap_hand = hand_heatmap_diff.sum() / (hand_valid + 1e-9)
        else:
            hand_valid_mask = gt_hand_pixel_finite.to(dtype=hand_heatmap_diff.dtype)
            loss_heatmap_hand = (hand_heatmap_diff * hand_valid_mask).sum() / (hand_valid_mask.sum() + 1e-9)

        pred_uv_heatmap = preds["pred_uv_heatmap"]
        pred_hand_heatmap = pred_uv_heatmap[:, :self.num_hand_joints]
        gt_hand_heatmap = gt.get("target_joints_heatmap", None)
        if gt_hand_heatmap is not None:
            gt_hand_heatmap = gt_hand_heatmap.flatten(0, 1).to(dtype=pred_hand_heatmap.dtype, device=pred_hand_heatmap.device)
            loss_heatmap_hand_map = self._masked_heatmap_mse(pred_hand_heatmap, gt_hand_heatmap, joints_vis)
        else:
            loss_heatmap_hand_map = pred_jts_2d.new_tensor(0.0)

        loss_heatmap_hand = loss_heatmap_hand * self.heatmap_hand_weight
        loss_heatmap_hand_map = loss_heatmap_hand_map * self.heatmap_hand_map_weight

        loss_dict.update({
            'loss_2d_sv': loss_hand_2d_sv,
            'loss_heatmap_hand': loss_heatmap_hand,
            'loss_heatmap_hand_map': loss_heatmap_hand_map,
        })
        obj_prior_warmup = obj_warmup if stage_name == "stage2" else 1.0

        # ====================================================================
        # 2. Triangulation Loss (多视角几何收敛先验)
        # ====================================================================
        loss_triang_hand = self.criterion_joints(preds["ref_hand"], master_hand_joints_gt) * self.triangulation_hand_weight
        loss_triang_total = loss_triang_hand
        
        loss_dict['loss_triang_hand'] = loss_triang_hand
        loss_dict['loss_triang'] = loss_triang_total

        if enable_master_object_rotation and pred_obj_init_rot6d is not None:
            pred_obj_init_rot6d = torch.nan_to_num(pred_obj_init_rot6d, nan=0.0, posinf=10.0, neginf=-10.0)
            master_obj_rot6d_gt = torch.nan_to_num(master_obj_rot6d_gt, nan=0.0, posinf=10.0, neginf=-10.0)
            loss_obj_init_rot_geo = torch.mean(self.rotation_geodesic(pred_obj_init_rot6d, master_obj_rot6d_gt)) * self.obj_init_rot_weight * obj_prior_warmup * direct_pose_aux_scale
            loss_obj_init_rot_l1 = self.coord_loss(pred_obj_init_rot6d, master_obj_rot6d_gt) * self.obj_init_rot6d_weight * obj_prior_warmup * direct_pose_aux_scale
        else:
            loss_obj_init_rot_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_init_rot_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_init_rot = loss_obj_init_rot_geo + loss_obj_init_rot_l1
        if enable_master_object_rotation and pred_obj_init_trans is not None:
            pred_obj_init_trans = torch.nan_to_num(pred_obj_init_trans, nan=0.0, posinf=10.0, neginf=-10.0)
            master_obj_trans_gt = torch.nan_to_num(master_obj_trans_gt, nan=0.0, posinf=10.0, neginf=-10.0)
            loss_obj_init_trans = self.coord_loss(pred_obj_init_trans, master_obj_trans_gt) * self.obj_init_trans_weight * obj_prior_warmup * direct_pose_aux_scale
        else:
            loss_obj_init_trans = pred_mano_3d_mesh_sv.new_tensor(0.0)

        if enable_sv_object_rotation and pred_obj_view_rot6d_cam is not None:
            pred_obj_view_rot6d_cam = torch.nan_to_num(pred_obj_view_rot6d_cam, nan=0.0, posinf=10.0, neginf=-10.0)
            pred_obj_view_trans = torch.nan_to_num(pred_obj_view_trans, nan=0.0, posinf=10.0, neginf=-10.0) if pred_obj_view_trans is not None else None
            valid_view_weight = torch.ones(batch_size, n_views, dtype=pred_obj_view_rot6d_cam.dtype, device=pred_obj_view_rot6d_cam.device)

            if target_obj_rot6d_gt is not None:
                target_obj_rot6d_gt = torch.nan_to_num(target_obj_rot6d_gt, nan=0.0, posinf=10.0, neginf=-10.0)
                view_rot_geo_gt = self.rotation_geodesic(
                    pred_obj_view_rot6d_cam.reshape(-1, 6),
                    target_obj_rot6d_gt.reshape(-1, 6),
                ).view(batch_size, n_views)
                view_rot_l1_gt = torch.abs(pred_obj_view_rot6d_cam - target_obj_rot6d_gt).mean(dim=-1)
                loss_obj_view_rot_geo = (
                    (view_rot_geo_gt * valid_view_weight).sum() / (valid_view_weight.sum() + 1e-6)
                ) * self.obj_view_rot_weight * obj_prior_warmup * direct_pose_aux_scale
                loss_obj_view_rot_l1 = (
                    (view_rot_l1_gt * valid_view_weight).sum() / (valid_view_weight.sum() + 1e-6)
                ) * self.obj_view_rot6d_weight * obj_prior_warmup * direct_pose_aux_scale
            else:
                loss_obj_view_rot_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
                loss_obj_view_rot_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
            if pred_obj_view_trans is not None and target_obj_trans_gt is not None:
                target_obj_trans_gt = torch.nan_to_num(target_obj_trans_gt, nan=0.0, posinf=10.0, neginf=-10.0)
                view_trans_l1 = torch.abs(pred_obj_view_trans - target_obj_trans_gt).mean(dim=-1)
                loss_obj_view_trans = (
                    (view_trans_l1 * valid_view_weight).sum() / (valid_view_weight.sum() + 1e-6)
                ) * self.obj_view_trans_weight * obj_prior_warmup * direct_pose_aux_scale
            else:
                loss_obj_view_trans = pred_mano_3d_mesh_sv.new_tensor(0.0)
            if pred_obj_view_trans is not None and gt.get("target_obj_pc_sparse", None) is not None:
                pred_sv_obj_points = self._build_object_points_from_hand_pose(
                    gt["master_obj_sparse_rest"],
                    pred_obj_view_rot6d_cam,
                    gt["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1],
                    pred_obj_view_trans,
                )
                loss_obj_view_points = self.coord_loss(pred_sv_obj_points, gt["target_obj_pc_sparse"]) * self.obj_pose_points_weight * obj_prior_warmup
            else:
                loss_obj_view_points = pred_mano_3d_mesh_sv.new_tensor(0.0)

            if enable_sv_self_distill:
                final_master_rot6d = torch.nan_to_num(pred_obj_rot6d[-1].detach(), nan=0.0, posinf=10.0, neginf=-10.0)
                final_master_rotmat = rot6d_to_rotmat(final_master_rot6d)
                master_to_cam_rot = gt_T_c2m[..., :3, :3].to(dtype=final_master_rotmat.dtype, device=final_master_rotmat.device)
                distilled_view_rotmat = torch.matmul(master_to_cam_rot, final_master_rotmat.unsqueeze(1))
                distilled_view_rot6d = rotmat_to_rot6d(distilled_view_rotmat.reshape(-1, 3, 3)).view(batch_size, n_views, 6)
                distilled_view_rot6d = torch.nan_to_num(distilled_view_rot6d, nan=0.0, posinf=10.0, neginf=-10.0)
                view_rot_geo_self = self.rotation_geodesic(
                    pred_obj_view_rot6d_cam.reshape(-1, 6),
                    distilled_view_rot6d.reshape(-1, 6),
                ).view(batch_size, n_views)
                view_rot_l1_self = torch.abs(pred_obj_view_rot6d_cam - distilled_view_rot6d).mean(dim=-1)
                loss_obj_view_rot_self_geo = (
                    (view_rot_geo_self * valid_view_weight).sum() / (valid_view_weight.sum() + 1e-6)
                ) * self.obj_view_rot_weight * obj_prior_warmup * direct_pose_aux_scale
                loss_obj_view_rot_self_l1 = (
                    (view_rot_l1_self * valid_view_weight).sum() / (valid_view_weight.sum() + 1e-6)
                ) * self.obj_view_rot6d_weight * obj_prior_warmup * direct_pose_aux_scale
            else:
                loss_obj_view_rot_self_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
                loss_obj_view_rot_self_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
        else:
            loss_obj_view_rot_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_view_rot_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_view_trans = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_view_points = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_view_rot_self_geo = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_obj_view_rot_self_l1 = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_obj_view_rot = (
            loss_obj_view_rot_geo
            + loss_obj_view_rot_l1
            + loss_obj_view_trans
            + loss_obj_view_points
            + loss_obj_view_rot_self_geo
            + loss_obj_view_rot_self_l1
        )

        loss_dict['loss_obj_init_trans'] = loss_obj_init_trans
        loss_dict['loss_obj_init_rot'] = loss_obj_init_rot
        loss_dict['loss_obj_init_rot_geo'] = loss_obj_init_rot_geo
        loss_dict['loss_obj_init_rot_l1'] = loss_obj_init_rot_l1
        loss_dict['loss_obj_view_rot'] = loss_obj_view_rot
        loss_dict['loss_obj_view_rot_geo'] = loss_obj_view_rot_geo
        loss_dict['loss_obj_view_rot_l1'] = loss_obj_view_rot_l1
        loss_dict['loss_obj_view_trans'] = loss_obj_view_trans
        loss_dict['loss_obj_view_points'] = loss_obj_view_points
        loss_dict['loss_obj_view_rot_self_geo'] = loss_obj_view_rot_self_geo
        loss_dict['loss_obj_view_rot_self_l1'] = loss_obj_view_rot_self_l1

        # ====================================================================
        # 3. MANO Prior & Consistency Constraints (人体工学与视角一致性)
        # ====================================================================
        if enable_hand_refine:
            pred_mano_3d_mesh_kp_mv = pred_mano_mesh_kp_master.unsqueeze(1).repeat(1, n_views, 1, 1)
            pred_mano_3d_mesh_kp_mv = batch_cam_extr_transf(gt_T_c2m, pred_mano_3d_mesh_kp_mv).flatten(0, 1).detach()
            pred_mano_center_kp_mv = pred_mano_3d_mesh_kp_mv[:, 778 + self.center_idx, :].unsqueeze(1)
            pred_mano_3d_mesh_kp_mv = pred_mano_3d_mesh_kp_mv - pred_mano_center_kp_mv
            loss_mano_consistency_kp = self.coord_loss(pred_mano_3d_mesh_sv, pred_mano_3d_mesh_kp_mv) * self.mano_consistency_weight

            loss_mano_consistency_mesh = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_mano_consistency = loss_mano_consistency_kp
        else:
            loss_mano_consistency_kp = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_mano_consistency_mesh = pred_mano_3d_mesh_sv.new_tensor(0.0)
            loss_mano_consistency = pred_mano_3d_mesh_sv.new_tensor(0.0)

        zero_target_sv = torch.zeros_like(pred_pose_sv[:, 3:])
        zero_target_kp_master = torch.zeros_like(pred_pose_kp_master[:, 3:])
        if enable_hand_refine:
            loss_pose_reg_sv = self.coord_loss(pred_pose_sv[:, 3:], zero_target_sv) * self.cfg.LOSS.POSE_N
            loss_pose_reg_kp_master = self.coord_loss(pred_pose_kp_master[:, 3:], zero_target_kp_master) * self.cfg.LOSS.POSE_N
            loss_shape_reg_sv = self.coord_loss(pred_shape_sv, torch.zeros_like(pred_shape_sv)) * self.cfg.LOSS.SHAPE_N
            loss_shape_reg_kp_master = self.coord_loss(pred_shape_kp_master, torch.zeros_like(pred_shape_kp_master)) * self.cfg.LOSS.SHAPE_N
            loss_pose_reg_mesh_master = pred_pose_sv.new_tensor(0.0)
            loss_shape_reg_mesh_master = pred_shape_sv.new_tensor(0.0)
            loss_pose_reg_master = loss_pose_reg_kp_master
            loss_shape_reg_master = loss_shape_reg_kp_master
            loss_pose_reg = loss_pose_reg_sv + loss_pose_reg_master
            loss_shape_reg = loss_shape_reg_sv + loss_shape_reg_master
        else:
            loss_pose_reg_sv = pred_pose_sv.new_tensor(0.0)
            loss_pose_reg_master = pred_pose_sv.new_tensor(0.0)
            loss_pose_reg_kp_master = pred_pose_sv.new_tensor(0.0)
            loss_pose_reg_mesh_master = pred_pose_sv.new_tensor(0.0)
            loss_shape_reg_sv = pred_shape_sv.new_tensor(0.0)
            loss_shape_reg_master = pred_shape_sv.new_tensor(0.0)
            loss_shape_reg_kp_master = pred_shape_sv.new_tensor(0.0)
            loss_shape_reg_mesh_master = pred_shape_sv.new_tensor(0.0)
            loss_pose_reg = pred_pose_sv.new_tensor(0.0)
            loss_shape_reg = pred_shape_sv.new_tensor(0.0)

        loss_dict.update({
            'loss_mano_consist': loss_mano_consistency,
            'loss_mano_consist_kp': loss_mano_consistency_kp,
            'loss_mano_consist_mesh': loss_mano_consistency_mesh,
            'loss_pose_reg_sv': loss_pose_reg_sv,
            'loss_pose_reg_master': loss_pose_reg_master,
            'loss_pose_reg_kp_master': loss_pose_reg_kp_master,
            'loss_pose_reg_mesh_master': loss_pose_reg_mesh_master,
            'loss_pose_reg': loss_pose_reg,
            'loss_shape_reg_sv': loss_shape_reg_sv,
            'loss_shape_reg_master': loss_shape_reg_master,
            'loss_shape_reg_kp_master': loss_shape_reg_kp_master,
            'loss_shape_reg_mesh_master': loss_shape_reg_mesh_master,
            'loss_shape_reg': loss_shape_reg
        })

        # ====================================================================
        # 4. Deep Interaction Refinement (Point Transformer 的逐层 3D 监督)
        # ====================================================================
        loss_recon_total = pred_mano_3d_mesh_sv.new_tensor(0.0)
        # 假设你有多层预测，我们追踪最后一层的特定 loss 供记录
        final_chamfer = pred_mano_3d_mesh_sv.new_tensor(0.0)
        final_emd = pred_mano_3d_mesh_sv.new_tensor(0.0)
        final_penetration = pred_mano_3d_mesh_sv.new_tensor(0.0)
        final_obj_pose = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_mano_proj = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_mano_proj_kp = pred_mano_3d_mesh_sv.new_tensor(0.0)
        loss_mano_proj_mesh = pred_mano_3d_mesh_sv.new_tensor(0.0)

        for i in range(pred_hand_mesh_master.shape[0]):
            pred_hand_mesh_master_i = pred_hand_mesh_master[i]  # (B, 799, 3)
            if not torch.isfinite(pred_hand_mesh_master_i).all():
                logger.warning(
                    f"[NonFiniteDecoderHandState] stage={stage_name} layer={i} "
                    f"max_abs={float(torch.nan_to_num(pred_hand_mesh_master_i.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()):.4f}"
                )
            pred_hand_mesh_master_i = torch.nan_to_num(
                pred_hand_mesh_master_i,
                nan=0.0,
                posinf=self.data_preset_cfg.BBOX_3D_SIZE * 4.0,
                neginf=-self.data_preset_cfg.BBOX_3D_SIZE * 4.0,
            )
            pred_hand_joints_master_i = pred_hand_mesh_master_i[:, self.num_hand_verts:]  # (B, 21, 3)
            pred_obj_points_i = pred_obj_points_3d[i]  # (B, 2048, 3)
            pred_obj_rot6d_i = pred_obj_rot6d[i]  # (B, 6)
            pred_obj_trans_i = pred_obj_trans[i]  # (B, 3)

            if enable_hand_refine:
                loss_3d_joints = self.criterion_joints(pred_hand_joints_master_i, master_hand_joints_gt) * self.decoder_hand_weight
                loss_2d_joints = self.loss_proj_to_multicam(
                    pred_hand_joints_master_i, gt_T_c2m, gt_K, hand_joints_2d_gt, n_views, img_scale, visibility=joints_vis_mv
                ) * self.decoder_proj_weight
            else:
                zero = pred_hand_mesh_master_i.new_tensor(0.0)
                loss_3d_joints = zero
                loss_2d_joints = zero

            if enable_object_refine:
                pred_obj_rot6d_i = torch.nan_to_num(pred_obj_rot6d_i, nan=0.0, posinf=10.0, neginf=-10.0)
                pred_obj_trans_i = torch.nan_to_num(pred_obj_trans_i, nan=0.0, posinf=10.0, neginf=-10.0)
                pred_obj_points_i = torch.nan_to_num(pred_obj_points_i, nan=0.0, posinf=self.data_preset_cfg.BBOX_3D_SIZE * 8.0, neginf=-self.data_preset_cfg.BBOX_3D_SIZE * 8.0)
                master_obj_rot6d_gt = torch.nan_to_num(master_obj_rot6d_gt, nan=0.0, posinf=10.0, neginf=-10.0)
                master_obj_trans_gt = torch.nan_to_num(master_obj_trans_gt, nan=0.0, posinf=10.0, neginf=-10.0)
                master_obj_sparse_gt = torch.nan_to_num(master_obj_sparse_gt, nan=0.0, posinf=self.data_preset_cfg.BBOX_3D_SIZE * 8.0, neginf=-self.data_preset_cfg.BBOX_3D_SIZE * 8.0)
                loss_obj_rot_geo = torch.mean(self.rotation_geodesic(pred_obj_rot6d_i, master_obj_rot6d_gt)) * master_obj_rot_geo_weight * obj_warmup * direct_pose_aux_scale
                loss_obj_rot_l1 = self.coord_loss(pred_obj_rot6d_i, master_obj_rot6d_gt) * master_obj_rot_l1_weight * obj_warmup * direct_pose_aux_scale
                loss_obj_rot = loss_obj_rot_geo + loss_obj_rot_l1
                loss_obj_trans = self.coord_loss(pred_obj_trans_i, master_obj_trans_gt) * self.obj_pose_trans_weight * obj_warmup * direct_pose_aux_scale
                loss_obj_points = self.coord_loss(pred_obj_points_i, master_obj_sparse_gt) * self.obj_pose_points_weight * obj_warmup
                loss_obj_pose = loss_obj_rot + loss_obj_trans + loss_obj_points

                loss_chamfer_raw, _ = self.chamfer_loss(pred_obj_points_i.float(), master_obj_sparse_gt.float())
                loss_chamfer = loss_chamfer_raw * self.obj_chamfer_weight * obj_warmup
                loss_emd_raw = torch.mean(self.earth_mover_loss(pred_obj_points_i.float(), master_obj_sparse_gt.float(), transpose=False) / pred_obj_points_i.shape[1])
                loss_emd = loss_emd_raw * self.obj_emd_weight * obj_warmup

                obj_to_hand_dist = torch.cdist(pred_obj_points_i.float(), pred_hand_mesh_master_i.float())  # [B, 2048, 778]
                min_dist, _ = obj_to_hand_dist.min(dim=-1)
                loss_penetration = torch.mean(F.relu(0.002 - min_dist)) * self.obj_penetration_weight * obj_warmup
            else:
                zero = pred_hand_mesh_master_i.new_tensor(0.0)
                loss_obj_rot_geo = zero
                loss_obj_rot_l1 = zero
                loss_obj_rot = zero
                loss_obj_trans = zero
                loss_obj_points = zero
                loss_obj_pose = zero
                loss_chamfer = zero
                loss_emd = zero
                loss_penetration = zero

            # Accumulate layer-wise reconstruction loss
            layer_recon_loss = loss_3d_joints + loss_2d_joints + loss_obj_pose + loss_chamfer + loss_emd + loss_penetration
            loss_recon_total += layer_recon_loss
            loss_dict[f'dec{i}_recon'] = layer_recon_loss

            if i == pred_hand_mesh_master.shape[0] - 1:
                # Hand Mano Projection Loss (最后一层的 MANO 投影监督)
                pred_mano_joints_kp_master = pred_mano_mesh_kp_master[:, self.num_hand_verts:]  # (B, 21, 3)
                if enable_hand_refine:
                    loss_mano_proj_kp = self.criterion_joints(pred_mano_joints_kp_master, master_hand_joints_gt) * self.mano_proj_weight
                    loss_mano_proj_mesh = pred_mano_joints_kp_master.new_tensor(0.0)
                    loss_mano_proj = loss_mano_proj_kp
                else:
                    loss_mano_proj_kp = pred_mano_joints_kp_master.new_tensor(0.0)
                    loss_mano_proj_mesh = pred_mano_joints_kp_master.new_tensor(0.0)
                    loss_mano_proj = pred_mano_joints_kp_master.new_tensor(0.0)
                loss_dict['loss_mano_proj'] = loss_mano_proj
                loss_dict['loss_mano_proj_kp'] = loss_mano_proj_kp
                loss_dict['loss_mano_proj_mesh'] = loss_mano_proj_mesh
                # 记录最后一层的细分指标供展示
                loss_dict['loss_3d_jts'] = loss_3d_joints
                loss_dict['loss_2d_proj'] = loss_2d_joints
                final_chamfer = loss_chamfer
                final_emd = loss_emd
                final_penetration = loss_penetration
                loss_dict['loss_obj_rot'] = loss_obj_rot
                loss_dict['loss_obj_rot_geo'] = loss_obj_rot_geo
                loss_dict['loss_obj_rot_l1_aux'] = loss_obj_rot_l1
                loss_dict['loss_obj_trans'] = loss_obj_trans
                loss_dict['loss_obj_points'] = loss_obj_points
                final_obj_pose = loss_obj_pose

        loss_dict.update({
            'loss_obj_pose': final_obj_pose,
            'loss_obj_chamfer': final_chamfer,
            'loss_obj_emd': final_emd,
            'loss_penetration': final_penetration,
            'loss_obj_warmup': pred_mano_3d_mesh_sv.new_tensor(float(obj_warmup)),
        })

        loss_dict["loss_recon"] = loss_recon_total
        total_loss = (
            loss_hand_2d_sv
            + loss_heatmap_hand
            + loss_heatmap_hand_map
            + loss_triang_total
            + loss_obj_init_rot
            + loss_obj_init_trans
            + loss_obj_view_rot
            + loss_mano_consistency
            + loss_pose_reg
            + loss_shape_reg
            + loss_mano_proj
            + loss_recon_total
        )
        loss_dict['loss'] = total_loss

        return total_loss, loss_dict

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        """
        [ModuleAbstract 要求] 路由入口
        """
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "draw":
            return self.draw_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def training_step(self, batch, step_idx, **kwargs):
        img = batch["image"]  # (B, N, 3, H, W) 5 channels
        batch_size = img.size(0)
        n_views = img.size(1)
        
        # 提取预测结果
        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = self._stage2_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._maybe_log_stage_transition("train", epoch_idx, stage_name, preds, batch)
        
        # GT 数据
        joints_2d_gt = batch["target_joints_uvd"][:, :, :, :2]  # (B, N, 21, 2)
        verts_2d_gt = batch["target_verts_uvd"][:, :, :, :2]  # (B, N, 778, 2)
        joints_3d_master_gt = batch["master_joints_3d"]  # (B, 21, 3)
        verts_3d_master_gt = batch["master_verts_3d"]    # (B, 778, 3)
        obj_sparse_3d_master_gt = batch["master_obj_sparse"]  # (B, 2048, 3)
        verts_3d_gt = batch["target_verts_3d"]  # (B, N, 778, 3)
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)  # (B*N, 21, 3)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)    # (B*N, 778, 3)
        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])
        
        # 为了兼容性，使用 .get() 
        obj_pc_sparse = batch.get('target_obj_pc_sparse', None)
        
        # 计算损失
        loss, loss_dict = self.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        pose_metric_dict = self.compute_object_pose_metrics(preds, batch) if stage_name == "stage2" else self._zero_metric_dict(img.device)
        sv_obj_metric_dict = self.compute_sv_object_pose_metrics(preds, batch) if stage_name in ["stage1", "stage2"] else {
            "metric_sv_obj_rot_l1": img.new_tensor(0.0),
            "metric_sv_obj_rot_deg": img.new_tensor(0.0),
            "metric_sv_obj_trans_l1": img.new_tensor(0.0),
            "metric_sv_obj_trans_epe": img.new_tensor(0.0),
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
        
        # ==============================================================
        # 🌟 修复变量不对齐：从 hand_mesh_xyz 中切片分离 V 和 J
        # hand_mesh_xyz shape: (N_preds, B, 799, 3)
        # ==============================================================
        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]  # (BN, 799, 3)
        pred_mano_3d_joints_sv = pred_mano_3d_mesh_sv[:, self.num_hand_verts:]  # (BN, 21, 3)
        pred_mano_3d_verts_sv = pred_mano_3d_mesh_sv[:, :self.num_hand_verts]  # (BN, 778, 3)
        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"]  # (BN, 799, 2)
        pred_mano_2d_joints_sv = pred_mano_2d_mesh_sv[:, self.num_hand_verts:]  # (BN, 21, 2)
        pred_mano_2d_verts_sv = pred_mano_2d_mesh_sv[:, :self.num_hand_verts]  # (BN, 778, 2)
        pred_mano_3d_mesh_master = preds["mano_3d_mesh_master"]  # (B, 799, 3)
        pred_mano_3d_joints_master = pred_mano_3d_mesh_master[:, self.num_hand_verts:]  # (B, 21, 3)
        pred_mano_3d_verts_master = pred_mano_3d_mesh_master[:, :self.num_hand_verts]  # (B, 778, 3)
        pred_mano_3d_mesh_kp_master = preds.get("mano_3d_mesh_kp_master", None)
        pred_mano_3d_mesh_mesh_master = preds.get("mano_3d_mesh_mesh_master", None)
        # pred_hand_verts_3d = preds["hand_mesh_xyz"][:, :self.num_hand_verts]  # (B, 778, 3)
        # pred_hand_joints_3d = preds["hand_mesh_xyz"][:, self.num_hand_verts:] # (B, 21, 3)
        
        pred_ref_hand_joints_3d = preds["ref_hand"]  # (B, 21, 3)
        pred_obj_sparse_3d = preds["obj_xyz_master"][-1]    # (B, 2048, 3)
        
        self.MPJPE_SV_3D.feed(pred_mano_3d_joints_sv, gt_kp=joints_3d_rel_gt)
        self.MPVPE_SV_3D.feed(pred_mano_3d_verts_sv, gt_kp=verts_3d_rel_gt)
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
            self.summary.add_scalar("MPJPE_SV_3D", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D", self.MPVPE_SV_3D.get_result(), step_idx)
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
    
    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([
            self.MPJPE_SV_3D, self.MPVPE_SV_3D,
            self.MPJPE_MASTER_3D, self.MPVPE_MASTER_3D,
            self.MPJPE_KP_MASTER_3D, self.MPVPE_KP_MASTER_3D,
            self.MPJPE_MESH_MASTER_3D, self.MPVPE_MESH_MASTER_3D,
            self.MPRPE_HAND_3D,
            self.OBJ_RECON_SV, self.OBJ_RECON_MASTER,
        ], epoch_idx, comment=comment)
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.MPJPE_MASTER_3D.reset()
        self.MPVPE_MASTER_3D.reset()
        self.MPJPE_KP_MASTER_3D.reset()
        self.MPVPE_KP_MASTER_3D.reset()
        self.MPJPE_MESH_MASTER_3D.reset()
        self.MPVPE_MESH_MASTER_3D.reset()
        self.MPRPE_HAND_3D.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_RECON_MASTER.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        img = batch["image"]
        batch_size = img.size(0)
        n_views = img.size(1)

        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = self._stage2_object_warmup(epoch_idx, stage_name)
        use_full_recon_eval = self._use_full_val_recon(epoch_idx)
        self.current_val_recon_mode = "full" if use_full_recon_eval else "fast"
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._maybe_log_stage_transition("val", epoch_idx, stage_name, preds, batch)
        pose_metric_dict = self.compute_object_pose_metrics(preds, batch) if stage_name == "stage2" else self._zero_metric_dict(img.device)
        sv_obj_metric_dict = self.compute_sv_object_pose_metrics(preds, batch) if stage_name in ["stage1", "stage2"] else {
            "metric_sv_obj_rot_l1": img.new_tensor(0.0),
            "metric_sv_obj_rot_deg": img.new_tensor(0.0),
            "metric_sv_obj_trans_l1": img.new_tensor(0.0),
            "metric_sv_obj_trans_epe": img.new_tensor(0.0),
        }
        
        # GT 数据
        joints_2d_gt = batch["target_joints_uvd"][:, :, :, :2]  # (B, N, 21, 2)
        verts_2d_gt = batch["target_verts_uvd"][:, :, :, :2]  # (B, N, 778, 2)
        joints_3d_master_gt = batch["master_joints_3d"]  # (B, 21, 3)
        verts_3d_master_gt = batch["master_verts_3d"]    # (B, 778, 3)
        obj_sparse_3d_master_gt = batch["master_obj_sparse"]  # (B, 2048, 3)
        joints_3d_gt = batch["target_joints_3d"].flatten(0, 1)  # (B*N, 21, 3)
        verts_3d_gt = batch["target_verts_3d"].flatten(0, 1)  # (B*N, 778, 3)
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)  # (B*N, 21, 3)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)    # (B*N, 778, 3)
        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])
        
        # 为了兼容性，使用 .get() 
        obj_pc_sparse = batch.get('target_obj_pc_sparse', None)
        
        # ==============================================================
        # 🌟 修复变量不对齐：从 hand_mesh_xyz 中切片分离 V 和 J
        # hand_mesh_xyz shape: (N_preds, B, 799, 3)
        # ==============================================================
        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]  # (BN, 799, 3)
        pred_mano_3d_joints_sv = pred_mano_3d_mesh_sv[:, self.num_hand_verts:]  # (BN, 21, 3)
        pred_mano_3d_verts_sv = pred_mano_3d_mesh_sv[:, :self.num_hand_verts]  # (BN, 778, 3)
        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"]  # (BN, 799, 2)
        pred_mano_2d_joints_sv = pred_mano_2d_mesh_sv[:, self.num_hand_verts:]  # (BN, 21, 2)
        pred_mano_2d_verts_sv = pred_mano_2d_mesh_sv[:, :self.num_hand_verts]  # (BN, 778, 2)
        pred_mano_3d_mesh_master = preds["mano_3d_mesh_master"]  # (B, 799, 3)
        pred_mano_3d_joints_master = pred_mano_3d_mesh_master[:, self.num_hand_verts:]  # (B, 21, 3)
        pred_mano_3d_verts_master = pred_mano_3d_mesh_master[:, :self.num_hand_verts]  # (B, 778, 3)
        pred_mano_3d_mesh_kp_master = preds.get("mano_3d_mesh_kp_master", None)
        pred_mano_3d_mesh_mesh_master = preds.get("mano_3d_mesh_mesh_master", None)
        # pred_hand_verts_3d = preds["hand_mesh_xyz"][:, :self.num_hand_verts]  # (B, 778, 3)
        # pred_hand_joints_3d = preds["hand_mesh_xyz"][:, self.num_hand_verts:] # (B, 21, 3)
        
        pred_ref_hand_joints_3d = preds["ref_hand"]  # (B, 21, 3)
        pred_obj_sparse_3d = preds["obj_xyz_master"][-1]    # (B, 2048, 3)

        pred_mano_3d_joints_master_in_cam = batch_cam_extr_transf(T_c2m_gt, pred_mano_3d_joints_master.unsqueeze(1).repeat(1, n_views, 1, 1)).flatten(0, 1)  # (B, N, 21, 3)
        pred_mano_3d_verts_master_in_cam = batch_cam_extr_transf(T_c2m_gt, pred_mano_3d_verts_master.unsqueeze(1).repeat(1, n_views, 1, 1)).flatten(0, 1)  # (B, N, 778, 3)
        
        self.MPJPE_SV_3D.feed(pred_mano_3d_joints_sv, gt_kp=joints_3d_rel_gt)
        self.MPVPE_SV_3D.feed(pred_mano_3d_verts_sv, gt_kp=verts_3d_rel_gt)
        self.MPJPE_MASTER_3D.feed(pred_mano_3d_joints_master, gt_kp=joints_3d_master_gt)
        self.MPVPE_MASTER_3D.feed(pred_mano_3d_verts_master, gt_kp=verts_3d_master_gt)
        if pred_mano_3d_mesh_kp_master is not None:
            self.MPJPE_KP_MASTER_3D.feed(pred_mano_3d_mesh_kp_master[:, self.num_hand_verts:], gt_kp=joints_3d_master_gt)
            self.MPVPE_KP_MASTER_3D.feed(pred_mano_3d_mesh_kp_master[:, :self.num_hand_verts], gt_kp=verts_3d_master_gt)
        if pred_mano_3d_mesh_mesh_master is not None:
            self.MPJPE_MESH_MASTER_3D.feed(pred_mano_3d_mesh_mesh_master[:, self.num_hand_verts:], gt_kp=joints_3d_master_gt)
            self.MPVPE_MESH_MASTER_3D.feed(pred_mano_3d_mesh_mesh_master[:, :self.num_hand_verts], gt_kp=verts_3d_master_gt)
        self.MPRPE_HAND_3D.feed(pred_ref_hand_joints_3d, gt_kp=joints_3d_master_gt)
        self.PA_SV.feed(pred_mano_3d_joints_sv, joints_3d_rel_gt, pred_mano_3d_verts_sv, verts_3d_rel_gt)
        if pred_mano_3d_mesh_kp_master is not None:
            pred_mano_3d_joints_kp_master_in_cam = batch_cam_extr_transf(
                T_c2m_gt, pred_mano_3d_mesh_kp_master[:, self.num_hand_verts:].unsqueeze(1).repeat(1, n_views, 1, 1)
            ).flatten(0, 1)
            pred_mano_3d_verts_kp_master_in_cam = batch_cam_extr_transf(
                T_c2m_gt, pred_mano_3d_mesh_kp_master[:, :self.num_hand_verts].unsqueeze(1).repeat(1, n_views, 1, 1)
            ).flatten(0, 1)
            self.PA_KP_MASTER.feed(
                pred_mano_3d_joints_kp_master_in_cam, joints_3d_gt, pred_mano_3d_verts_kp_master_in_cam, verts_3d_gt
            )
        if pred_mano_3d_mesh_mesh_master is not None:
            pred_mano_3d_joints_mesh_master_in_cam = batch_cam_extr_transf(
                T_c2m_gt, pred_mano_3d_mesh_mesh_master[:, self.num_hand_verts:].unsqueeze(1).repeat(1, n_views, 1, 1)
            ).flatten(0, 1)
            pred_mano_3d_verts_mesh_master_in_cam = batch_cam_extr_transf(
                T_c2m_gt, pred_mano_3d_mesh_mesh_master[:, :self.num_hand_verts].unsqueeze(1).repeat(1, n_views, 1, 1)
            ).flatten(0, 1)
            self.PA_MESH_MASTER.feed(
                pred_mano_3d_joints_mesh_master_in_cam, joints_3d_gt, pred_mano_3d_verts_mesh_master_in_cam, verts_3d_gt
            )
        obj_eval_rest = batch.get("master_obj_eval_rest", None) if use_full_recon_eval else None
        if obj_eval_rest is not None and preds.get("obj_rot6d", None) is not None and preds.get("obj_trans", None) is not None:
            pred_obj_dense_3d = self._build_object_points_from_hand_pose(
                obj_eval_rest,
                preds["obj_rot6d"][-1],
                pred_mano_3d_joints_master[:, self.center_idx:self.center_idx + 1],
                preds["obj_trans"][-1],
            )
            gt_obj_dense_3d = self._build_object_points_from_hand_pose(
                obj_eval_rest,
                batch["master_obj_rot6d_label"],
                joints_3d_master_gt[:, self.center_idx:self.center_idx + 1],
                batch["master_obj_t_label_rel"],
            )
            self.OBJ_RECON_MASTER.feed(pred_obj_dense_3d, gt_obj_dense_3d)
        else:
            self.OBJ_RECON_MASTER.feed(pred_obj_sparse_3d, obj_sparse_3d_master_gt)

        if (
            obj_eval_rest is not None
            and preds.get("obj_view_rot6d_cam", None) is not None
            and preds.get("obj_view_trans", None) is not None
            and batch.get("target_rot6d_label", None) is not None
            and batch.get("target_t_label_rel", None) is not None
            and batch.get("target_joints_3d", None) is not None
        ):
            pred_sv_obj_dense_3d = self._build_object_points_from_hand_pose(
                obj_eval_rest,
                preds["obj_view_rot6d_cam"],
                batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1],
                preds["obj_view_trans"],
            )
            gt_sv_obj_dense_3d = self._build_object_points_from_hand_pose(
                obj_eval_rest,
                batch["target_rot6d_label"],
                batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1],
                batch["target_t_label_rel"],
            )
            self.OBJ_RECON_SV.feed(pred_sv_obj_dense_3d.flatten(0, 1), gt_sv_obj_dense_3d.flatten(0, 1))
        elif obj_pc_sparse is not None and preds.get("obj_view_rot6d_cam", None) is not None and preds.get("obj_view_trans", None) is not None:
            pred_sv_obj_sparse_3d = self._build_object_points_from_hand_pose(
                batch["master_obj_sparse_rest"],
                preds["obj_view_rot6d_cam"],
                batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1],
                preds["obj_view_trans"],
            )
            self.OBJ_RECON_SV.feed(pred_sv_obj_sparse_3d.flatten(0, 1), obj_pc_sparse.flatten(0, 1))

        self.summary.add_scalar("MPJPE_SV_3D_val", self.MPJPE_SV_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_SV_3D_val", self.MPVPE_SV_3D.get_result(), step_idx)
        self.summary.add_scalar("MPJPE_MASTER_3D_val", self.MPJPE_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_MASTER_3D_val", self.MPVPE_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPJPE_KP_MASTER_3D_val", self.MPJPE_KP_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_KP_MASTER_3D_val", self.MPVPE_KP_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPJPE_MESH_MASTER_3D_val", self.MPJPE_MESH_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_MESH_MASTER_3D_val", self.MPVPE_MESH_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPRPE_HAND_3D_val", self.MPRPE_HAND_3D.get_result(), step_idx)
        self.summary.add_scalar("PA_SV_val", self.PA_SV.get_result(), step_idx)
        self.summary.add_scalar("PA_KP_MASTER_val", self.PA_KP_MASTER.get_result(), step_idx)
        self.summary.add_scalar("PA_MESH_MASTER_val", self.PA_MESH_MASTER.get_result(), step_idx)
        self.summary.add_scalar("OBJREC_MASTER_CD_val", self.OBJ_RECON_MASTER.cd.avg, step_idx)
        self.summary.add_scalar("OBJREC_MASTER_FS_5_val", self.OBJ_RECON_MASTER.fs_5.avg, step_idx)
        self.summary.add_scalar("OBJREC_MASTER_FS_10_val", self.OBJ_RECON_MASTER.fs_10.avg, step_idx)
        self.summary.add_scalar("OBJREC_SV_CD_val", self.OBJ_RECON_SV.cd.avg, step_idx)
        self.summary.add_scalar("OBJREC_SV_FS_5_val", self.OBJ_RECON_SV.fs_5.avg, step_idx)
        self.summary.add_scalar("OBJREC_SV_FS_10_val", self.OBJ_RECON_SV.fs_10.avg, step_idx)
        self.summary.add_scalar("VAL_RECON_IS_FULL_val", float(use_full_recon_eval), step_idx)
        if stage_name == "stage2":
            self.OBJ_POSE_VAL.feed(
                pred_rot6d=preds["obj_rot6d"][-1],
                pred_trans=preds["obj_trans"][-1],
                gt_rot6d=batch["master_obj_rot6d_label"],
                gt_trans=batch["master_obj_t_label_rel"],
                obj_points_rest=batch["master_obj_sparse_rest"],
            )
            obj_measures = self.OBJ_POSE_VAL.get_measures()
            self.summary.add_scalar("OBJ_ADD_val", obj_measures["Obj_add"], step_idx)
            self.summary.add_scalar("OBJ_ADDS_val", obj_measures["Obj_adds"], step_idx)
            self.summary.add_scalar("OBJ_TRANS_EPE_val", obj_measures["Obj_trans_epe"], step_idx)
            self.summary.add_scalar("OBJ_ROT_DEG_val", obj_measures["Obj_rot_deg"], step_idx)
        pose_metric_log_dict = self._pose_metrics_for_logging(pose_metric_dict)
        metric_log_dict = {**pose_metric_log_dict, **sv_obj_metric_dict}
        self.loss_metric.feed(metric_log_dict, batch_size)
        for k, v in metric_log_dict.items():
            self.summary.add_scalar(f"{k}_val", v.item(), step_idx)
        self._log_sv_view_debug_scalars(preds, batch, step_idx, suffix="_val")
        self._maybe_log_stage_transition_loss("val", epoch_idx, stage_name, {}, metric_log_dict)

        if step_idx % (self.train_log_interval * 5) == 0:
            self._log_visualizations("val", batch, preds, step_idx, stage_name)

        return None
    
    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_metric([
            self.MPJPE_SV_3D, self.MPVPE_SV_3D,
            self.MPJPE_MASTER_3D, self.MPVPE_MASTER_3D,
            self.MPJPE_KP_MASTER_3D, self.MPVPE_KP_MASTER_3D,
            self.MPJPE_MESH_MASTER_3D, self.MPVPE_MESH_MASTER_3D,
            self.MPRPE_HAND_3D,
            self.PA_SV, self.PA_KP_MASTER, self.PA_MESH_MASTER,
            self.OBJ_POSE_VAL, self.OBJ_RECON_SV, self.OBJ_RECON_MASTER,
        ], epoch_idx, comment=comment, summary=self.format_metric(mode="val_full"))
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.MPJPE_MASTER_3D.reset()
        self.MPVPE_MASTER_3D.reset()
        self.MPJPE_KP_MASTER_3D.reset()
        self.MPVPE_KP_MASTER_3D.reset()
        self.MPJPE_MESH_MASTER_3D.reset()
        self.MPVPE_MESH_MASTER_3D.reset()
        self.MPRPE_HAND_3D.reset()
        self.PA_SV.reset()
        self.PA_KP_MASTER.reset()
        self.PA_MESH_MASTER.reset()
        self.OBJ_POSE_VAL.reset()
        self.OBJ_RECON_SV.reset()
        self.OBJ_RECON_MASTER.reset()
        self.loss_metric.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        """
        [ModuleAbstract 要求] 测试步 (通常直接复用 validation_step)
        """
        return self.validation_step(batch, step_idx, **kwargs)
    
    def draw_step(self, batch, step_idx, **kwargs):
        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = self._stage2_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._log_visualizations("draw", batch, preds, step_idx, stage_name)
        return preds

    def get_metric(self, preds, batch, **kwargs):
        """
        [ModuleAbstract 预留] 离线指标计算
        """
        pass

    def format_metric(self, mode="train"):
        def _fmt_mm_value(value):
            return f"{float(value) * 1000.0:.1f}"

        def _fmt_mm_pair(v1, v2):
            return f"{float(v1) * 1000.0:.1f}/{float(v2) * 1000.0:.1f}"

        def _fmt_obj_recon(metric):
            return (
                f"{metric.fs_5.avg:.3f}/"
                f"{metric.fs_10.avg:.3f}/"
                f"{metric.cd.avg:.3f}"
            )

        def _get_loss(name, default=0.0):
            meter = self.loss_metric._losses.get(name, None)
            return float(meter.avg) if meter is not None else float(default)

        if mode == "train":
            l_hm = _get_loss('loss_heatmap_hand') + _get_loss('loss_heatmap_hand_map')
            l_obj_init_rot = _get_loss("loss_obj_init_rot")
            l_obj_init_trans = _get_loss("loss_obj_init_trans")
            l_obj_init = l_obj_init_rot + l_obj_init_trans
            l_obj_view_direct = (
                _get_loss("loss_obj_view_rot_geo")
                + _get_loss("loss_obj_view_rot_l1")
                + _get_loss("loss_obj_view_trans")
            )
            l_obj_view_points = _get_loss("loss_obj_view_points")
            l_obj_self = _get_loss("loss_obj_view_rot_self_geo") + _get_loss("loss_obj_view_rot_self_l1")
            l_obj_master_aux = _get_loss("loss_obj_rot") + _get_loss("loss_obj_trans")
            l_obj_master_points = _get_loss("loss_obj_points")
            m_obj_rot_deg = _get_loss('metric_obj_rot_deg')
            m_obj_trans = _get_loss('metric_obj_trans_epe')
            m_sv_obj_rot_deg = _get_loss('metric_sv_obj_rot_deg')
            m_sv_obj_trans = _get_loss('metric_sv_obj_trans_epe')
            l_total = _get_loss('loss')
            l_triang = _get_loss('loss_triang')
            l_3d_jts = _get_loss('loss_3d_jts')
            l_2d_proj = _get_loss('loss_2d_proj')

            stage_short = self._stage_display_name(self.current_stage_name, short=True)
            warmup_suffix = f" W{self.current_stage2_warmup:.2f}" if self.current_stage_name == "stage2" and self.current_stage2_warmup < 1.0 else ""
            if self.current_stage_name == "stage1":
                kp_j = self.MPJPE_KP_MASTER_3D.get_result()
                kp_v = self.MPVPE_KP_MASTER_3D.get_result()
                return (f"{stage_short}{warmup_suffix} | "
                        f"L {l_total:.3f} | "
                        f"B H/T/J/P {l_hm:.3f}/{l_triang:.3f}/{l_3d_jts:.3f}/{l_2d_proj:.3f} | "
                        f"SVL I/D/P/S {l_obj_init:.3f}/{l_obj_view_direct:.3f}/{l_obj_view_points:.3f}/{l_obj_self:.3f} | "
                        f"KP {_fmt_mm_pair(kp_j, kp_v)} | "
                        f"SV {m_sv_obj_rot_deg:.1f}/{_fmt_mm_value(m_sv_obj_trans)} | "
                        f"RS {_fmt_obj_recon(self.OBJ_RECON_SV)} "
                        f"RM {_fmt_obj_recon(self.OBJ_RECON_MASTER)}")
            if self.current_stage_name == "stage2":
                kp_j = self.MPJPE_KP_MASTER_3D.get_result()
                kp_v = self.MPVPE_KP_MASTER_3D.get_result()
                return (f"{stage_short}{warmup_suffix} | "
                        f"L {l_total:.3f} | "
                        f"B H/T/J {l_hm:.3f}/{l_triang:.3f}/{l_3d_jts:.3f} | "
                        f"SVL I/D/P/S {l_obj_init:.3f}/{l_obj_view_direct:.3f}/{l_obj_view_points:.3f}/{l_obj_self:.3f} | "
                        f"ML A/P {l_obj_master_aux:.3f}/{l_obj_master_points:.3f} | "
                        f"KP {_fmt_mm_pair(kp_j, kp_v)} | "
                        f"SV {m_sv_obj_rot_deg:.1f}/{_fmt_mm_value(m_sv_obj_trans)} "
                        f"M {m_obj_rot_deg:.1f}/{_fmt_mm_value(m_obj_trans)} | "
                        f"RS {_fmt_obj_recon(self.OBJ_RECON_SV)} "
                        f"RM {_fmt_obj_recon(self.OBJ_RECON_MASTER)}")
            kp_j = self.MPJPE_KP_MASTER_3D.get_result()
            kp_v = self.MPVPE_KP_MASTER_3D.get_result()
            return (f"{stage_short}{warmup_suffix} | "
                    f"L {l_total:.3f} | "
                    f"B H/T/J {l_hm:.3f}/{l_triang:.3f}/{l_3d_jts:.3f} | "
                    f"SVL I/D/P/S {l_obj_init:.3f}/{l_obj_view_direct:.3f}/{l_obj_view_points:.3f}/{l_obj_self:.3f} | "
                    f"ML A/P {l_obj_master_aux:.3f}/{l_obj_master_points:.3f} | "
                    f"KP {_fmt_mm_pair(kp_j, kp_v)} | "
                    f"SV {m_sv_obj_rot_deg:.1f}/{_fmt_mm_value(m_sv_obj_trans)} "
                    f"M {m_obj_rot_deg:.1f}/{_fmt_mm_value(m_obj_trans)} | "
                    f"RS {_fmt_obj_recon(self.OBJ_RECON_SV)} "
                    f"RM {_fmt_obj_recon(self.OBJ_RECON_MASTER)}")
        elif mode == "test":
            metric_toshow = [self.PA, self.MPJPE_HAND_3D]
        else:
            pa_sv = self.PA_SV.get_measures()
            pa_kp = self.PA_KP_MASTER.get_measures()
            pa_mesh = self.PA_MESH_MASTER.get_measures()
            sv_j = self.MPJPE_SV_3D.get_result()
            sv_v = self.MPVPE_SV_3D.get_result()
            kp_j = self.MPJPE_KP_MASTER_3D.get_result()
            kp_v = self.MPVPE_KP_MASTER_3D.get_result()
            mesh_j = self.MPJPE_MESH_MASTER_3D.get_result()
            mesh_v = self.MPVPE_MESH_MASTER_3D.get_result()
            ref_j = self.MPRPE_HAND_3D.get_result()
            sv_obj_rot_deg = _get_loss('metric_sv_obj_rot_deg')
            sv_obj_trans_epe = _get_loss('metric_sv_obj_trans_epe')
            l_obj_init = _get_loss("loss_obj_init_rot") + _get_loss("loss_obj_init_trans")
            l_obj_view_direct = (
                _get_loss("loss_obj_view_rot_geo")
                + _get_loss("loss_obj_view_rot_l1")
                + _get_loss("loss_obj_view_trans")
            )
            l_obj_view_points = _get_loss("loss_obj_view_points")
            l_obj_self = _get_loss("loss_obj_view_rot_self_geo") + _get_loss("loss_obj_view_rot_self_l1")
            l_obj_master_aux = _get_loss("loss_obj_rot") + _get_loss("loss_obj_trans")
            l_obj_master_points = _get_loss("loss_obj_points")

            def _fmt_triplet(name, pa_dict, mp, mv):
                if mp <= 0 and mv <= 0:
                    return f"{name} PA/MP/MV -/-/-"
                return (
                    f"{name} PAJ/PAV {_fmt_mm_pair(pa_dict['pa_mpjpe'], pa_dict['pa_mpvpe'])} "
                    f"MP/MV {_fmt_mm_pair(mp, mv)}"
                )

            def _fmt_pa_only(name, pa_dict, mp, mv):
                if mp <= 0 and mv <= 0:
                    return f"{name} -/-"
                return f"{name} {_fmt_mm_pair(pa_dict['pa_mpjpe'], pa_dict['pa_mpvpe'])}"

            def _fmt_triplet_full(name, pa_dict, mp, mv):
                if mp <= 0 and mv <= 0:
                    return f"{name} PA-MPJPE/PA-MPVPE/MPJPE/MPVPE -/-/-/-"
                return (
                    f"{name} PA-MPJPE/PA-MPVPE {_fmt_mm_pair(pa_dict['pa_mpjpe'], pa_dict['pa_mpvpe'])} "
                    f"MPJPE/MPVPE {_fmt_mm_pair(mp, mv)}"
                )

            stage_prefix = self._stage_display_name(self.current_stage_name, short=True)
            recon_mode = self.current_val_recon_mode.upper()

            if mode == "val_full":
                msg = " | ".join([
                    f"RecEval {recon_mode}",
                    _fmt_triplet_full("SingleView", pa_sv, sv_j, sv_v),
                    _fmt_triplet_full("KeypointMaster", pa_kp, kp_j, kp_v),
                    (
                        f"Object ADD/ADD-S {_fmt_mm_pair(self.OBJ_POSE_VAL.add.avg, self.OBJ_POSE_VAL.adds.avg)} "
                        f"TranslationEPE {_fmt_mm_value(self.OBJ_POSE_VAL.trans_epe.avg)} "
                        f"RotationError {self.OBJ_POSE_VAL.rot_deg.avg:.1f} deg"
                        if self.current_stage_name == "stage2" and self.OBJ_POSE_VAL.count > 0 else
                        "Object ADD/ADD-S/TranslationEPE/RotationError -/-/-/-"
                    ),
                    (
                        f"SVObjectRot/Trans {sv_obj_rot_deg:.1f} deg/{_fmt_mm_value(sv_obj_trans_epe)}"
                        if self.current_stage_name in ["stage1", "stage2"] else
                        "SVObjectRot/Trans -/-"
                    ),
                    f"SVLoss I/D/P/S {l_obj_init:.3f}/{l_obj_view_direct:.3f}/{l_obj_view_points:.3f}/{l_obj_self:.3f}",
                    (
                        f"MasterLoss Aux/Pts {l_obj_master_aux:.3f}/{l_obj_master_points:.3f}"
                        if self.current_stage_name == "stage2" else
                        "MasterLoss Aux/Pts -/-"
                    ),
                    f"SVObjectRecon FS@5/10/CD {_fmt_obj_recon(self.OBJ_RECON_SV)}",
                    f"MasterObjectRecon FS@5/10/CD {_fmt_obj_recon(self.OBJ_RECON_MASTER)}",
                    f"Reference Hand {_fmt_mm_value(ref_j)}",
                ])
                return f"{stage_prefix} | {msg}"

            msg = " | ".join([
                f"RecEval {recon_mode}",
                f"PA { _fmt_pa_only('SV', pa_sv, sv_j, sv_v) } { _fmt_pa_only('KP', pa_kp, kp_j, kp_v) }",
                (
                    f"Obj A/S {_fmt_mm_pair(self.OBJ_POSE_VAL.add.avg, self.OBJ_POSE_VAL.adds.avg)}"
                    if self.current_stage_name == "stage2" and self.OBJ_POSE_VAL.count > 0 else
                    "Obj A/S -/-"
                ),
                (
                    f"Rot {self.OBJ_POSE_VAL.rot_deg.avg:.1f} Tr {_fmt_mm_value(self.OBJ_POSE_VAL.trans_epe.avg)}"
                    if self.current_stage_name == "stage2" and self.OBJ_POSE_VAL.count > 0 else
                    "Rot/Tr -/-"
                ),
                (
                    f"SVRot {sv_obj_rot_deg:.1f} SVTr {_fmt_mm_value(sv_obj_trans_epe)}"
                    if self.current_stage_name in ["stage1", "stage2"] else
                    "SVRot/SVTr -/-"
                ),
                f"SVL I/D/P/S {l_obj_init:.3f}/{l_obj_view_direct:.3f}/{l_obj_view_points:.3f}/{l_obj_self:.3f}",
                (
                    f"ML A/P {l_obj_master_aux:.3f}/{l_obj_master_points:.3f}"
                    if self.current_stage_name == "stage2" else
                    "ML A/P -/-"
                ),
                f"SVObjRec {_fmt_obj_recon(self.OBJ_RECON_SV)}",
                f"MasterObjRec {_fmt_obj_recon(self.OBJ_RECON_MASTER)}",
                f"Ref {_fmt_mm_value(ref_j)}",
            ])
            return f"{stage_prefix} | {msg}"

        return " | ".join([str(me) for me in metric_toshow])
