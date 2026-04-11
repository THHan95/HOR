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
from ..metrics.pa_eval import PAEval
from ..metrics.rle import RLELoss
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import param_size
from ..utils.net_utils import init_weights, constant_init
from ..utils.recorder import Recorder
from ..utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, batch_persp_project, mano_to_openpose,
                               rot6d_to_rotmat)
from ..viztools.draw import (draw_batch_joint_images, draw_batch_mesh_images_pred,
                             draw_batch_hand_mesh_images_2d, draw_batch_joint_center_images, draw_batch_joint_pair_overlay_images,
                             draw_batch_joint_triplet_overlay_images, draw_batch_master_space_3d,
                             tile_batch_images)
# from .common.networks.tgs.models.snowflake.model_spdpp import SnowflakeModelSPDPP
# from .common.networks.tgs.models.snowflake.model_spdpp import mask_generation
from .backbones import build_backbone
from .bricks.conv import ConvBlock
from .bricks.utils import ManoDecoder, Linear, SelfAttn, HOT, GraphRegression, Proj2World
from .model_abstraction import ModuleAbstract
from .heads import build_head
from pytorch3d.loss import chamfer_distance
from lib.criterions.emd.emd import earth_mover_distance

# 引入 RLE 的核心流模型 (请确保把 rlepose 的 real_nvp.py 放在了 lib/models/layers/ 下)
from .layers.real_nvp import RealNVP

@MODEL.register_module()
class POEM_RLE(nn.Module, ModuleAbstract):
    def __init__(self, cfg):
        super(POEM_RLE, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET
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

        # self.pointcloud_upsampler = SnowflakeModelSPDPP(
        #     input_channels=256,
        #     dim_feat=256,
        #     num_p0=2048,
        #     up_factors=[2, 4],
        #     SPD_type='SPD_PP',
        #     token_type='image_token'
        # )
        
        self._build_heads()
        self._build_flows()

        if self.joints_loss_type == "l2":
            self.criterion_joints = torch.nn.MSELoss()
        else:
            self.criterion_joints = torch.nn.L1Loss()

        self.coord_loss = nn.L1Loss()
        self.criterion_rle = RLELoss()
        self.chamfer_loss = chamfer_distance
        self.earth_mover_loss = earth_mover_distance
        self.obj_pose_rot_weight = cfg.LOSS.get("OBJ_POSE_ROT_N", 1.0)
        self.obj_pose_rot6d_weight = cfg.LOSS.get("OBJ_POSE_ROT6D_N", 0.1)
        self.obj_pose_trans_weight = cfg.LOSS.get("OBJ_POSE_TRANS_N", 5.0)
        self.obj_pose_points_weight = cfg.LOSS.get("OBJ_POSE_POINTS_N", 10.0)
        self.obj_chamfer_weight = cfg.LOSS.get("OBJ_CHAMFER_N", 5.0)
        self.obj_emd_weight = cfg.LOSS.get("OBJ_EMD_N", 1.0)
        self.obj_penetration_weight = cfg.LOSS.get("OBJ_PENETRATION_N", 5.0)
        self.decoder_hand_weight = cfg.LOSS.get("DECODER_HAND_N", 10.0)
        self.decoder_proj_weight = cfg.LOSS.get("DECODER_PROJ_N", 10.0)
        self.triangulation_weight = cfg.LOSS.get("TRIANGULATION_N", 10.0)
        self.triangulation_hand_weight = cfg.LOSS.get("TRIANGULATION_HAND_N", self.triangulation_weight)
        self.triangulation_obj_weight = cfg.LOSS.get("TRIANGULATION_OBJ_N", 1.0)
        self.mano_consistency_weight = cfg.LOSS.get("MANO_CONSIST_N", 10.0)
        self.mano_proj_weight = cfg.LOSS.get("MANO_PROJ_N", 50.0)
        self.stage1_end_epoch = cfg.TRAIN.get("STAGE1_END_EPOCH", 4)
        self.stage2_end_epoch = cfg.TRAIN.get("STAGE2_END_EPOCH", 11)
        self.stage3_warmup_epochs = cfg.TRAIN.get("STAGE3_WARMUP_EPOCHS", 3)
        self.current_stage_name = "stage3"
        self.current_stage3_warmup = 1.0
        self._debug_stage_log_state = {"train": None, "val": None}

        self.loss_metric = LossMetric(cfg)
        self.PA_SV = PAEval(cfg, mesh_score=True)
        self.PA_KP_MASTER = PAEval(cfg, mesh_score=True)
        self.PA_MESH_MASTER = PAEval(cfg, mesh_score=True)
        self.OBJ_POSE_VAL = ObjectPoseMetric(cfg, name="Obj")
        self.MPJPE_MASTER_3D = MeanEPE(cfg, "Master_J")
        self.MPVPE_MASTER_3D = MeanEPE(cfg, "Master_V")
        self.MPJPE_KP_MASTER_3D = MeanEPE(cfg, "KP_Master_J")
        self.MPVPE_KP_MASTER_3D = MeanEPE(cfg, "KP_Master_V")
        self.MPJPE_MESH_MASTER_3D = MeanEPE(cfg, "Mesh_Master_J")
        self.MPVPE_MESH_MASTER_3D = MeanEPE(cfg, "Mesh_Master_V")
        self.MPJPE_SV_3D = MeanEPE(cfg, "SV_J")
        self.MPVPE_SV_3D = MeanEPE(cfg, "SV_V")
        self.MPRPE_HAND_3D = MeanEPE(cfg, "Ref_J")
        self.MPRPE_OBJ_3D = MeanEPE(cfg, "Ref_O")
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
        self._set_module_requires_grad(self.hand_mlp, True)
        self._set_module_requires_grad(self.fc_hand_coord, True)
        self._set_module_requires_grad(self.fc_hand_sigma, True)
        self._set_module_requires_grad(self.obj_mlp, True)
        self._set_module_requires_grad(self.fc_obj_coord, True)
        self._set_module_requires_grad(self.fc_obj_sigma, True)
        self._set_module_requires_grad(self.hot_pose, True)
        self._set_module_requires_grad(self.att_0, True)
        self._set_module_requires_grad(self.att_1, True)
        self._set_module_requires_grad(self.mano_fuse, True)
        self._set_module_requires_grad(self.mano_fc, True)
        self._set_module_requires_grad(self.flow_hand, True)
        self._set_module_requires_grad(self.flow_obj, True)
        self._set_module_requires_grad(self.ptEmb_head, True)
        if hasattr(self.ptEmb_head, "center_shift_layer"):
            self._set_module_requires_grad(self.ptEmb_head.center_shift_layer, False)
        if hasattr(self.ptEmb_head, "position_encoder"):
            self._set_module_requires_grad(self.ptEmb_head.position_encoder, False)
        if hasattr(self.ptEmb_head, "transformer"):
            for block in self.ptEmb_head.transformer._iter_blocks():
                self._set_module_requires_grad(block.hand_anchor_update_layer, False)
                self._set_module_requires_grad(block.hand_refine_layer, False)
                self._set_module_requires_grad(block.obj_update_layer.vec_attn.reg_branch, False)

        if stage_name == "stage1":
            # Stage1 only optimizes the single-view hand branch and triangulation hand prior.
            self._set_module_requires_grad(self.obj_mlp, False)
            self._set_module_requires_grad(self.fc_obj_coord, False)
            self._set_module_requires_grad(self.fc_obj_sigma, False)
            self._set_module_requires_grad(self.flow_obj, False)
            self._set_module_requires_grad(self.ptEmb_head, False)
        elif stage_name == "stage2":
            # Stage2 uses hand refinement but still does not supervise object refinement or mesh-master branch.
            self._set_module_requires_grad(self.obj_mlp, False)
            self._set_module_requires_grad(self.fc_obj_coord, False)
            self._set_module_requires_grad(self.fc_obj_sigma, False)
            self._set_module_requires_grad(self.flow_obj, False)
            if hasattr(self.ptEmb_head, "transformer"):
                for block in self.ptEmb_head.transformer._iter_blocks():
                    self._set_module_requires_grad(block.obj_update_layer, False)
            self._set_module_requires_grad(self.ptEmb_head.obj_init_rot_regressor, False)
            self._set_module_requires_grad(self.ptEmb_head.obj_pose_delta_regressor, False)
            self._set_module_requires_grad(self.ptEmb_head.obj_feat_fuser, False)
            self._set_module_requires_grad(self.ptEmb_head.final_mesh_mano_regressor, False)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
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

    def _build_heads(self):
        """初始化特征解耦 MLP 和 预测头"""
        # 手部专属 MLP
        self.hand_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), nn.BatchNorm1d(self.feat_dim), nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_size[1]), nn.BatchNorm1d(self.feat_size[1]), nn.ReLU(inplace=True)
        )
        # 物体专属 MLP
        self.obj_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), nn.BatchNorm1d(self.feat_dim), nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_size[1]), nn.BatchNorm1d(self.feat_size[1]), nn.ReLU(inplace=True)
        )
        
        self.fc_hand_coord = Linear(self.feat_size[1], self.num_hand_joints * 2)
        self.fc_hand_sigma = Linear(self.feat_size[1], self.num_hand_joints * 2, norm=False)
        self.fc_obj_coord = Linear(self.feat_size[1], self.num_obj_joints * 2)
        self.fc_obj_sigma = Linear(self.feat_size[1], self.num_obj_joints * 2, norm=False)

        self.fc_layers = [self.fc_hand_coord, self.fc_hand_sigma, self.fc_obj_coord, self.fc_obj_sigma]

    def _build_flows(self):
        """初始化 RLE 的 Normalizing Flow (RealNVP)"""
        def nets():
            return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())
        def nett():
            return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))
            
        prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float32)
        
        self.flow_hand = RealNVP(nets, nett, masks, prior)
        self.flow_obj = RealNVP(nets, nett, masks, prior)

    def _resolve_stage(self, epoch_idx):
        if epoch_idx < self.stage1_end_epoch:
            return "stage1"
        if epoch_idx < self.stage2_end_epoch:
            return "stage2"
        return "stage3"

    @staticmethod
    def _stage_to_interaction_mode(stage_name):
        return "ho" if stage_name == "stage3" else "hand"

    def _stage3_object_warmup(self, epoch_idx, stage_name):
        if stage_name != "stage3":
            return 0.0
        if self.stage3_warmup_epochs <= 0:
            return 1.0
        progress = (epoch_idx - self.stage2_end_epoch + 1) / float(self.stage3_warmup_epochs)
        return float(max(0.0, min(1.0, progress)))

    def _maybe_log_stage_transition(self, mode, epoch_idx, stage_name, preds, batch):
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
            f"[{mode}][StageDebug] epoch={epoch_idx} stage={stage_name} "
            f"interaction={interaction_mode} warmup={self.current_stage3_warmup:.2f} "
            f"image={tuple(batch['image'].shape)} "
            f"master_obj_sparse_rest={tuple(batch['master_obj_sparse_rest'].shape) if 'master_obj_sparse_rest' in batch else None} "
            f"ref_hand={tuple(preds['ref_hand'].shape) if 'ref_hand' in preds else None} "
            f"ref_obj={tuple(preds['ref_obj'].shape) if 'ref_obj' in preds else None} "
            f"mano_sv={tuple(preds['mano_3d_mesh_sv'].shape) if 'mano_3d_mesh_sv' in preds else None} "
            f"mano_master={tuple(preds['mano_3d_mesh_master'].shape) if 'mano_3d_mesh_master' in preds else None} "
            f"hand_states={_shape(hand_states)} "
            f"anchor_states={_shape(anchor_states)} "
            f"obj_states={_shape(obj_states)} "
            f"obj_rot6d={_shape(obj_rot_states)} "
            f"obj_trans={_shape(obj_trans_states)}"
        )

    def _maybe_log_stage_transition_loss(self, mode, epoch_idx, stage_name, loss_dict, pose_metric_dict=None):
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
            f"[{mode}][StageDebugLoss] epoch={epoch_idx} stage={stage_name} "
            f"loss={(_to_float('loss') or 0.0):.4f} "
            f"rle={((_to_float('loss_rle_hand') or 0.0) + (_to_float('loss_rle_obj') or 0.0)):.4f} "
            f"triang={(_to_float('loss_triang') or 0.0):.4f} "
            f"triang_h={(_to_float('loss_triang_hand') or 0.0):.4f} "
            f"triang_o={(_to_float('loss_triang_obj') or 0.0):.4f} "
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
            "metric_obj_center_epe": zero,
            "metric_obj_add": zero,
            "metric_obj_adds": zero,
        }

    @staticmethod
    def _normed_uv_to_pixel(coords_uv, image_hw):
        img_h, img_w = image_hw
        scale = coords_uv.new_tensor([img_w, img_h])
        return (coords_uv + 1.0) * 0.5 * scale

    @staticmethod
    def _project_master_points(points_master, T_c2m, K):
        points_mv = points_master.unsqueeze(1).repeat(1, T_c2m.shape[1], 1, 1)
        points_cam = batch_cam_extr_transf(T_c2m, points_mv)
        points_2d = batch_cam_intr_projection(K, points_cam)
        return points_cam, points_2d

    def _log_visualizations(self, mode, batch, preds, step_idx, stage_name):
        img = batch["image"]
        batch_size, n_views = img.shape[:2]
        img_h, img_w = img.shape[-2:]
        batch_id = min(0, batch_size - 1)

        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])

        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"].view(batch_size, n_views, 799, 2)
        pred_mano_2d_joints_sv = pred_mano_2d_mesh_sv[:, :, self.num_hand_verts:]
        pred_mano_2d_verts_sv = pred_mano_2d_mesh_sv[:, :, :self.num_hand_verts]

        pred_ref_joints_2d = preds["pred_hand"].view(batch_size, n_views, self.num_hand_joints, 2)
        pred_ref_center_2d = preds["pred_obj"].view(batch_size, n_views, 1, 2)

        gt_joints_2d = self._normed_uv_to_pixel(batch["target_joints_uvd"][..., :2], (img_h, img_w))
        gt_obj_center_2d = self._normed_uv_to_pixel(batch["target_obj_center_uv"], (img_h, img_w))
        pred_sv_joints_2d = self._normed_uv_to_pixel(pred_mano_2d_joints_sv, (img_h, img_w))
        pred_sv_verts_2d = self._normed_uv_to_pixel(pred_mano_2d_verts_sv, (img_h, img_w))
        pred_ref_joints_2d = self._normed_uv_to_pixel(pred_ref_joints_2d, (img_h, img_w))
        pred_ref_center_2d = self._normed_uv_to_pixel(pred_ref_center_2d, (img_h, img_w))

        pred_master_mesh = preds["mano_3d_mesh_master"]
        _, pred_master_mesh_2d = self._project_master_points(pred_master_mesh, T_c2m_gt, K_gt)
        pred_master_joints_2d = pred_master_mesh_2d[:, :, self.num_hand_verts:]
        pred_master_verts_2d = pred_master_mesh_2d[:, :, :self.num_hand_verts]
        pred_kp_master_mesh = preds.get("mano_3d_mesh_kp_master", None)
        pred_mesh_master_mesh = preds.get("mano_3d_mesh_mesh_master", None)
        if pred_kp_master_mesh is not None:
            _, pred_kp_master_mesh_2d = self._project_master_points(pred_kp_master_mesh, T_c2m_gt, K_gt)
            pred_kp_master_joints_2d = pred_kp_master_mesh_2d[:, :, self.num_hand_verts:]
            pred_kp_master_verts_2d = pred_kp_master_mesh_2d[:, :, :self.num_hand_verts]
        else:
            pred_kp_master_joints_2d = None
            pred_kp_master_verts_2d = None
        if pred_mesh_master_mesh is not None:
            _, pred_mesh_master_mesh_2d = self._project_master_points(pred_mesh_master_mesh, T_c2m_gt, K_gt)
            pred_mesh_master_joints_2d = pred_mesh_master_mesh_2d[:, :, self.num_hand_verts:]
            pred_mesh_master_verts_2d = pred_mesh_master_mesh_2d[:, :, :self.num_hand_verts]
        else:
            pred_mesh_master_joints_2d = None
            pred_mesh_master_verts_2d = None

        pred_obj_center = preds["obj_center_xyz_master"][-1]
        pred_obj_sparse = preds["obj_xyz_master"][-1]
        _, pred_obj_center_2d_proj = self._project_master_points(pred_obj_center, T_c2m_gt, K_gt)
        _, pred_obj_sparse_2d_proj = self._project_master_points(pred_obj_sparse, T_c2m_gt, K_gt)

        gt_obj_sparse = batch["master_obj_sparse"]
        _, gt_obj_sparse_2d_proj = self._project_master_points(gt_obj_sparse, T_c2m_gt, K_gt)

        img_views = img[batch_id]
        gt_joints_2d_views = gt_joints_2d[batch_id]
        gt_obj_center_2d_views = gt_obj_center_2d[batch_id]
        pred_sv_joints_2d_views = pred_sv_joints_2d[batch_id]
        pred_sv_verts_2d_views = pred_sv_verts_2d[batch_id]
        pred_ref_joints_2d_views = pred_ref_joints_2d[batch_id]
        pred_ref_center_2d_views = pred_ref_center_2d[batch_id]
        pred_master_joints_2d_views = pred_master_joints_2d[batch_id]
        pred_master_verts_2d_views = pred_master_verts_2d[batch_id]
        pred_kp_master_joints_2d_views = pred_kp_master_joints_2d[batch_id] if pred_kp_master_joints_2d is not None else None
        pred_kp_master_verts_2d_views = pred_kp_master_verts_2d[batch_id] if pred_kp_master_verts_2d is not None else None
        pred_mesh_master_joints_2d_views = pred_mesh_master_joints_2d[batch_id] if pred_mesh_master_joints_2d is not None else None
        pred_mesh_master_verts_2d_views = pred_mesh_master_verts_2d[batch_id] if pred_mesh_master_verts_2d is not None else None
        pred_obj_center_2d_views = pred_obj_center_2d_proj[batch_id]
        pred_obj_sparse_2d_views = pred_obj_sparse_2d_proj[batch_id]
        gt_obj_sparse_2d_views = gt_obj_sparse_2d_proj[batch_id]
        gt_verts_2d_views = self._normed_uv_to_pixel(batch["target_verts_uvd"][batch_id, ..., :2], (img_h, img_w))

        tag_prefix = f"img/{mode}_{stage_name}"
        self.summary.add_image(
            f"{tag_prefix}/sv_joints_2d",
            tile_batch_images(draw_batch_joint_images(pred_sv_joints_2d_views, gt_joints_2d_views, img_views, step_idx, n_sample=n_views)),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/sv_mesh_2d",
            tile_batch_images(draw_batch_hand_mesh_images_2d(
                gt_verts2d=gt_verts_2d_views,
                pred_verts2d=pred_sv_verts_2d_views,
                face=self.face,
                tensor_image=img_views,
                n_sample=n_views,
            )),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/ref_joints_obj_center_2d",
            tile_batch_images(draw_batch_joint_center_images(pred_ref_joints_2d_views, gt_joints_2d_views, pred_ref_center_2d_views, gt_obj_center_2d_views, img_views, n_sample=n_views)),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/master_joints_2d",
            tile_batch_images(draw_batch_joint_images(pred_master_joints_2d_views, gt_joints_2d_views, img_views, step_idx, n_sample=n_views)),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/master_mesh_2d",
            tile_batch_images(draw_batch_hand_mesh_images_2d(
                gt_verts2d=gt_verts_2d_views,
                pred_verts2d=pred_master_verts_2d_views,
                face=self.face,
                tensor_image=img_views,
                n_sample=n_views,
            )),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/joint_triplet_overlay_2d",
            tile_batch_images(draw_batch_joint_triplet_overlay_images(
                pred_sv_joints_2d_views,
                pred_master_joints_2d_views,
                gt_joints_2d_views,
                img_views,
                n_sample=n_views,
            )),
            step_idx,
            dataformats="HWC",
        )
        if pred_kp_master_joints_2d_views is not None and stage_name == "stage3":
            self.summary.add_image(
                f"{tag_prefix}/kp_master_joints_2d",
                tile_batch_images(draw_batch_joint_images(pred_kp_master_joints_2d_views, gt_joints_2d_views, img_views, step_idx, n_sample=n_views)),
                step_idx,
                dataformats="HWC",
            )
            self.summary.add_image(
                f"{tag_prefix}/kp_master_mesh_2d",
                tile_batch_images(draw_batch_hand_mesh_images_2d(
                    gt_verts2d=gt_verts_2d_views,
                    pred_verts2d=pred_kp_master_verts_2d_views,
                    face=self.face,
                    tensor_image=img_views,
                    n_sample=n_views,
                )),
                step_idx,
                dataformats="HWC",
            )
        if pred_mesh_master_joints_2d_views is not None and stage_name == "stage3":
            self.summary.add_image(
                f"{tag_prefix}/mesh_master_joints_2d",
                tile_batch_images(draw_batch_joint_images(pred_mesh_master_joints_2d_views, gt_joints_2d_views, img_views, step_idx, n_sample=n_views)),
                step_idx,
                dataformats="HWC",
            )
            self.summary.add_image(
                f"{tag_prefix}/mesh_master_mesh_2d",
                tile_batch_images(draw_batch_hand_mesh_images_2d(
                    gt_verts2d=gt_verts_2d_views,
                    pred_verts2d=pred_mesh_master_verts_2d_views,
                    face=self.face,
                    tensor_image=img_views,
                    n_sample=n_views,
                )),
                step_idx,
                dataformats="HWC",
            )
        if stage_name == "stage3":
            if pred_kp_master_verts_2d_views is not None:
                self.summary.add_image(
                    f"{tag_prefix}/kp_master_mesh_obj_2d",
                    tile_batch_images(draw_batch_mesh_images_pred(
                        gt_verts2d=gt_verts_2d_views,
                        pred_verts2d=pred_kp_master_verts_2d_views,
                        face=self.face,
                        gt_obj2d=gt_obj_sparse_2d_views,
                        pred_obj2d=pred_obj_sparse_2d_views,
                        gt_objc2d=gt_obj_center_2d_views,
                        pred_objc2d=pred_obj_center_2d_views,
                        intr=K_gt[batch_id],
                        tensor_image=img_views,
                        n_sample=n_views,
                    )),
                    step_idx,
                    dataformats="HWC",
                )
            self.summary.add_image(
                f"{tag_prefix}/master_mesh_obj_2d",
                tile_batch_images(draw_batch_mesh_images_pred(
                    gt_verts2d=gt_verts_2d_views,
                    pred_verts2d=pred_master_verts_2d_views,
                    face=self.face,
                    gt_obj2d=gt_obj_sparse_2d_views,
                    pred_obj2d=pred_obj_sparse_2d_views,
                    gt_objc2d=gt_obj_center_2d_views,
                    pred_objc2d=pred_obj_center_2d_views,
                    intr=K_gt[batch_id],
                    tensor_image=img_views,
                    n_sample=n_views,
                )),
                step_idx,
                dataformats="HWC",
            )
        pred_hand_master = preds["hand_mesh_xyz_master"][-1]
        pred_hand_joints_master = pred_hand_master[:, self.num_hand_verts:]
        pred_hand_verts_master = pred_hand_master[:, :self.num_hand_verts]
        pred_hand_anchors_master = preds["all_hand_joints_xyz_master"][-1] if "all_hand_joints_xyz_master" in preds else None
        self.summary.add_image(
            f"{tag_prefix}/master_space_3d",
            tile_batch_images(draw_batch_master_space_3d(
                gt_hand_joints=batch["master_joints_3d"],
                pred_hand_joints=pred_hand_joints_master,
                gt_hand_verts=batch["master_verts_3d"],
                pred_hand_verts=pred_hand_verts_master,
                gt_obj_pts=batch["master_obj_sparse"] if stage_name == "stage3" else None,
                pred_obj_pts=pred_obj_sparse if stage_name == "stage3" else None,
                pred_hand_anchors=pred_hand_anchors_master,
                n_sample=min(batch_size, 2),
                image_size=(320, 320),
            ), max_cols=2),
            step_idx,
            dataformats="HWC",
        )
        if stage_name == "stage3" and pred_kp_master_mesh is not None:
            self.summary.add_image(
                f"{tag_prefix}/kp_master_space_3d",
                tile_batch_images(draw_batch_master_space_3d(
                    gt_hand_joints=batch["master_joints_3d"],
                    pred_hand_joints=pred_kp_master_mesh[:, self.num_hand_verts:],
                    gt_hand_verts=batch["master_verts_3d"],
                    pred_hand_verts=pred_kp_master_mesh[:, :self.num_hand_verts],
                    gt_obj_pts=batch["master_obj_sparse"],
                    pred_obj_pts=pred_obj_sparse,
                    pred_hand_anchors=pred_hand_anchors_master,
                    n_sample=min(batch_size, 2),
                    image_size=(320, 320),
                ), max_cols=2),
                step_idx,
                dataformats="HWC",
            )
        if stage_name == "stage3" and pred_mesh_master_mesh is not None:
            self.summary.add_image(
                f"{tag_prefix}/mesh_master_space_3d",
                tile_batch_images(draw_batch_master_space_3d(
                    gt_hand_joints=batch["master_joints_3d"],
                    pred_hand_joints=pred_mesh_master_mesh[:, self.num_hand_verts:],
                    gt_hand_verts=batch["master_verts_3d"],
                    pred_hand_verts=pred_mesh_master_mesh[:, :self.num_hand_verts],
                    gt_obj_pts=batch["master_obj_sparse"],
                    pred_obj_pts=pred_obj_sparse,
                    pred_hand_anchors=pred_hand_anchors_master,
                    n_sample=min(batch_size, 2),
                    image_size=(320, 320),
                ), max_cols=2),
                step_idx,
                dataformats="HWC",
            )
        if preds.get("interaction_mode", "ho") == "hand" and "all_hand_joints_xyz_master" in preds:
            _, pred_hand_anchor_2d = self._project_master_points(preds["all_hand_joints_xyz_master"][-1], T_c2m_gt, K_gt)
            pred_hand_anchor_2d_views = pred_hand_anchor_2d[batch_id]
            self.summary.add_image(
                f"{tag_prefix}/hand_anchor_refinement_2d",
                tile_batch_images(draw_batch_joint_pair_overlay_images(
                    pred_ref_joints_2d_views,
                    pred_hand_anchor_2d_views,
                    img_views,
                    n_sample=n_views,
                )),
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

        # NOTE: @licheng  merging multi-level features in bacbone.
        mlvl_feat = self.feat_decode(img_feats)  # (BxN, 128, 32, 32)
        mlvl_feat = mlvl_feat.view(batch_size, num_cams, *mlvl_feat.shape[1:])  # (B, N, 128, 32, 32)
    
        hand_feat = self.hand_mlp(global_feat) # (B*N, 256)
        obj_feat = self.obj_mlp(global_feat)   # (B*N, 256)
        
        # The RLE branch is sensitive to reduced precision. Keep coordinate/sigma heads in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            hand_coord_pred = self.fc_hand_coord(hand_feat.float()).view(batch_size * num_cams, self.num_hand_joints, 2)
            hand_sigma_pred = self.fc_hand_sigma(hand_feat.float()).view(batch_size * num_cams, self.num_hand_joints, -1)
            obj_coord_pred = self.fc_obj_coord(obj_feat.float()).view(batch_size * num_cams, self.num_obj_joints, 2)
            obj_sigma_pred = self.fc_obj_sigma(obj_feat.float()).view(batch_size * num_cams, self.num_obj_joints, -1)

        pred_hand_jts = hand_coord_pred
        hand_sigma = hand_sigma_pred.sigmoid()
        hand_sigma = torch.clamp(hand_sigma, min=0.01, max=0.99)  # Prevent numerical instability
        hand_score = 1 - hand_sigma.detach()
        hand_score = torch.mean(hand_score, dim=2, keepdim=True)  # (B*V, J_hand, 1)

        pred_obj_jts = obj_coord_pred
        obj_sigma = obj_sigma_pred.sigmoid()
        obj_sigma = torch.clamp(obj_sigma, min=0.01, max=0.99)  # Prevent numerical instability
        obj_score = 1 - obj_sigma.detach()
        obj_score = torch.mean(obj_score, dim=2, keepdim=True)  # (B*V, J_obj, 1)

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

        if self.training:
            joints_vis = batch["target_joints_vis"].flatten(0, 1).unsqueeze(-1)  # (B*V, J, 1)
            obj_center_vis = batch["target_obj_center_vis"].flatten(0, 1).unsqueeze(-1)  # (B*V, 1, 1)
            gt_hand_uv = batch["target_joints_uvd"][..., :2].reshape(pred_hand_jts.shape)
            gt_obj_uv = batch["target_obj_center_uv"][..., :2].reshape(pred_obj_jts.shape)

            # # ==========================================
            # # 🌟 抢救排查代码：拦截 NaN 病毒
            # # ==========================================
            # if torch.isnan(pred_hand_jts).any():
            #     print("\n[FATAL ERROR] pred_hand_jts contains NaN!")
            # if torch.isnan(gt_hand_uv).any():
            #     print("\n[FATAL ERROR] gt_hand_uv (DataLoader input) contains NaN!")
            # if torch.isnan(hand_sigma).any():
            #     print("\n[FATAL ERROR] hand_sigma contains NaN!")
            # if torch.isnan(obj_sigma).any():
            #     print("\n[FATAL ERROR] obj_sigma contains NaN!")
            # # ==========================================
            
            # RealNVP is numerically fragile under AMP. Keep the whole RLE/flow path in FP32.
            with torch.cuda.amp.autocast(enabled=False):
                pred_hand_jts_fp32 = pred_hand_jts.float()
                pred_obj_jts_fp32 = pred_obj_jts.float()
                gt_hand_uv_fp32 = gt_hand_uv.float()
                gt_obj_uv_fp32 = gt_obj_uv.float()
                hand_sigma_fp32 = hand_sigma.float().clamp(min=1e-4)
                obj_sigma_fp32 = obj_sigma.float().clamp(min=1e-4)
                joints_vis_fp32 = joints_vis.float()
                obj_center_vis_fp32 = obj_center_vis.float()

                hand_bar_mu = (pred_hand_jts_fp32 - gt_hand_uv_fp32) / hand_sigma_fp32 * joints_vis_fp32
                obj_bar_mu = (pred_obj_jts_fp32 - gt_obj_uv_fp32) / obj_sigma_fp32 * obj_center_vis_fp32
                if not torch.isfinite(pred_hand_jts_fp32).all():
                    logger.error("[RLEFinite] pred_hand_jts became non-finite before flow")
                    raise RuntimeError("pred_hand_jts became non-finite before flow")
                if not torch.isfinite(hand_sigma_fp32).all():
                    logger.error("[RLEFinite] hand_sigma became non-finite before flow")
                    raise RuntimeError("hand_sigma became non-finite before flow")
                if not torch.isfinite(gt_hand_uv_fp32).all():
                    logger.error("[RLEFinite] gt_hand_uv became non-finite before flow")
                    raise RuntimeError("gt_hand_uv became non-finite before flow")
                if not torch.isfinite(hand_bar_mu).all():
                    logger.error(
                        f"[RLEFinite] hand_bar_mu became non-finite "
                        f"pred_hand_absmax={pred_hand_jts_fp32.abs().max().item():.4f} "
                        f"hand_sigma_min={hand_sigma_fp32.min().item():.6f} "
                        f"hand_sigma_max={hand_sigma_fp32.max().item():.6f} "
                        f"gt_hand_absmax={gt_hand_uv_fp32.abs().max().item():.4f}"
                    )
                    raise RuntimeError("hand_bar_mu became non-finite before flow")
                if not torch.isfinite(obj_bar_mu).all():
                    logger.error(
                        f"[RLEFinite] obj_bar_mu became non-finite "
                        f"pred_obj_absmax={pred_obj_jts_fp32.abs().max().item():.4f} "
                        f"obj_sigma_min={obj_sigma_fp32.min().item():.6f} "
                        f"obj_sigma_max={obj_sigma_fp32.max().item():.6f} "
                        f"gt_obj_absmax={gt_obj_uv_fp32.abs().max().item():.4f}"
                    )
                    raise RuntimeError("obj_bar_mu became non-finite before flow")

                hand_log_phi = self.flow_hand.log_prob(hand_bar_mu.view(-1, 2)).reshape(batch_size*num_cams, self.num_hand_joints, 1)
                obj_log_phi = self.flow_obj.log_prob(obj_bar_mu.view(-1, 2)).reshape(batch_size*num_cams, self.num_obj_joints, 1)

                hand_nf_loss = torch.log(hand_sigma_fp32) - hand_log_phi
                obj_nf_loss = torch.log(obj_sigma_fp32) - obj_log_phi
            # total_nf_loss = hand_nf_loss.mean() + obj_nf_loss.mean()
        else:
            hand_nf_loss = None
            obj_nf_loss = None

        sv_preds = {
            "pred_hand": pred_hand_jts,
            "sigma_hand": hand_sigma,
            "pred_obj": pred_obj_jts,
            "sigma_obj": obj_sigma,
            "hand_nf_loss": hand_nf_loss,
            "obj_nf_loss": obj_nf_loss,
            "conf_hand": hand_score,
            "conf_obj": obj_score,
            "mano_3d_mesh_sv": coord_xyz_sv,
            "mano_2d_mesh_sv": coord_uv_sv,
            "mano_pose_euler_sv": pose_euler_sv,
            "mano_shape_sv": shape_sv,
        }

        # 🌟 增加判断逻辑：如果是 'sv' 模式，直接提前返回 sv_preds
        # 假设你传入的参数名叫 mode (需要在函数的 def 定义里加上 mode="mv" 或者 mode="sv")
        if self.mode == 'sv':
            return sv_preds

        # ==========================================
        # 如果不是 'sv'，继续执行后续的多视角/特征增强逻辑
        # ==========================================

        ref_hand, ref_obj = self.project(pred_hand_jts, pred_obj_jts, hand_score, obj_score, K, T_c2m, batch_size, num_cams)  # (B, J_hand, 3), (B, J_obj, 3)

        proj_preds = {
            "ref_hand": ref_hand,
            "ref_obj": ref_obj,
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
            reference_hand=ref_hand,
            reference_obj=ref_obj,
            obj_template=batch["master_obj_sparse_rest"],
            interaction_mode=interaction_mode,
            stage3_warmup=self.current_stage3_warmup,
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
        
        # 3. 提取并重命名 pt_preds 中的关键信息放入最终字典
        final_preds.update({
            "hand_mesh_xyz_master": pt_preds["all_hand_mesh_xyz_master"],
            "obj_xyz_master": pt_preds["all_obj_xyz_master"],
            "obj_center_xyz_master": pt_preds["all_obj_center_xyz_master"],
            "obj_rot6d": pt_preds["all_obj_rot6d"],
            "obj_trans": pt_preds["all_obj_trans"],
            "mano_3d_mesh_master": pt_preds["pred_mano_mesh_xyz_master"],
            "mano_3d_mesh_kp_master": pt_preds["pred_kp_mano_mesh_xyz_master"],
            "mano_3d_mesh_mesh_master": pt_preds.get("pred_mesh_mano_mesh_xyz_master", None),
            "pred_pose": pt_preds["pred_mano_params_master"]['pose_euler'],
            "pred_shape": pt_preds["pred_mano_params_master"]['shape'],
            "pred_kp_pose": pt_preds["pred_kp_mano_params_master"]['pose_euler'],
            "pred_kp_shape": pt_preds["pred_kp_mano_params_master"]['shape'],
            "pred_mesh_pose": pt_preds.get("pred_mesh_mano_params_master", {}).get('pose_euler', None) if pt_preds.get("pred_mesh_mano_params_master", None) is not None else None,
            "pred_mesh_shape": pt_preds.get("pred_mesh_mano_params_master", {}).get('shape', None) if pt_preds.get("pred_mesh_mano_params_master", None) is not None else None,
            "pred_obj_trans_master": pt_preds["pred_obj_trans_master"],
            "interaction_mode": pt_preds.get("interaction_mode", interaction_mode),
            "all_hand_joints_xyz_master": pt_preds.get("all_hand_joints_xyz_master", None),
            # "ball_points": pt_preds["obj_xyz"],
        })

        return final_preds
    
    @staticmethod
    def loss_proj_to_multicam(pred_points, T_c2m, K, gt_joints_2d, n_views, img_scale):
        pred_points = pred_points.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)
        pred_points_in_cam = batch_cam_extr_transf(T_c2m, pred_points)
        pred_points_2d = batch_cam_intr_projection(K, pred_points_in_cam)  # (B, N, 21, 2)
        multicam_proj_offset = torch.clamp(pred_points_2d - gt_joints_2d, min=-.5 * img_scale,
                                           max=.5 * img_scale) / img_scale
        loss_2d_points = torch.sum(torch.pow(multicam_proj_offset, 2), dim=3)  # (B, N, 21, 2)
        loss_2d_points = torch.mean(loss_2d_points)
        return loss_2d_points

    @staticmethod
    def compute_object_pose_metrics(preds, gt):
        pred_obj_rot6d = preds["obj_rot6d"][-1].detach()
        pred_obj_trans = preds["obj_trans"][-1].detach()
        pred_obj_center = preds["obj_center_xyz_master"][-1].detach().squeeze(1)

        gt_obj_rot6d = gt["master_obj_rot6d_label"].detach()
        gt_obj_trans = gt["master_obj_t_label_rel"].detach()
        gt_obj_center = gt["master_obj_center"].detach().squeeze(1)

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
            "metric_obj_center_epe": torch.mean(torch.norm(pred_obj_center - gt_obj_center, dim=-1)),
        }
        pred_obj_points_pose = (
            torch.matmul(rot6d_to_rotmat(pred_obj_rot6d), gt["master_obj_sparse_rest"].detach().transpose(1, 2)).transpose(1, 2)
            + pred_obj_trans.unsqueeze(1)
        )
        gt_obj_points_pose = (
            torch.matmul(rot6d_to_rotmat(gt_obj_rot6d), gt["master_obj_sparse_rest"].detach().transpose(1, 2)).transpose(1, 2)
            + gt_obj_trans.unsqueeze(1)
        )
        metric_dict["metric_obj_add"] = torch.mean(torch.norm(pred_obj_points_pose - gt_obj_points_pose, dim=-1).mean(dim=-1))
        metric_dict["metric_obj_adds"] = torch.mean(torch.cdist(pred_obj_points_pose.float(), gt_obj_points_pose.float()).min(dim=-1)[0].mean(dim=-1))
        return metric_dict

    @staticmethod
    def rotation_geodesic(pred_rot6d, gt_rot6d):
        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return torch.acos(cos_theta)

    def compute_loss(self, preds, gt, stage_name="stage3", epoch_idx=None, **kwargs):
        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]  # (BN, 799, 3)
        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"]  # (BN, 799, 2)
        pred_mano_mesh_master = preds["mano_3d_mesh_master"]  # (B, 799, 3)
        pred_mano_mesh_kp_master = preds["mano_3d_mesh_kp_master"]  # (B, 799, 3)
        pred_mano_mesh_mesh_master = preds.get("mano_3d_mesh_mesh_master", None)
        pred_hand_mesh_master = preds["hand_mesh_xyz_master"]  # (N_preds, B, 799, 3)
        pred_obj_points_3d = preds["obj_xyz_master"]  # (N_preds, B, 2048, 3)
        pred_obj_rot6d = preds["obj_rot6d"]  # (N_preds, B, 6)
        pred_obj_trans = preds["obj_trans"]  # (N_preds, B, 3)
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
        joints_vis = gt.get("target_joints_vis", None)
        joints_vis = joints_vis.flatten(0, 1) if joints_vis is not None else None  # (B*N, 21)
        obj_center_vis = gt.get("target_obj_center_vis", None)
        obj_center_vis = obj_center_vis.flatten(0, 1) if obj_center_vis is not None else None  # (B*N, 1)
        enable_hand_refine = stage_name in ["stage2", "stage3"]
        enable_object_refine = stage_name == "stage3"
        obj_warmup = self._stage3_object_warmup(epoch_idx, stage_name) if epoch_idx is not None else float(enable_object_refine)

        loss_dict = {}

        # ====================================================================
        # 1. Base 2D & RLE Loss (早期视角的初步感知)
        # ====================================================================
        pred_jts_2d = pred_mano_2d_mesh_sv[:, self.num_hand_verts:] # (BN, 21, 2)
        gt_jts_2d = hand_joints_sv_gt[..., :2]                      # (BN, 21, 2)

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
        loss_rle_hand = self.criterion_rle(
            preds["hand_nf_loss"],
            preds["pred_hand"],
            preds["sigma_hand"],
            gt["target_joints_uvd"][..., :2].reshape(preds["pred_hand"].shape),
            vis=joints_vis,
        )
        if enable_object_refine and obj_center_vis is not None:
            loss_rle_obj = self.criterion_rle(
                preds["obj_nf_loss"],
                preds["pred_obj"],
                preds["sigma_obj"],
                gt["target_obj_center_uv"][..., :2].reshape(preds["pred_obj"].shape),
                vis=obj_center_vis,
            )
        else:
            loss_rle_obj = pred_jts_2d.new_tensor(0.0)
        
        # 权重缩放
        loss_rle_hand = loss_rle_hand / self.num_hand_joints * self.cfg.LOSS.RLE_HAND_N
        loss_rle_obj = loss_rle_obj / self.num_obj_joints * self.cfg.LOSS.RLE_OBJ_N * obj_warmup

        loss_dict.update({
            'loss_2d_sv': loss_hand_2d_sv,
            'loss_rle_hand': loss_rle_hand,
            'loss_rle_obj': loss_rle_obj,
        })

        # ====================================================================
        # 2. Triangulation Loss (多视角几何收敛先验)
        # ====================================================================
        loss_triang_hand = self.criterion_joints(preds["ref_hand"], master_hand_joints_gt) * self.triangulation_hand_weight
        if enable_object_refine:
            loss_triang_obj = self.criterion_joints(preds["ref_obj"], gt["master_obj_center"]) * self.triangulation_obj_weight * obj_warmup
        else:
            loss_triang_obj = loss_triang_hand.new_tensor(0.0)
        loss_triang_total = loss_triang_hand + loss_triang_obj
        
        loss_dict['loss_triang_hand'] = loss_triang_hand
        loss_dict['loss_triang_obj'] = loss_triang_obj
        loss_dict['loss_triang'] = loss_triang_total

        # ====================================================================
        # 3. MANO Prior & Consistency Constraints (人体工学与视角一致性)
        # ====================================================================
        if enable_hand_refine:
            pred_mano_3d_mesh_kp_mv = pred_mano_mesh_kp_master.unsqueeze(1).repeat(1, n_views, 1, 1)
            pred_mano_3d_mesh_kp_mv = batch_cam_extr_transf(gt_T_c2m, pred_mano_3d_mesh_kp_mv).flatten(0, 1).detach()
            pred_mano_center_kp_mv = pred_mano_3d_mesh_kp_mv[:, 778 + self.center_idx, :].unsqueeze(1)
            pred_mano_3d_mesh_kp_mv = pred_mano_3d_mesh_kp_mv - pred_mano_center_kp_mv
            loss_mano_consistency_kp = self.coord_loss(pred_mano_3d_mesh_sv, pred_mano_3d_mesh_kp_mv) * self.mano_consistency_weight

            if pred_mano_mesh_mesh_master is not None:
                pred_mano_3d_mesh_mesh_mv = pred_mano_mesh_mesh_master.unsqueeze(1).repeat(1, n_views, 1, 1)
                pred_mano_3d_mesh_mesh_mv = batch_cam_extr_transf(gt_T_c2m, pred_mano_3d_mesh_mesh_mv).flatten(0, 1).detach()
                pred_mano_center_mesh_mv = pred_mano_3d_mesh_mesh_mv[:, 778 + self.center_idx, :].unsqueeze(1)
                pred_mano_3d_mesh_mesh_mv = pred_mano_3d_mesh_mesh_mv - pred_mano_center_mesh_mv
                loss_mano_consistency_mesh = self.coord_loss(pred_mano_3d_mesh_sv, pred_mano_3d_mesh_mesh_mv) * self.mano_consistency_weight
            else:
                loss_mano_consistency_mesh = pred_mano_3d_mesh_sv.new_tensor(0.0)

            if stage_name == "stage2":
                loss_mano_consistency = loss_mano_consistency_kp
            elif stage_name == "stage3" and pred_mano_mesh_mesh_master is not None:
                loss_mano_consistency = loss_mano_consistency_mesh
            else:
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
            if pred_pose_mesh_master is not None and pred_shape_mesh_master is not None:
                zero_target_mesh_master = torch.zeros_like(pred_pose_mesh_master[:, 3:])
                loss_pose_reg_mesh_master = self.coord_loss(pred_pose_mesh_master[:, 3:], zero_target_mesh_master) * self.cfg.LOSS.POSE_N
                loss_shape_reg_mesh_master = self.coord_loss(pred_shape_mesh_master, torch.zeros_like(pred_shape_mesh_master)) * self.cfg.LOSS.SHAPE_N
            else:
                loss_pose_reg_mesh_master = pred_pose_sv.new_tensor(0.0)
                loss_shape_reg_mesh_master = pred_shape_sv.new_tensor(0.0)
            loss_pose_reg_master = loss_pose_reg_kp_master + loss_pose_reg_mesh_master
            loss_shape_reg_master = loss_shape_reg_kp_master + loss_shape_reg_mesh_master
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
            pred_hand_joints_master_i = pred_hand_mesh_master_i[:, self.num_hand_verts:]  # (B, 21, 3)
            pred_obj_points_i = pred_obj_points_3d[i]  # (B, 2048, 3)
            pred_obj_rot6d_i = pred_obj_rot6d[i]  # (B, 6)
            pred_obj_trans_i = pred_obj_trans[i]  # (B, 3)

            if enable_hand_refine:
                loss_3d_joints = self.criterion_joints(pred_hand_joints_master_i, master_hand_joints_gt) * self.decoder_hand_weight
                loss_2d_joints = self.loss_proj_to_multicam(pred_hand_joints_master_i, gt_T_c2m, gt_K, hand_joints_2d_gt, n_views, img_scale) * self.decoder_proj_weight
            else:
                zero = pred_hand_mesh_master_i.new_tensor(0.0)
                loss_3d_joints = zero
                loss_2d_joints = zero

            if enable_object_refine:
                loss_obj_rot_geo = torch.mean(self.rotation_geodesic(pred_obj_rot6d_i, master_obj_rot6d_gt)) * self.obj_pose_rot_weight * obj_warmup
                loss_obj_rot_l1 = self.coord_loss(pred_obj_rot6d_i, master_obj_rot6d_gt) * self.obj_pose_rot6d_weight * obj_warmup
                loss_obj_rot = loss_obj_rot_geo + loss_obj_rot_l1
                loss_obj_trans = self.coord_loss(pred_obj_trans_i, master_obj_trans_gt) * self.obj_pose_trans_weight * obj_warmup
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
                    if pred_mano_mesh_mesh_master is not None:
                        pred_mano_joints_mesh_master = pred_mano_mesh_mesh_master[:, self.num_hand_verts:]
                        loss_mano_proj_mesh = self.criterion_joints(pred_mano_joints_mesh_master, master_hand_joints_gt) * self.mano_proj_weight
                    else:
                        loss_mano_proj_mesh = pred_mano_joints_kp_master.new_tensor(0.0)
                    loss_mano_proj = loss_mano_proj_kp + loss_mano_proj_mesh
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
            + loss_rle_hand
            + loss_rle_obj
            + loss_triang_total
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
        self.current_stage3_warmup = self._stage3_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._maybe_log_stage_transition("train", epoch_idx, stage_name, preds, batch)
        
        # GT 数据
        joints_2d_gt = batch["target_joints_uvd"][:, :, :, :2]  # (B, N, 21, 2)
        verts_2d_gt = batch["target_verts_uvd"][:, :, :, :2]  # (B, N, 778, 2)
        obj_2d_center_gt = batch["target_obj_center_uv"]  # (B, N, 1, 2)
        joints_3d_master_gt = batch["master_joints_3d"]  # (B, 21, 3)
        verts_3d_master_gt = batch["master_verts_3d"]    # (B, 778, 3)
        obj_sparse_3d_master_gt = batch["master_obj_sparse"]  # (B, 2048, 3)
        obj_center_3d_master_gt = batch["master_obj_center"]  # (B, 1, 3)
        verts_3d_gt = batch["target_verts_3d"]  # (B, N, 778, 3)
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)  # (B*N, 21, 3)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)    # (B*N, 778, 3)
        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])
        
        # 为了兼容性，使用 .get() 
        obj_pc_sparse = batch.get('target_obj_pc_sparse', None)
        obj_pc_center = batch.get('target_obj_center_3d', None)
        
        # 计算损失
        loss, loss_dict = self.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        pose_metric_dict = self.compute_object_pose_metrics(preds, batch) if stage_name == "stage3" else self._zero_metric_dict(img.device)
        
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
        pred_ref_obj_joints_3d = preds["ref_obj"]    # (B, 1, 3)
        pred_obj_sparse_3d = preds["obj_xyz_master"][-1]    # (B, 2048, 3)
        pred_obj_center_3d = preds["obj_center_xyz_master"][-1]  # (B, 1, 3)
        pred_ref_joints_2d = preds["pred_hand"].view(batch_size, n_views, self.num_hand_joints, 2)  # (B, N, 21, 2)
        pred_obj_center_2d = preds["pred_obj"].view(batch_size, n_views, 1, 2)  # (B, N, 1, 2)
        
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
        self.MPRPE_OBJ_3D.feed(pred_ref_obj_joints_3d, gt_kp=obj_center_3d_master_gt)
        self.loss_metric.feed({**loss_dict, **pose_metric_dict}, batch_size)
        self._maybe_log_stage_transition_loss("train", epoch_idx, stage_name, loss_dict, pose_metric_dict)

        if step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)
            for k, v in pose_metric_dict.items():
                self.summary.add_scalar(k, v.item(), step_idx)
            self.summary.add_scalar("MPJPE_SV_3D", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D", self.MPVPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPJPE_MASTER_3D", self.MPJPE_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_MASTER_3D", self.MPVPE_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPJPE_KP_MASTER_3D", self.MPJPE_KP_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_KP_MASTER_3D", self.MPVPE_KP_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPJPE_MESH_MASTER_3D", self.MPJPE_MESH_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_MESH_MASTER_3D", self.MPVPE_MESH_MASTER_3D.get_result(), step_idx)
            self.summary.add_scalar("MPRPE_HAND_3D", self.MPRPE_HAND_3D.get_result(), step_idx)
            self.summary.add_scalar("MPRPE_OBJ_3D", self.MPRPE_OBJ_3D.get_result(), step_idx)
            if step_idx % (self.train_log_interval * 10) == 0:
                with torch.no_grad():
                    self._log_visualizations("train", batch, preds, step_idx, stage_name)
                
        return None, loss_dict
    
    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([
            self.MPJPE_SV_3D, self.MPVPE_SV_3D,
            self.MPJPE_MASTER_3D, self.MPVPE_MASTER_3D,
            self.MPJPE_KP_MASTER_3D, self.MPVPE_KP_MASTER_3D,
            self.MPJPE_MESH_MASTER_3D, self.MPVPE_MESH_MASTER_3D,
            self.MPRPE_HAND_3D, self.MPRPE_OBJ_3D
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
        self.MPRPE_OBJ_3D.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        img = batch["image"]
        batch_size = img.size(0)
        n_views = img.size(1)

        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        interaction_mode = self._stage_to_interaction_mode(stage_name)
        self.current_stage_name = stage_name
        self.current_stage3_warmup = self._stage3_object_warmup(epoch_idx, stage_name)
        preds = self._forward_impl(batch, interaction_mode=interaction_mode)
        self._maybe_log_stage_transition("val", epoch_idx, stage_name, preds, batch)
        pose_metric_dict = self.compute_object_pose_metrics(preds, batch) if stage_name == "stage3" else self._zero_metric_dict(img.device)
        
        # GT 数据
        joints_2d_gt = batch["target_joints_uvd"][:, :, :, :2]  # (B, N, 21, 2)
        verts_2d_gt = batch["target_verts_uvd"][:, :, :, :2]  # (B, N, 778, 2)
        obj_2d_center_gt = batch["target_obj_center_uv"]  # (B, N, 1, 2)
        joints_3d_master_gt = batch["master_joints_3d"]  # (B, 21, 3)
        verts_3d_master_gt = batch["master_verts_3d"]    # (B, 778, 3)
        obj_sparse_3d_master_gt = batch["master_obj_sparse"]  # (B, 2048, 3)
        obj_center_3d_master_gt = batch["master_obj_center"]  # (B, 1, 3)
        joints_3d_gt = batch["target_joints_3d"].flatten(0, 1)  # (B*N, 21, 3)
        verts_3d_gt = batch["target_verts_3d"].flatten(0, 1)  # (B*N, 778, 3)
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)  # (B*N, 21, 3)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)    # (B*N, 778, 3)
        K_gt = batch["target_cam_intr"]
        T_c2m_gt = torch.linalg.inv(batch["target_cam_extr"])
        
        # 为了兼容性，使用 .get() 
        obj_pc_sparse = batch.get('target_obj_pc_sparse', None)
        obj_pc_center = batch.get('target_obj_center_3d', None)
        
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
        pred_ref_obj_joints_3d = preds["ref_obj"]    # (B, 1, 3)
        pred_obj_sparse_3d = preds["obj_xyz_master"][-1]    # (B, 2048, 3)
        pred_obj_center_3d = preds["obj_center_xyz_master"][-1]  # (B, 1, 3)
        pred_ref_joints_2d = preds["pred_hand"].view(batch_size, n_views, self.num_hand_joints, 2)  # (B, N, 21, 2)
        pred_obj_center_2d = preds["pred_obj"].view(batch_size, n_views, 1, 2)  # (B, N, 1, 2)

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
        self.MPRPE_OBJ_3D.feed(pred_ref_obj_joints_3d, gt_kp=obj_center_3d_master_gt)
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

        self.summary.add_scalar("MPJPE_SV_3D_val", self.MPJPE_SV_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_SV_3D_val", self.MPVPE_SV_3D.get_result(), step_idx)
        self.summary.add_scalar("MPJPE_MASTER_3D_val", self.MPJPE_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_MASTER_3D_val", self.MPVPE_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPJPE_KP_MASTER_3D_val", self.MPJPE_KP_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_KP_MASTER_3D_val", self.MPVPE_KP_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPJPE_MESH_MASTER_3D_val", self.MPJPE_MESH_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_MESH_MASTER_3D_val", self.MPVPE_MESH_MASTER_3D.get_result(), step_idx)
        self.summary.add_scalar("MPRPE_HAND_3D_val", self.MPRPE_HAND_3D.get_result(), step_idx)
        self.summary.add_scalar("MPRPE_OBJ_3D_val", self.MPRPE_OBJ_3D.get_result(), step_idx)
        self.summary.add_scalar("PA_SV_val", self.PA_SV.get_result(), step_idx)
        self.summary.add_scalar("PA_KP_MASTER_val", self.PA_KP_MASTER.get_result(), step_idx)
        self.summary.add_scalar("PA_MESH_MASTER_val", self.PA_MESH_MASTER.get_result(), step_idx)
        if stage_name == "stage3":
            self.OBJ_POSE_VAL.feed(
                pred_rot6d=preds["obj_rot6d"][-1],
                pred_trans=preds["obj_trans"][-1],
                pred_center=preds["obj_center_xyz_master"][-1],
                gt_rot6d=batch["master_obj_rot6d_label"],
                gt_trans=batch["master_obj_t_label_rel"],
                gt_center=batch["master_obj_center"],
                obj_points_rest=batch["master_obj_sparse_rest"],
            )
            obj_measures = self.OBJ_POSE_VAL.get_measures()
            self.summary.add_scalar("OBJ_ADD_val", obj_measures["Obj_add"], step_idx)
            self.summary.add_scalar("OBJ_ADDS_val", obj_measures["Obj_adds"], step_idx)
            self.summary.add_scalar("OBJ_CENTER_EPE_val", obj_measures["Obj_center_epe"], step_idx)
            self.summary.add_scalar("OBJ_TRANS_EPE_val", obj_measures["Obj_trans_epe"], step_idx)
            self.summary.add_scalar("OBJ_ROT_DEG_val", obj_measures["Obj_rot_deg"], step_idx)
        for k, v in pose_metric_dict.items():
            self.summary.add_scalar(f"{k}_val", v.item(), step_idx)
        self._maybe_log_stage_transition_loss("val", epoch_idx, stage_name, {}, pose_metric_dict)

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
            self.MPRPE_HAND_3D, self.MPRPE_OBJ_3D,
            self.PA_SV, self.PA_KP_MASTER, self.PA_MESH_MASTER,
            self.OBJ_POSE_VAL,
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
        self.MPRPE_OBJ_3D.reset()
        self.PA_SV.reset()
        self.PA_KP_MASTER.reset()
        self.PA_MESH_MASTER.reset()
        self.OBJ_POSE_VAL.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        """
        [ModuleAbstract 要求] 测试步 (通常直接复用 validation_step)
        """
        return self.validation_step(batch, step_idx, **kwargs)
    
    def draw_step(self, batch, step_idx, **kwargs):
        pred = self._forward_impl(batch)
        B, N, _, _, _ = batch["image"].size()
        img = batch["image"][0]  # (N, 3, H, W)  4 channels
        batch_joints_2d = batch['target_joints_uvd'][0]
        batch_center_2d = batch['target_obj_center_uv'][0]
        batch_verts_3d = batch["target_verts_3d"][0]
        batch_intr = batch["target_cam_intr"][0]
        # 尝试提取稀疏和密集点云 (为了兼容性，使用 .get() 和抛异常机制)
        batch_master_verts = batch["master_verts_3d"]
        batch_master_obj_center = batch["master_obj_center"]
        try:
            obj_pc_sparse = batch['target_obj_pc_sparse'][0]
            obj_pc_center = batch['target_obj_center_3d'][0]
        except KeyError:
            obj_pc_sparse = None
            obj_pc_center = None

        gt_T_c2m = torch.linalg.inv(batch["target_cam_extr"])
        gt_K = batch["target_cam_intr"]
        pred_hand_joints_3d = pred["mano_3d_mesh_master"][:, self.num_hand_verts:]  # (B, 21, 3)
        pred_hand_verts_3d = pred["mano_3d_mesh_master"][:, :self.num_hand_verts]  # (B, 778, 3)
        pred_obj_sparse_3d = pred["obj_xyz_master"][-1]
        pred_obj_center_3d = pred["obj_center_xyz_master"][-1]  # (B, 1, 3)
        # ball_points = pred["ball_points"]
        pred_joints = pred_hand_joints_3d.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, 21, 3)
        pred_joints_in_cam = batch_cam_extr_transf(gt_T_c2m, pred_joints)
        pred_joints_2d = batch_cam_intr_projection(gt_K, pred_joints_in_cam)  # (B, N, 21, 2)
        pred_verts = pred_hand_verts_3d.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, 778, 3)
        pred_verts_in_cam = batch_cam_extr_transf(gt_T_c2m, pred_verts)
        pred_obj = pred_obj_sparse_3d.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, 2048, 3)
        pred_obj_in_cam = batch_cam_extr_transf(gt_T_c2m, pred_obj)
        pred_obj_center = pred_obj_center_3d.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, 1, 3)
        pred_obj_center_in_cam = batch_cam_extr_transf(gt_T_c2m, pred_obj_center)
        # bp = ball_points.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, num_ball_points, 3)
        # bp_in_cam = batch_cam_extr_transf(gt_T_c2m, bp)

        pred_ref_joints_2d = pred['pred_hand'].view(B, N, 21, 2)[0]
        pred_ref_center_2d = pred['pred_obj'].view(B, N, 1, 2)[0]
        pred_hand_joints_2d = pred_joints_2d[0] / 256  # 归一化到0~1
        pred_hand_verts_3d = pred_verts_in_cam[0]
        pred_obj_center_3d = pred_obj_center_in_cam[0]
        pred_obj_sparse_3d = pred_obj_in_cam[0]
        # ball_points_3d = bp_in_cam[0]

        face = self.face
        # draw_list = draw_batch_mesh_images_gt(batch_verts_3d, face, obj_pc_sparse, obj_pc_center, batch_intr, img)
        joints_ref_draw_list = draw_batch_joint_center_images(pred_ref_joints_2d, batch_joints_2d, pred_ref_center_2d, batch_center_2d, img)
        joints_ref_row1 = cv2.hconcat(joints_ref_draw_list[:4])  # 前4张拼成第一行
        joints_ref_row2 = cv2.hconcat(joints_ref_draw_list[4:])  # 后4张拼成第二行

        joints_draw_list = draw_batch_joint_center_images(pred_hand_joints_2d, batch_joints_2d, pred_ref_center_2d, batch_center_2d, img)
        joints_row1 = cv2.hconcat(joints_draw_list[:4])  # 前4张拼成第一行
        joints_row2 = cv2.hconcat(joints_draw_list[4:])  # 后4张拼成第二行

        mesh_draw_list = draw_batch_mesh_images_pred(batch_verts_3d, pred_hand_verts_3d, face, obj_pc_sparse,
                                                     pred_obj_sparse_3d, obj_pc_center, pred_obj_center_3d, batch_intr, img)
        mesh_row1 = cv2.hconcat(mesh_draw_list[:4])  # 前4张拼成第一行
        mesh_row2 = cv2.hconcat(mesh_draw_list[4:])  # 后

        # 3. 垂直拼接：把两行拼成一张大图
        joints_big_image_ref = cv2.vconcat([joints_ref_row1, joints_ref_row2])
        cv2.imwrite('stitched_ref_image.jpg', cv2.cvtColor(joints_big_image_ref, cv2.COLOR_RGB2BGR))
        print("参考 joints图片已保存为 stitched_ref_image.jpg")

        joints_big_image = cv2.vconcat([joints_row1, joints_row2])
        cv2.imwrite('stitched_image.jpg', cv2.cvtColor(joints_big_image, cv2.COLOR_RGB2BGR))
        print("joints图片已保存为 stitched_joints_image.jpg")

        mesh_big_image = cv2.vconcat([mesh_row1, mesh_row2])
        cv2.imwrite('stitched_mesh_image.jpg', cv2.cvtColor(mesh_big_image, cv2.COLOR_RGB2BGR))
        print("mesh图片已保存为 stitched_mesh_image.jpg")

        return mesh_draw_list

    def get_metric(self, preds, batch, **kwargs):
        """
        [ModuleAbstract 预留] 离线指标计算
        """
        pass

    def format_metric(self, mode="train"):
        if mode == "train":
            l_rle = self.loss_metric.get_loss('loss_rle_hand') + self.loss_metric.get_loss('loss_rle_obj')
            l_obj_pose = self.loss_metric.get_loss('loss_obj_pose')
            m_obj_rot_deg = self.loss_metric.get_loss('metric_obj_rot_deg')
            m_obj_trans = self.loss_metric.get_loss('metric_obj_trans_epe')
            l_total = self.loss_metric.get_loss('loss')
            l_triang = self.loss_metric.get_loss('loss_triang')
            l_3d_jts = self.loss_metric.get_loss('loss_3d_jts')
            l_2d_proj = self.loss_metric.get_loss('loss_2d_proj')

            stage_short = self.current_stage_name.replace("stage", "S")
            warmup_suffix = f" W{self.current_stage3_warmup:.2f}" if self.current_stage_name == "stage3" and self.current_stage3_warmup < 1.0 else ""
            if self.current_stage_name == "stage1":
                return (f"{stage_short}{warmup_suffix} | "
                        f"Tot {l_total:.3f} | "
                        f"RLE {l_rle:.3f} | "
                        f"Tri {l_triang:.3f}")
            if self.current_stage_name == "stage2":
                kp_j = self.MPJPE_KP_MASTER_3D.get_result()
                kp_v = self.MPVPE_KP_MASTER_3D.get_result()
                return (f"{stage_short}{warmup_suffix} | "
                        f"Tot {l_total:.3f} | "
                        f"Tri {l_triang:.3f} | "
                        f"3DJ {l_3d_jts:.3f} | "
                        f"2DP {l_2d_proj:.3f} | "
                        f"KPJ {kp_j:.3f} | "
                        f"KPV {kp_v:.3f}")
            kp_j = self.MPJPE_KP_MASTER_3D.get_result()
            kp_v = self.MPVPE_KP_MASTER_3D.get_result()
            mesh_j = self.MPJPE_MESH_MASTER_3D.get_result()
            mesh_v = self.MPVPE_MESH_MASTER_3D.get_result()
            return (f"{stage_short}{warmup_suffix} | "
                    f"Tot {l_total:.3f} | "
                    f"Tri {l_triang:.3f} | "
                    f"3DJ {l_3d_jts:.3f} | "
                    f"KPJ {kp_j:.3f} | "
                    f"KPV {kp_v:.3f} | "
                    f"MJ {mesh_j:.3f} | "
                    f"MV {mesh_v:.3f} | "
                    f"Obj {l_obj_pose:.3f} | "
                    f"Rot {m_obj_rot_deg:.1f} | "
                    f"Tr {m_obj_trans:.3f}")
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
            ref_o = self.MPRPE_OBJ_3D.get_result()

            def _fmt_triplet(name, pa_dict, mp, mv):
                if mp <= 0 and mv <= 0:
                    return f"{name} PA/MP/MV -/-/- mm"
                return (
                    f"{name} PAJ/PAV {pa_dict['pa_mpjpe'] * 1000.0:.1f}/{pa_dict['pa_mpvpe'] * 1000.0:.1f} "
                    f"MP/MV {mp * 1000.0:.1f}/{mv * 1000.0:.1f} mm"
                )

            def _fmt_pa_only(name, pa_dict, mp, mv):
                if mp <= 0 and mv <= 0:
                    return f"{name} -/-"
                return f"{name} {pa_dict['pa_mpjpe'] * 1000.0:.1f}/{pa_dict['pa_mpvpe'] * 1000.0:.1f}"

            def _fmt_triplet_full(name, pa_dict, mp, mv):
                if mp <= 0 and mv <= 0:
                    return f"{name} PA-MPJPE/PA-MPVPE/MPJPE/MPVPE -/-/-/- mm"
                return (
                    f"{name} PA-MPJPE/PA-MPVPE {pa_dict['pa_mpjpe'] * 1000.0:.1f}/{pa_dict['pa_mpvpe'] * 1000.0:.1f} "
                    f"MPJPE/MPVPE {mp * 1000.0:.1f}/{mv * 1000.0:.1f} mm"
                )

            if mode == "val_full":
                msg = " | ".join([
                    _fmt_triplet_full("SingleView", pa_sv, sv_j, sv_v),
                    _fmt_triplet_full("KeypointMaster", pa_kp, kp_j, kp_v),
                    _fmt_triplet_full("MeshMaster", pa_mesh, mesh_j, mesh_v),
                    (
                        f"Object ADD/ADD-S {self.OBJ_POSE_VAL.add.avg * 1000.0:.1f}/{self.OBJ_POSE_VAL.adds.avg * 1000.0:.1f} "
                        f"CenterEPE/TranslationEPE {self.OBJ_POSE_VAL.center_epe.avg * 1000.0:.1f}/{self.OBJ_POSE_VAL.trans_epe.avg * 1000.0:.1f} mm "
                        f"RotationError {self.OBJ_POSE_VAL.rot_deg.avg:.1f} deg"
                        if self.current_stage_name == "stage3" and self.OBJ_POSE_VAL.count > 0 else
                        "Object ADD/ADD-S/CenterEPE/TranslationEPE/RotationError -/-/-/-/-"
                    ),
                    f"Reference Hand/Object {ref_j * 1000.0:.1f}/{ref_o * 1000.0:.1f} mm",
                ])
                return msg

            msg = " | ".join([
                f"PA { _fmt_pa_only('SV', pa_sv, sv_j, sv_v) } { _fmt_pa_only('KP', pa_kp, kp_j, kp_v) } { _fmt_pa_only('Mesh', pa_mesh, mesh_j, mesh_v) } mm",
                (
                    f"Obj A/S {self.OBJ_POSE_VAL.add.avg * 1000.0:.1f}/{self.OBJ_POSE_VAL.adds.avg * 1000.0:.1f}"
                    if self.current_stage_name == "stage3" and self.OBJ_POSE_VAL.count > 0 else
                    "Obj A/S -/-"
                ),
                f"Ref {ref_j * 1000.0:.1f}/{ref_o * 1000.0:.1f}",
            ])
            if "mm" not in msg:
                msg = f"{msg} mm"
            return msg

        return " | ".join([str(me) for me in metric_toshow])
