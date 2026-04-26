from itertools import combinations, product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manotorch.manolayer import MANOOutput, ManoLayer

from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import CONST
from ..utils.transform import batch_cam_intr_projection, batch_persp_project, rot6d_to_rotmat
from ..viztools.draw import (
    draw_batch_hand_mesh_images_2d,
    draw_batch_joint_images,
    draw_batch_object_point_cloud_images,
    tile_batch_images,
)
from .HOR_heatmap_centerrot_slim import POEM_HeatmapCenterRotSlim


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


class _HOPRegTransHead(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int = 9):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(inp_dim, inp_dim // 2),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(inp_dim // 2, out_dim)

    def forward(self, x):
        return self.final_layer(self.decoder(x))


class _HOPRegManoBranch(nn.Module):
    def __init__(self, inp_dim: int, center_idx: int, cfg):
        super().__init__()
        self.center_idx = center_idx
        self.inp_dim = inp_dim
        self.ncomps = int(cfg.get("NCOMPS", 15))
        self.use_pca = bool(cfg.get("USE_PCA", True))
        self.use_shape = bool(cfg.get("USE_SHAPE", True))
        self.mano_assets_root = str(cfg.get("MANO_ASSETS_ROOT", "assets/mano_v1_2"))
        self.mano_side = str(cfg.get("SIDE", CONST.SIDE))
        self.pose_abs_max = 10.0
        self.shape_abs_max = 10.0
        self.coord_abs_max = 2.0

        base_neurons = [self.inp_dim, 512, 512]
        base_layers = []
        for in_neurons, out_neurons in zip(base_neurons[:-1], base_neurons[1:]):
            base_layers.append(nn.Linear(in_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        if self.use_pca:
            mano_pose_size = self.ncomps + 3
        else:
            mano_pose_size = 16 * 9
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        if not self.use_pca:
            self.pose_reg.bias.data.fill_(0)
            weight_mask = self.pose_reg.weight.data.new(np.identity(3)).view(9).repeat(16)
            self.pose_reg.weight.data = torch.abs(
                weight_mask.unsqueeze(1).repeat(1, base_neurons[-1]).float() * self.pose_reg.weight.data
            )

        if self.use_shape:
            self.shape_reg = nn.Linear(base_neurons[-1], 10)
        else:
            self.shape_reg = None

        self.mano_layer = ManoLayer(
            ncomps=self.ncomps,
            center_idx=center_idx,
            side=self.mano_side,
            mano_assets_root=self.mano_assets_root,
            use_pca=self.use_pca,
            flat_hand_mean=False,
        )
        self.faces = getattr(self.mano_layer, "th_faces", None)
        if self.faces is None:
            self.faces = self.mano_layer.get_mano_closed_faces()

    def forward(self, feat):
        feat = self.base_layer(feat)
        pose = torch.nan_to_num(self.pose_reg(feat), nan=0.0, posinf=self.pose_abs_max, neginf=-self.pose_abs_max)
        if self.use_shape:
            shape = torch.nan_to_num(
                self.shape_reg(feat),
                nan=0.0,
                posinf=self.shape_abs_max,
                neginf=-self.shape_abs_max,
            )
        else:
            shape = feat.new_zeros(feat.shape[0], 10)

        mano_out: MANOOutput = self.mano_layer(pose, shape)
        joints_3d = torch.nan_to_num(mano_out.joints, nan=0.0, posinf=self.coord_abs_max, neginf=-self.coord_abs_max)
        hand_verts_3d = torch.nan_to_num(mano_out.verts, nan=0.0, posinf=self.coord_abs_max, neginf=-self.coord_abs_max)
        mano_full_pose = torch.nan_to_num(
            mano_out.full_poses,
            nan=0.0,
            posinf=np.pi,
            neginf=-np.pi,
        )

        return {
            "joints_3d": joints_3d,
            "hand_verts_3d": hand_verts_3d,
            "mano_shape": shape,
            "mano_pca_pose": pose,
            "mano_full_pose": mano_full_pose,
            "mano_pose_aa": mano_full_pose,
        }


@MODEL.register_module()
class POEM_SV_HOPRegNet(POEM_HeatmapCenterRotSlim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.hopreg_hidden_dim = int(cfg.get("HOPREG_HIDDEN_DIM", self.feat_size[0]))
        hopreg_mano_cfg = cfg.get("HOPREG_MANO", {})
        self.hopreg_hand_branch = _HOPRegManoBranch(self.feat_size[0], self.center_idx, hopreg_mano_cfg)
        self.hopreg_obj_transfhead = _HOPRegTransHead(self.feat_size[0], out_dim=9)

        self.arti_mano_weight = float(cfg.LOSS.get("ARTI_MANO_N", 1.0))
        self.arti_joints_group_weight = float(cfg.LOSS.get("ARTI_JOINTS_GROUP_N", 1.0))
        self.arti_mano_joints_3d_weight = float(cfg.LOSS.get("ARTI_MANO_JOINTS_3D_N", 0.0))
        self.arti_mano_verts_3d_weight = float(cfg.LOSS.get("ARTI_MANO_VERTS_3D_N", 0.0))
        self.arti_joints_3d_weight = float(cfg.LOSS.get("ARTI_JOINTS_3D_N", 1.0))
        self.arti_verts_3d_weight = float(cfg.LOSS.get("ARTI_VERTS_3D_N", 0.0))
        self.arti_corners_3d_weight = float(cfg.LOSS.get("ARTI_CORNERS_3D_N", 0.2))
        self.arti_hand_ord_weight = float(cfg.LOSS.get("ARTI_HAND_ORD_N", 0.1))
        self.arti_scene_ord_weight = float(cfg.LOSS.get("ARTI_SCENE_ORD_N", 0.1))
        self.arti_hand_ord_joint_weight = float(cfg.LOSS.get("ARTI_HAND_ORD_JOINT_N", 1.0))
        self.arti_hand_ord_part_weight = float(cfg.LOSS.get("ARTI_HAND_ORD_PART_N", 1.0))
        self.arti_hand_n_virtual_views = int(
            cfg.LOSS.get("ARTI_HAND_N_VIRTUAL_VIEWS", cfg.LOSS.get("ARTI_N_VIRTUAL_VIEWS", 20))
        )
        self.arti_scene_n_virtual_views = int(
            cfg.LOSS.get("ARTI_SCENE_N_VIRTUAL_VIEWS", cfg.LOSS.get("ARTI_N_VIRTUAL_VIEWS", 40))
        )
        self.arti_pose_reg_weight = float(cfg.LOSS.get("ARTI_MANO_POSE_REG_N", cfg.LOSS.get("POSE_N", 5e-6)))
        self.arti_shape_reg_weight = float(cfg.LOSS.get("ARTI_MANO_SHAPE_REG_N", cfg.LOSS.get("SHAPE_N", 5e-7)))

        self._arti_joint_pairs_idx = list(combinations(range(self.num_hand_joints), 2))
        self._arti_part_pairs_idx = list(combinations(range(self.num_hand_joints - 1), 2))
        self._arti_ho_pairs_idx = list(product(range(self.num_hand_joints), range(8)))

        for name in [
            "project",
            "ptEmb_head",
            "feat_delayer",
            "feat_in",
            "hot_pose",
            "att_0",
            "att_1",
            "mano_fuse",
            "mano_fc",
            "sv_object_pose_feat",
            "sv_object_token_adapter",
            "sv_hand_token_adapter",
            "sv_object_hand_cross_attn",
            "sv_object_pose_norm",
            "sv_object_pose_ffn",
            "sv_object_rot_regressor",
            "sv_object_trans_regressor",
            "sv_object_init_pose_token",
            "sv_object_init_pose_head",
            "uv_delayer",
            "uv_out",
            "slim_hand_feat",
            "slim_hand_pose_head",
            "slim_hand_shape_head",
            "slim_hand_cam_head",
            "slim_obj_feat",
            "slim_obj_center_head",
            "slim_obj_rot_head",
        ]:
            if hasattr(self, name):
                delattr(self, name)

        self.face = self.hopreg_hand_branch.faces
        active_params_m = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"{type(self).__name__} initialized with ArtiBoost-Reg style branches "
            f"(GT hand root, PCA-MANO regression, object rot6d + trans_rel)"
        )
        logger.info(
            f"{type(self).__name__} active parameter count after stripping unused POEM modules: "
            f"{active_params_m:.3f}M"
        )

    def _ensure_view_batch(self, batch):
        """Allow the HOPRegNet branch to consume true single-view batches."""
        img = batch["image"]
        if img.dim() == 5:
            return batch
        if img.dim() != 4:
            raise ValueError(f"Expected `image` to have 4 or 5 dims, got {tuple(img.shape)}")

        batch_size = img.shape[0]
        adapted = {}
        for key, value in batch.items():
            if (
                torch.is_tensor(value)
                and value.shape[:1] == (batch_size,)
                and not key.startswith("master_")
                and key not in {"master_id"}
            ):
                adapted[key] = value.unsqueeze(1)
            else:
                adapted[key] = value

        device = img.device
        dtype = img.dtype
        if "target_cam_extr" not in adapted:
            adapted["target_cam_extr"] = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
                batch_size, 1, 1, 1
            )
        if "master_id" not in adapted:
            adapted["master_id"] = torch.zeros(batch_size, device=device, dtype=torch.long)

        def _first_available(*keys):
            for source_key in keys:
                value = adapted.get(source_key, None)
                if torch.is_tensor(value):
                    if value.dim() >= 2 and value.shape[1] == 1:
                        return value[:, 0]
                    return value
            return None

        master_map = {
            "master_joints_3d": ("target_joints_3d_no_rot", "target_joints_3d"),
            "master_verts_3d": ("target_verts_3d_no_rot", "target_verts_3d"),
            "master_obj_kp21": ("target_obj_kp21_no_rot", "target_obj_kp21"),
            "master_obj_kp21_rest": ("target_obj_kp21_rest", "obj_kp21_rest"),
            "master_obj_sparse": ("target_obj_sparse_no_rot", "target_obj_pc_sparse", "obj_pc_sparse"),
            "master_obj_sparse_rest": ("target_obj_pc_sparse_rest", "obj_pc_sparse_rest"),
            "master_obj_eval_rest": ("target_obj_pc_eval_rest", "obj_pc_eval_rest"),
            "master_obj_rot6d_label": ("target_rot6d_label", "rot6d_label"),
            "master_obj_t_label_rel": ("target_t_label_rel", "t_label_rel"),
        }
        for target_key, source_keys in master_map.items():
            if target_key not in adapted:
                master_value = _first_available(*source_keys)
                if master_value is not None:
                    adapted[target_key] = master_value

        return adapted

    def set_train_stage(self, stage_name: str):
        requested_stage = stage_name
        stage_name = "stage1"
        self.current_stage_name = stage_name
        self.current_stage2_warmup = 0.0

        modules_to_enable = [
            getattr(self, "img_backbone", None),
            getattr(self, "hopreg_hand_branch", None),
            getattr(self, "hopreg_obj_transfhead", None),
        ]

        for module in modules_to_enable:
            self._set_module_requires_grad(module, True)

        if hasattr(self.img_backbone, "fc") and isinstance(self.img_backbone.fc, nn.Module):
            self._set_module_requires_grad(self.img_backbone.fc, False)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        if self.debug_logs:
            logger.warning(
                f"[StageFreeze][HOPRegNet] requested_stage={requested_stage} forced_stage={stage_name} "
                f"trainable={trainable_params / 1e6:.2f}M "
                f"frozen={frozen_params / 1e6:.2f}M"
            )

    def _resolve_stage(self, epoch_idx):
        return "stage1"

    @staticmethod
    def _stage_to_interaction_mode(stage_name):
        return "hand"

    def _feed_sv_object_recon(self, preds, batch, use_full_recon_eval=False):
        obj_eval_rest = batch.get("master_obj_eval_rest", None) if use_full_recon_eval else None
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
            return

        obj_pc_sparse = batch.get("target_obj_pc_sparse", None)
        if (
            obj_pc_sparse is not None
            and preds.get("obj_view_rot6d_cam", None) is not None
            and preds.get("obj_view_trans", None) is not None
        ):
            pred_sv_obj_sparse_3d = self._build_object_points_from_hand_pose(
                batch["master_obj_sparse_rest"],
                preds["obj_view_rot6d_cam"],
                batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1],
                preds["obj_view_trans"],
            )
            self.OBJ_RECON_SV.feed(pred_sv_obj_sparse_3d.flatten(0, 1), obj_pc_sparse.flatten(0, 1))

    def training_step(self, batch, step_idx, **kwargs):
        batch = self._ensure_view_batch(batch)
        img = batch["image"]
        batch_size = img.size(0)
        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = 0.0

        preds = self._forward_impl(batch, interaction_mode="hand")
        self._maybe_log_stage_transition("train", epoch_idx, stage_name, preds, batch)

        loss, loss_dict = self.compute_loss(preds, batch, stage_name=stage_name, epoch_idx=epoch_idx)
        sv_obj_metric_dict = self.compute_sv_object_pose_metrics(preds, batch)
        metrics_finite = self._metric_dict_is_finite(sv_obj_metric_dict)
        loss_finite = self._loss_dict_is_finite(loss_dict)
        if (not loss_finite) or (not metrics_finite):
            logger.warning(
                f"[TrainNonFiniteEarlyExit] epoch={epoch_idx} stage={stage_name} step={step_idx} "
                f"loss_finite={loss_finite} sv_metrics_finite={metrics_finite}"
            )
            self._maybe_log_stage_transition_loss("train", epoch_idx, stage_name, loss_dict, sv_obj_metric_dict)
            return None, loss_dict

        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]
        pred_mano_3d_joints_sv = pred_mano_3d_mesh_sv[:, self.num_hand_verts:]
        pred_mano_3d_verts_sv = pred_mano_3d_mesh_sv[:, :self.num_hand_verts]
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)

        self.MPJPE_SV_3D.feed(pred_mano_3d_joints_sv, gt_kp=joints_3d_rel_gt)
        self.MPVPE_SV_3D.feed(pred_mano_3d_verts_sv, gt_kp=verts_3d_rel_gt)
        self.PA_SV.feed(pred_mano_3d_joints_sv, joints_3d_rel_gt, pred_mano_3d_verts_sv, verts_3d_rel_gt)
        self._feed_sv_object_recon(preds, batch, use_full_recon_eval=False)

        metric_log_dict = dict(sv_obj_metric_dict)
        self.loss_metric.feed({**loss_dict, **metric_log_dict}, batch_size)
        self._maybe_log_stage_transition_loss("train", epoch_idx, stage_name, loss_dict, metric_log_dict)

        if self.summary is not None and step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(k, v.item(), step_idx)
            for k, v in metric_log_dict.items():
                self.summary.add_scalar(k, v.item(), step_idx)
            pa_sv = self.PA_SV.get_measures()
            self.summary.add_scalar("MPJPE_SV_3D", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D", self.MPVPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("PA_SV_J", pa_sv["pa_mpjpe"], step_idx)
            self.summary.add_scalar("PA_SV_V", pa_sv["pa_mpvpe"], step_idx)
            self.summary.add_scalar("OBJREC_SV_CD", self.OBJ_RECON_SV.cd.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_5", self.OBJ_RECON_SV.fs_5.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_10", self.OBJ_RECON_SV.fs_10.avg, step_idx)
            if step_idx % (self.train_log_interval * 10) == 0:
                with torch.no_grad():
                    self._log_visualizations("train", batch, preds, step_idx, stage_name)

        return None, loss_dict

    def on_train_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric(
            [self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.PA_SV, self.OBJ_RECON_SV],
            epoch_idx,
            comment=comment,
            summary=self.format_metric(mode="train"),
        )
        self.loss_metric.reset()
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.PA_SV.reset()
        self.OBJ_RECON_SV.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        batch = self._ensure_view_batch(batch)
        img = batch["image"]
        batch_size = img.size(0)
        epoch_idx = kwargs.get("epoch_idx", 0)
        stage_name = self._resolve_stage(epoch_idx)
        self.current_stage_name = stage_name
        self.current_stage2_warmup = 0.0
        use_full_recon_eval = self._use_full_val_recon(epoch_idx)
        self.current_val_recon_mode = "full" if use_full_recon_eval else "fast"

        preds = self._forward_impl(batch, interaction_mode="hand")
        self._maybe_log_stage_transition("val", epoch_idx, stage_name, preds, batch)

        sv_obj_metric_dict = self.compute_sv_object_pose_metrics(preds, batch)
        if not self._metric_dict_is_finite(sv_obj_metric_dict):
            logger.warning(
                f"[ValNonFiniteEarlyExit] epoch={epoch_idx} stage={stage_name} step={step_idx} "
                "sv_metrics_finite=False"
            )
            self._maybe_log_stage_transition_loss("val", epoch_idx, stage_name, {}, sv_obj_metric_dict)
            return None

        pred_mano_3d_mesh_sv = preds["mano_3d_mesh_sv"]
        pred_mano_3d_joints_sv = pred_mano_3d_mesh_sv[:, self.num_hand_verts:]
        pred_mano_3d_verts_sv = pred_mano_3d_mesh_sv[:, :self.num_hand_verts]
        joints_3d_rel_gt = batch["target_joints_3d_rel"].flatten(0, 1)
        verts_3d_rel_gt = batch["target_verts_3d_rel"].flatten(0, 1)

        self.MPJPE_SV_3D.feed(pred_mano_3d_joints_sv, gt_kp=joints_3d_rel_gt)
        self.MPVPE_SV_3D.feed(pred_mano_3d_verts_sv, gt_kp=verts_3d_rel_gt)
        self.PA_SV.feed(pred_mano_3d_joints_sv, joints_3d_rel_gt, pred_mano_3d_verts_sv, verts_3d_rel_gt)
        self._feed_sv_object_recon(preds, batch, use_full_recon_eval=use_full_recon_eval)

        metric_log_dict = dict(sv_obj_metric_dict)
        self.loss_metric.feed(metric_log_dict, batch_size)
        self._maybe_log_stage_transition_loss("val", epoch_idx, stage_name, {}, metric_log_dict)

        if self.summary is not None:
            pa_sv = self.PA_SV.get_measures()
            self.summary.add_scalar("MPJPE_SV_3D_val", self.MPJPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_SV_3D_val", self.MPVPE_SV_3D.get_result(), step_idx)
            self.summary.add_scalar("PA_SV_J_val", pa_sv["pa_mpjpe"], step_idx)
            self.summary.add_scalar("PA_SV_V_val", pa_sv["pa_mpvpe"], step_idx)
            self.summary.add_scalar("OBJREC_SV_CD_val", self.OBJ_RECON_SV.cd.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_5_val", self.OBJ_RECON_SV.fs_5.avg, step_idx)
            self.summary.add_scalar("OBJREC_SV_FS_10_val", self.OBJ_RECON_SV.fs_10.avg, step_idx)
            self.summary.add_scalar("VAL_RECON_IS_FULL_val", float(use_full_recon_eval), step_idx)
            for k, v in metric_log_dict.items():
                self.summary.add_scalar(f"{k}_val", v.item(), step_idx)
            if step_idx % (self.train_log_interval * 5) == 0:
                with torch.no_grad():
                    self._log_visualizations("val", batch, preds, step_idx, stage_name)

        return None

    def on_val_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_metric(
            [self.MPJPE_SV_3D, self.MPVPE_SV_3D, self.PA_SV, self.OBJ_RECON_SV],
            epoch_idx,
            comment=comment,
            summary=self.format_metric(mode="val_full"),
        )
        self.MPJPE_SV_3D.reset()
        self.MPVPE_SV_3D.reset()
        self.PA_SV.reset()
        self.OBJ_RECON_SV.reset()
        self.loss_metric.reset()

    def testing_step(self, batch, step_idx, **kwargs):
        return self.validation_step(batch, step_idx, **kwargs)

    def _compute_hand_ord_loss(self, pred_joints_abs, gt_joints_abs, joint_valid):
        pred_masked = pred_joints_abs * joint_valid.unsqueeze(-1).to(dtype=pred_joints_abs.dtype)
        gt_masked = gt_joints_abs * joint_valid.unsqueeze(-1).to(dtype=gt_joints_abs.dtype)
        view_vecs = _sample_view_vectors(
            self.arti_hand_n_virtual_views,
            device=pred_joints_abs.device,
            dtype=pred_joints_abs.dtype,
        )
        view_vecs = view_vecs.unsqueeze(0).expand(pred_joints_abs.shape[0], -1, -1)

        pred_jpairs = self._joints_to_joint_pairs(pred_masked)
        gt_jpairs = self._joints_to_joint_pairs(gt_masked)
        pred_ppairs = self._joints_to_part_pairs(pred_masked)
        gt_ppairs = self._joints_to_part_pairs(gt_masked)

        if pred_jpairs.shape[1] > 0:
            joint_take = max(1, pred_jpairs.shape[1] // 3)
            joint_idx = torch.randperm(pred_jpairs.shape[1], device=pred_joints_abs.device)[:joint_take]
            pred_jpairs = pred_jpairs[:, joint_idx, :]
            gt_jpairs = gt_jpairs[:, joint_idx, :]
            gt_joint_ord = _jointlevel_ordinal_relation(gt_jpairs, view_vecs)
            gt_joint_sign = torch.sign(gt_joint_ord)
            pred_joint_ord = _jointlevel_ordinal_relation(pred_jpairs, view_vecs)
            joint_ord_loss = torch.log1p(F.relu(-1.0 * gt_joint_sign * pred_joint_ord)).mean()
        else:
            joint_ord_loss = pred_joints_abs.new_tensor(0.0)

        if pred_ppairs.shape[1] > 0:
            part_take = max(1, pred_ppairs.shape[1] // 3)
            part_idx = torch.randperm(pred_ppairs.shape[1], device=pred_joints_abs.device)[:part_take]
            pred_ppairs = pred_ppairs[:, part_idx, :]
            gt_ppairs = gt_ppairs[:, part_idx, :]
            gt_part_ord = _partlevel_ordinal_relation(gt_ppairs, view_vecs)
            gt_part_sign = torch.sign(gt_part_ord)
            pred_part_ord = _partlevel_ordinal_relation(pred_ppairs, view_vecs)
            part_ord_loss = F.relu(-1.0 * gt_part_sign * pred_part_ord).mean()
        else:
            part_ord_loss = pred_joints_abs.new_tensor(0.0)

        total = (
            self.arti_hand_ord_joint_weight * joint_ord_loss
            + self.arti_hand_ord_part_weight * part_ord_loss
        )
        return total, joint_ord_loss, part_ord_loss

    def _compute_scene_ord_loss(self, pred_joints_abs, gt_joints_abs, joint_valid, pred_corners_abs, gt_corners_abs, corner_valid):
        pred_joints_masked = pred_joints_abs * joint_valid.unsqueeze(-1).to(dtype=pred_joints_abs.dtype)
        gt_joints_masked = gt_joints_abs * joint_valid.unsqueeze(-1).to(dtype=gt_joints_abs.dtype)
        pred_corners_masked = pred_corners_abs * corner_valid.unsqueeze(-1).to(dtype=pred_corners_abs.dtype)
        gt_corners_masked = gt_corners_abs * corner_valid.unsqueeze(-1).to(dtype=gt_corners_abs.dtype)

        pred_pairs = self._joints_and_corners_to_pairs(pred_joints_masked, pred_corners_masked)
        gt_pairs = self._joints_and_corners_to_pairs(gt_joints_masked, gt_corners_masked)
        if pred_pairs.shape[1] == 0:
            return pred_joints_abs.new_tensor(0.0)

        ho_take = max(1, pred_pairs.shape[1] // 3)
        ho_idx = torch.randperm(pred_pairs.shape[1], device=pred_joints_abs.device)[:ho_take]
        pred_pairs = pred_pairs[:, ho_idx, :]
        gt_pairs = gt_pairs[:, ho_idx, :]

        view_vecs = _sample_view_vectors(
            self.arti_scene_n_virtual_views,
            device=pred_joints_abs.device,
            dtype=pred_joints_abs.dtype,
        )
        view_vecs = view_vecs.unsqueeze(0).expand(pred_joints_abs.shape[0], -1, -1)

        gt_ord = _jointlevel_ordinal_relation(gt_pairs, view_vecs)
        gt_sign = torch.sign(gt_ord)
        pred_ord = _jointlevel_ordinal_relation(pred_pairs, view_vecs)
        return torch.log1p(F.relu(-1.0 * gt_sign * pred_ord)).mean()

    @staticmethod
    def _artiboost_zero_mask_mse(pred, gt, valid_mask):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=10.0, neginf=-10.0)
        valid = valid_mask.to(dtype=pred.dtype).unsqueeze(-1)
        return F.mse_loss(pred * valid, gt * valid)

    def _forward_impl(self, batch, **kwargs):
        batch = self._ensure_view_batch(batch)
        img = batch["image"]
        batch_size, num_cams = img.shape[:2]
        img_h, img_w = img.shape[-2:]

        _, _, global_feat = self.extract_img_feat(img)
        feat = global_feat.reshape(batch_size * num_cams, -1)
        dtype = feat.dtype
        device = feat.device

        mano_results = self.hopreg_hand_branch(feat)
        pred_joints_rel = mano_results["joints_3d"]
        pred_verts_rel = mano_results["hand_verts_3d"]
        pred_mesh_rel = torch.cat([pred_verts_rel, pred_joints_rel], dim=1)

        gt_root_joint = batch["target_joints_3d"][:, :, self.center_idx].to(device=device, dtype=dtype).reshape(batch_size * num_cams, 3)
        pred_joints_abs = pred_joints_rel + gt_root_joint.unsqueeze(1)
        pred_verts_abs = pred_verts_rel + gt_root_joint.unsqueeze(1)
        pred_mesh_abs = torch.cat([pred_verts_abs, pred_joints_abs], dim=1)

        cam_intr = batch["target_cam_intr"].to(device=device, dtype=dtype).reshape(batch_size * num_cams, 3, 3)
        pred_joints_2d = batch_persp_project(pred_joints_abs, cam_intr)
        pred_verts_2d = batch_persp_project(pred_verts_abs, cam_intr)
        pred_mesh_2d = torch.cat([pred_verts_2d, pred_joints_2d], dim=1)

        obj_pose = self.hopreg_obj_transfhead(feat)
        pred_obj_trans_rel = obj_pose[:, :3].view(batch_size, num_cams, 3)
        pred_obj_rot6d_cam = obj_pose[:, 3:].view(batch_size, num_cams, 6)
        pred_obj_center_abs = gt_root_joint.view(batch_size, num_cams, 3) + pred_obj_trans_rel

        corners_can = batch["master_obj_kp21_rest"][:, :8].to(device=device, dtype=dtype)
        corners_can_views = corners_can.unsqueeze(1).expand(-1, num_cams, -1, -1).reshape(batch_size * num_cams, 8, 3)
        pred_corners_abs = self._build_object_points_from_pose_grad(
            corners_can_views,
            pred_obj_rot6d_cam.reshape(batch_size * num_cams, 6),
            pred_obj_center_abs.reshape(batch_size * num_cams, 3),
        ).view(batch_size, num_cams, 8, 3)
        pred_center_2d = batch_persp_project(
            pred_obj_center_abs.reshape(batch_size * num_cams, self.num_obj_joints, 3),
            batch["target_cam_intr"].to(device=device, dtype=dtype).reshape(batch_size * num_cams, 3, 3),
        ).view(batch_size, num_cams, self.num_obj_joints, 2)

        pred_mesh_abs_views = pred_mesh_abs.view(batch_size, num_cams, self.num_hand_verts + self.num_hand_joints, 3)
        pred_mesh_rel_views = pred_mesh_rel.view(batch_size, num_cams, self.num_hand_verts + self.num_hand_joints, 3)
        master_ids = batch["master_id"].to(device=device, dtype=torch.long)
        pred_mesh_master = self._gather_master_view(pred_mesh_abs_views, master_ids)
        pred_ref_hand = pred_mesh_master[:, self.num_hand_verts:]
        pred_obj_center_master = self._gather_master_view(pred_obj_center_abs.unsqueeze(2), master_ids)
        pred_obj_rot6d_master = self._gather_master_view(pred_obj_rot6d_cam, master_ids)
        pred_obj_trans_master = self._gather_master_view(pred_obj_trans_rel, master_ids)

        master_obj_sparse_rest = batch["master_obj_sparse_rest"].to(device=device, dtype=dtype)
        pred_obj_sparse_master = self._build_object_points_from_pose_grad(
            master_obj_sparse_rest,
            pred_obj_rot6d_master,
            pred_obj_center_master.squeeze(1),
        )

        pred_joints_norm = self._pixels_to_norm(pred_joints_2d, (img_h, img_w))
        pred_center_norm = self._pixels_to_norm(
            pred_center_2d.view(batch_size * num_cams, self.num_obj_joints, 2),
            (img_h, img_w),
        )
        pred_mesh_2d_norm = self._pixels_to_norm(pred_mesh_2d, (img_h, img_w))

        pred_uv_conf = torch.ones(
            batch_size * num_cams,
            self.num_hand_joints + self.num_obj_joints,
            2,
            device=device,
            dtype=dtype,
        )
        pred_uv_heatmap = torch.zeros(
            batch_size * num_cams,
            self.num_hand_joints + self.num_obj_joints,
            self.data_preset_cfg.HEATMAP_SIZE[1],
            self.data_preset_cfg.HEATMAP_SIZE[0],
            device=device,
            dtype=dtype,
        )

        return {
            "pred_hand": pred_joints_norm,
            "pred_obj": pred_center_norm,
            "pred_hand_pixel": pred_joints_2d,
            "pred_obj_pixel": pred_center_2d.view(batch_size * num_cams, self.num_obj_joints, 2),
            "pred_uv_heatmap": pred_uv_heatmap,
            "pred_uv_conf_base": pred_uv_conf,
            "pred_uv_conf": pred_uv_conf,
            "conf_hand": torch.ones(batch_size * num_cams, self.num_hand_joints, 2, device=device, dtype=dtype),
            "mano_3d_mesh_sv": pred_mesh_rel,
            "mano_3d_mesh_sv_abs": pred_mesh_abs,
            "mano_2d_mesh_sv": pred_mesh_2d_norm,
            "mano_pose_euler_sv": mano_results["mano_full_pose"],
            "mano_pca_pose_sv": mano_results["mano_pca_pose"],
            "mano_pose_6d_sv": torch.zeros(batch_size * num_cams, 16 * 6, device=device, dtype=dtype),
            "mano_shape_sv": mano_results["mano_shape"],
            "mano_cam_sv": torch.zeros(batch_size * num_cams, 3, device=device, dtype=dtype),
            "ref_hand": pred_ref_hand,
            "ref_obj_center": pred_obj_center_master,
            "pred_obj_center_abs": pred_obj_center_abs,
            "pred_obj_corners_abs": pred_corners_abs,
            "hand_mesh_xyz_master": pred_mesh_master.unsqueeze(0),
            "obj_xyz_master": pred_obj_sparse_master.unsqueeze(0),
            "obj_rot6d": pred_obj_rot6d_master.unsqueeze(0),
            "obj_trans": pred_obj_trans_master.unsqueeze(0),
            "obj_view_rot6d_cam": pred_obj_rot6d_cam,
            "obj_view_trans": pred_obj_trans_rel,
            "obj_view_trans_master": pred_obj_center_abs,
            "obj_init_rot6d": pred_obj_rot6d_master,
            "obj_init_trans": pred_obj_trans_master,
            "obj_fused_rot6d_sv": pred_obj_rot6d_cam,
            "obj_fused_trans_sv": pred_obj_trans_rel,
            "mano_3d_mesh_master": pred_mesh_master,
            "mano_3d_mesh_kp_master": pred_mesh_master,
            "mano_3d_mesh_mesh_master": None,
            "pred_pose": mano_results["mano_pca_pose"].view(batch_size, num_cams, -1).mean(dim=1),
            "pred_shape": mano_results["mano_shape"].view(batch_size, num_cams, -1).mean(dim=1),
            "pred_kp_pose": mano_results["mano_pca_pose"].view(batch_size, num_cams, -1).mean(dim=1),
            "pred_kp_shape": mano_results["mano_shape"].view(batch_size, num_cams, -1).mean(dim=1),
            "pred_mesh_pose": None,
            "pred_mesh_shape": None,
            "pred_obj_trans_master": pred_obj_center_master.squeeze(1),
            "interaction_mode": "hopreg_direct",
            "all_hand_joints_xyz_master": pred_ref_hand.unsqueeze(0),
        }

    def compute_loss(self, preds, gt, stage_name="stage1", epoch_idx=None, **kwargs):
        zero = preds["mano_3d_mesh_sv"].sum() * 0.0
        batch_size, num_views = gt["image"].shape[:2]
        dtype = preds["mano_3d_mesh_sv"].dtype
        device = preds["mano_3d_mesh_sv"].device

        pred_mesh_abs = preds["mano_3d_mesh_sv_abs"]
        pred_joints_abs = pred_mesh_abs[:, self.num_hand_verts:]
        pred_verts_abs = pred_mesh_abs[:, :self.num_hand_verts]

        gt_joints_abs = gt["target_joints_3d"].to(device=device, dtype=dtype).reshape(batch_size * num_views, self.num_hand_joints, 3)
        gt_verts_abs = gt["target_verts_3d"].to(device=device, dtype=dtype).reshape(batch_size * num_views, self.num_hand_verts, 3)
        joints_vis = gt.get("target_joints_vis", None)
        if joints_vis is not None:
            joints_vis = joints_vis.to(device=device, dtype=dtype).reshape(batch_size * num_views, self.num_hand_joints)
            joint_valid = torch.isfinite(gt_joints_abs).all(dim=-1) & (joints_vis > 0)
        else:
            joint_valid = torch.isfinite(gt_joints_abs).all(dim=-1)
        verts_valid = torch.isfinite(gt_verts_abs).all(dim=-1)

        pred_obj_center_abs = preds["pred_obj_center_abs"].reshape(batch_size * num_views, self.num_obj_joints, 3)
        gt_obj_center_abs = gt["target_obj_kp21"][:, :, -self.num_obj_joints:].to(device=device, dtype=dtype).reshape(batch_size * num_views, self.num_obj_joints, 3)
        center_valid = torch.isfinite(gt_obj_center_abs).all(dim=-1) & (gt_obj_center_abs[..., 2] > 1e-6)

        pred_corners_abs = preds["pred_obj_corners_abs"].reshape(batch_size * num_views, 8, 3)
        gt_corners_abs = gt["target_obj_kp21"][:, :, :8].to(device=device, dtype=dtype).reshape(batch_size * num_views, 8, 3)
        corner_valid = torch.isfinite(gt_corners_abs).all(dim=-1) & (gt_corners_abs[..., 2] > 1e-6)

        pred_center_pixel = preds["pred_obj_pixel"]
        gt_center_pixel = batch_persp_project(
            gt_obj_center_abs,
            gt["target_cam_intr"].to(device=device, dtype=dtype).reshape(batch_size * num_views, 3, 3),
        )

        loss_pose_reg_sv = zero
        if self.arti_pose_reg_weight > 0.0:
            pred_pose = preds["mano_pca_pose_sv"][:, 3:]
            loss_pose_reg_sv = F.mse_loss(pred_pose, torch.zeros_like(pred_pose)) * self.arti_pose_reg_weight

        loss_shape_reg_sv = zero
        if self.arti_shape_reg_weight > 0.0:
            pred_shape = preds["mano_shape_sv"]
            loss_shape_reg_sv = F.mse_loss(pred_shape, torch.zeros_like(pred_shape)) * self.arti_shape_reg_weight

        loss_arti_mano_joints_3d = zero
        if self.arti_mano_joints_3d_weight > 0.0:
            loss_arti_mano_joints_3d = (
                self._artiboost_zero_mask_mse(pred_joints_abs, gt_joints_abs, joint_valid)
                * self.arti_mano_joints_3d_weight
            )

        loss_arti_mano_verts_3d = zero
        if self.arti_mano_verts_3d_weight > 0.0:
            loss_arti_mano_verts_3d = (
                self._artiboost_zero_mask_mse(pred_verts_abs, gt_verts_abs, verts_valid)
                * self.arti_mano_verts_3d_weight
            )

        loss_arti_joints_3d = zero
        if self.arti_joints_3d_weight > 0.0:
            loss_arti_joints_3d = (
                self._artiboost_zero_mask_mse(pred_joints_abs, gt_joints_abs, joint_valid)
                * self.arti_joints_3d_weight
            )

        loss_arti_verts_3d = zero
        if self.arti_verts_3d_weight > 0.0:
            loss_arti_verts_3d = (
                self._artiboost_zero_mask_mse(pred_verts_abs, gt_verts_abs, verts_valid)
                * self.arti_verts_3d_weight
            )

        loss_arti_corners_3d = zero
        if self.arti_corners_3d_weight > 0.0:
            loss_arti_corners_3d = (
                self._artiboost_zero_mask_mse(pred_corners_abs, gt_corners_abs, corner_valid)
                * self.arti_corners_3d_weight
            )

        hand_ord_joint = zero
        hand_ord_part = zero
        loss_arti_hand_ord = zero
        if self.arti_hand_ord_weight > 0.0:
            hand_ord_total, hand_ord_joint, hand_ord_part = self._compute_hand_ord_loss(
                pred_joints_abs,
                gt_joints_abs,
                joint_valid,
            )
            loss_arti_hand_ord = hand_ord_total * self.arti_hand_ord_weight

        loss_arti_scene_ord = zero
        if self.arti_scene_ord_weight > 0.0:
            scene_ord = self._compute_scene_ord_loss(
                pred_joints_abs,
                gt_joints_abs,
                joint_valid,
                pred_corners_abs,
                gt_corners_abs,
                corner_valid,
            )
            loss_arti_scene_ord = scene_ord * self.arti_scene_ord_weight

        loss_arti_mano = (
            loss_pose_reg_sv
            + loss_shape_reg_sv
            + loss_arti_mano_joints_3d
            + loss_arti_mano_verts_3d
        )
        loss_arti_joints = (
            loss_arti_joints_3d
            + loss_arti_verts_3d
            + loss_arti_corners_3d
        )
        total_loss = (
            self.arti_mano_weight * loss_arti_mano
            + self.arti_joints_group_weight * loss_arti_joints
            + loss_arti_hand_ord
            + loss_arti_scene_ord
        )

        loss_dict = {
            "loss_2d_sv": zero,
            "loss_arti_mano": loss_arti_mano,
            "loss_arti_joints": loss_arti_joints,
            "loss_pose_reg_sv": loss_pose_reg_sv,
            "loss_shape_reg_sv": loss_shape_reg_sv,
            "loss_arti_center_2d": zero,
            "loss_arti_center_3d": zero,
            "loss_arti_mano_joints_3d": loss_arti_mano_joints_3d,
            "loss_arti_mano_verts_3d": loss_arti_mano_verts_3d,
            "loss_arti_joints_3d": loss_arti_joints_3d,
            "loss_arti_verts_3d": loss_arti_verts_3d,
            "loss_arti_corners_3d": loss_arti_corners_3d,
            "loss_arti_hand_ord": loss_arti_hand_ord,
            "loss_arti_hand_ord_joint": hand_ord_joint,
            "loss_arti_hand_ord_part": hand_ord_part,
            "loss_arti_scene_ord": loss_arti_scene_ord,
            "metric_obj_center_2d_epe_px": self._masked_epe(pred_center_pixel, gt_center_pixel, center_valid).detach(),
            "metric_obj_center_3d_l1_mm": (self._masked_l1(pred_obj_center_abs, gt_obj_center_abs, center_valid).detach() * 1000.0),
            "metric_obj_corner_3d_l1_mm": (self._masked_l1(pred_corners_abs, gt_corners_abs, corner_valid).detach() * 1000.0),
            "metric_obj_center_conf_mean": zero.detach() + 1.0,
            "metric_obj_center_depth_mm": (pred_obj_center_abs[..., 2].mean().detach() * 1000.0),
            "loss": total_loss,
        }
        return total_loss, loss_dict

    def _log_visualizations(self, mode, batch, preds, step_idx, stage_name):
        if self.summary is None or not hasattr(self.summary, "add_image"):
            return

        img = batch["image"]
        batch_size, n_views = img.shape[:2]
        img_h, img_w = img.shape[-2:]
        batch_id = 0

        K_gt = batch["target_cam_intr"].to(device=img.device, dtype=img.dtype)

        pred_mano_2d_mesh_sv = preds["mano_2d_mesh_sv"].view(batch_size, n_views, self.num_hand_verts + self.num_hand_joints, 2)
        pred_sv_joints_2d = self._normed_uv_to_pixel(pred_mano_2d_mesh_sv[:, :, self.num_hand_verts:], (img_h, img_w))
        pred_sv_verts_2d = self._normed_uv_to_pixel(pred_mano_2d_mesh_sv[:, :, :self.num_hand_verts], (img_h, img_w))
        gt_joints_2d = self._normed_uv_to_pixel(
            batch["target_joints_uvd"][..., :2].to(device=img.device, dtype=img.dtype),
            (img_h, img_w),
        )
        gt_verts_2d = self._normed_uv_to_pixel(
            batch["target_verts_uvd"][..., :2].to(device=img.device, dtype=img.dtype),
            (img_h, img_w),
        )

        pred_corners_2d = batch_cam_intr_projection(
            K_gt,
            preds["pred_obj_corners_abs"].to(device=img.device, dtype=img.dtype),
        )
        gt_corners_2d = batch_cam_intr_projection(
            K_gt,
            batch["target_obj_kp21"][:, :, :8].to(device=img.device, dtype=img.dtype),
        )
        pred_obj_center_2d = preds["pred_obj_pixel"].view(batch_size, n_views, self.num_obj_joints, 2)
        gt_obj_center_2d = batch_cam_intr_projection(
            K_gt,
            batch["target_obj_kp21"][:, :, -self.num_obj_joints:].to(device=img.device, dtype=img.dtype),
        )

        obj_sparse_rest = batch["master_obj_sparse_rest"].to(device=img.device, dtype=img.dtype)
        pred_sv_obj_sparse_3d = self._build_object_points_from_hand_pose(
            obj_sparse_rest,
            preds["obj_view_rot6d_cam"],
            batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1].to(device=img.device, dtype=img.dtype),
            preds["obj_view_trans"],
        )
        pred_obj_sparse_2d = batch_cam_intr_projection(K_gt, pred_sv_obj_sparse_3d)

        if (
            batch.get("target_rot6d_label", None) is not None
            and batch.get("target_t_label_rel", None) is not None
        ):
            gt_sv_obj_sparse_3d = self._build_object_points_from_hand_pose(
                obj_sparse_rest,
                batch["target_rot6d_label"].to(device=img.device, dtype=img.dtype),
                batch["target_joints_3d"][:, :, self.center_idx:self.center_idx + 1].to(device=img.device, dtype=img.dtype),
                batch["target_t_label_rel"].to(device=img.device, dtype=img.dtype),
            )
            gt_obj_sparse_2d = batch_cam_intr_projection(K_gt, gt_sv_obj_sparse_3d)
        else:
            gt_obj_sparse_2d = None

        pred_obj_view_rot_deg = None
        if batch.get("target_rot6d_label", None) is not None and preds.get("obj_view_rot6d_cam", None) is not None:
            pred_obj_view_rot_deg = torch.rad2deg(
                self.rotation_geodesic(
                    preds["obj_view_rot6d_cam"][batch_id],
                    batch["target_rot6d_label"][batch_id].to(
                        device=preds["obj_view_rot6d_cam"].device,
                        dtype=preds["obj_view_rot6d_cam"].dtype,
                    ),
                )
            ).unsqueeze(-1)

        pred_obj_trans_mm = None
        if batch.get("target_t_label_rel", None) is not None and preds.get("obj_view_trans", None) is not None:
            pred_obj_trans_mm = (
                torch.linalg.norm(
                    preds["obj_view_trans"][batch_id] - batch["target_t_label_rel"][batch_id].to(
                        device=preds["obj_view_trans"].device,
                        dtype=preds["obj_view_trans"].dtype,
                    ),
                    dim=-1,
                ) * 1000.0
            ).unsqueeze(-1)

        tag_prefix = self._stage_tb_prefix(mode, stage_name)
        self.summary.add_image(
            f"{tag_prefix}/hopreg_hand_joints_2d",
            tile_batch_images(
                draw_batch_joint_images(
                    pred_sv_joints_2d[batch_id],
                    gt_joints_2d[batch_id],
                    img[batch_id],
                    step_idx,
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/hopreg_hand_mesh_2d",
            tile_batch_images(
                draw_batch_hand_mesh_images_2d(
                    gt_verts2d=gt_verts_2d[batch_id],
                    pred_verts2d=pred_sv_verts_2d[batch_id],
                    face=self.face,
                    tensor_image=img[batch_id],
                    n_sample=n_views,
                )
            ),
            step_idx,
            dataformats="HWC",
        )
        self.summary.add_image(
            f"{tag_prefix}/hopreg_obj_bbox_pointcloud_2d",
            tile_batch_images(
                draw_batch_object_point_cloud_images(
                    gt_obj_points2d=None if gt_obj_sparse_2d is None else gt_obj_sparse_2d[batch_id],
                    pred_obj_points2d=pred_obj_sparse_2d[batch_id],
                    tensor_image=img[batch_id],
                    gt_corners2d=gt_corners_2d[batch_id],
                    pred_corners2d=pred_corners_2d[batch_id],
                    gt_objc2d=gt_obj_center_2d[batch_id],
                    pred_objc2d=pred_obj_center_2d[batch_id],
                    pred_obj_rot_error=pred_obj_view_rot_deg,
                    pred_obj_trans_error=pred_obj_trans_mm,
                    n_sample=n_views,
                    max_points=512,
                )
            ),
            step_idx,
            dataformats="HWC",
        )

    def format_metric(self, mode="train"):
        def _mm(value):
            return f"{float(value) * 1000.0:.1f}"

        def _cd(value):
            return f"{float(value):.3f}"

        def _get_meter(name, default=0.0):
            meter = self.loss_metric._losses.get(name, None)
            return float(meter.avg) if meter is not None else float(default)

        stage_short = self._stage_display_name(self.current_stage_name, short=True)

        if mode == "train":
            l_total = _get_meter("loss")
            l_joint_group = _get_meter("loss_arti_joints")
            l_hj = _get_meter("loss_arti_joints_3d")
            l_ok = _get_meter("loss_arti_corners_3d")
            l_ho = _get_meter("loss_arti_hand_ord")
            l_so = _get_meter("loss_arti_scene_ord")
            m_obj_2d_px = _get_meter("metric_obj_center_2d_epe_px")
            m_obj_c_mm = _get_meter("metric_obj_center_3d_l1_mm")
            m_obj_k_mm = _get_meter("metric_obj_corner_3d_l1_mm")
            m_sv_rot_deg = _get_meter("metric_sv_obj_rot_deg")
            m_sv_trans_mm = _get_meter("metric_sv_obj_trans_epe")
            m_sv_obj_mssd = _get_meter("metric_sv_obj_mssd")
            m_sv_obj_cd = float(self.OBJ_RECON_SV.cd.avg)
            sv_j = self.MPJPE_SV_3D.get_result()
            sv_v = self.MPVPE_SV_3D.get_result()
            pa_sv = self.PA_SV.get_measures()
            return (
                f"{stage_short} | "
                f"L {l_total:.3f} | "
                f"J {l_joint_group:.4f} H/O {l_hj:.4f}/{l_ok:.4f} | "
                f"Ord H/S {l_ho:.4f}/{l_so:.4f} | "
                f"H MAJ/V {_mm(sv_j)}/{_mm(sv_v)} PAJ/V {_mm(pa_sv['pa_mpjpe'])}/{_mm(pa_sv['pa_mpvpe'])} | "
                f"O px/C/K {m_obj_2d_px:.1f}/{m_obj_c_mm:.1f}/{m_obj_k_mm:.1f} | "
                f"O MSSD/CD {_mm(m_sv_obj_mssd)}/{_cd(m_sv_obj_cd)} | "
                f"RT {m_sv_rot_deg:.1f}/{_mm(m_sv_trans_mm)}"
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
            sv_obj_cd = float(self.OBJ_RECON_SV.cd.avg)
            recon_mode = self.current_val_recon_mode.upper()
            return (
                f"{stage_short} | Rec {recon_mode} | "
                f"H MAJ/V {_mm(sv_j)}/{_mm(sv_v)} PAJ/V {_mm(pa_sv['pa_mpjpe'])}/{_mm(pa_sv['pa_mpvpe'])} | "
                f"O A/S/MSSD/CD {_mm(sv_obj_add)}/{_mm(sv_obj_adds)}/{_mm(sv_obj_mssd)}/{_cd(sv_obj_cd)} | "
                f"RT {sv_obj_rot_deg:.1f}/{_mm(sv_obj_trans_epe)}"
            )

        return super().format_metric(mode=mode)
