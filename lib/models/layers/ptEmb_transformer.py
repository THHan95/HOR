import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from dataclasses import dataclass
from typing import Callable

from ...utils.builder import TRANSFORMER
from ...utils.net_utils import xavier_init
from ...utils.transform import inverse_sigmoid
from ...utils.logger import logger
from ...utils.misc import param_size
from ...utils.points_utils import index_points
from transformers.models.bert.modeling_bert import BertConfig
from ..bricks.pt_metro_transformer import point_METRO_block
from ..bricks.point_transformers import (
    ptTransformerBlock,
    ptTransformerBlock_CrossAttn,
)


@dataclass
class HORHandRefineContext:
    hand_center_point: torch.Tensor
    anchor_verts_xyz_master: torch.Tensor
    anchor_verts_xyz_norm: torch.Tensor
    anchor_verts_feats_norm: torch.Tensor


@dataclass
class HORHandObjectContext:
    obj_template: torch.Tensor
    hand_center_point: torch.Tensor
    current_rot6d: torch.Tensor
    current_trans_norm: torch.Tensor
    obj_view_conf: torch.Tensor
    obj_view_conf_xy: torch.Tensor
    obj_occ_map: torch.Tensor
    feat_map: torch.Tensor
    img_metas: dict
    inp_res: torch.Tensor
    obj_center_feat: torch.Tensor
    global_mv_feat: torch.Tensor
    sample_multiview_features_fn: Callable
    project_points_to_views_fn: Callable
    build_object_points_from_pose_fn: Callable
    build_object_point_view_weights_fn: Callable
    update_object_pose_fn: Callable
    obj_feat_fuser: nn.Module
    hand_update_scale: float
    obj_feat_update_scale: float
    pose_delta_scale: float


@dataclass
class HORHandAnchorOutput:
    all_hand_mesh_xyz_master: torch.Tensor
    all_hand_joints_xyz_master: torch.Tensor
    final_hand_mesh_xyz_norm: torch.Tensor
    final_hand_mesh_feats_norm: torch.Tensor


@dataclass
class HORMeshObjOutput:
    all_hand_mesh_xyz_master: torch.Tensor
    all_obj_xyz_master: torch.Tensor
    all_obj_center_xyz_master: torch.Tensor
    all_obj_rot6d: torch.Tensor
    all_obj_trans: torch.Tensor
    final_hand_mesh_xyz_norm: torch.Tensor
    final_hand_mesh_feats_norm: torch.Tensor


@dataclass
class HORDefaultOutput:
    all_hand_mesh_xyz_master: torch.Tensor
    all_obj_xyz_master: torch.Tensor
    all_obj_feats: torch.Tensor

# @TRANSFORMER.register_module()
# class PtEmbedTRv2(nn.Module):

#     def __init__(self, cfg):
#         super(PtEmbedTRv2, self).__init__()
#         self._is_init = False

#         self.nblocks = cfg.N_BLOCKS
#         self.nneighbor = cfg.N_NEIGHBOR
#         self.nneighbor_query = cfg.N_NEIGHBOR_QUERY
#         self.nneighbor_decay = cfg.get("N_NEIGHBOR_DECAY", True)
#         self.transformer_dim = cfg.TRANSFORMER_DIM
#         self.feat_dim = cfg.POINTS_FEAT_DIM
#         self.with_point_embed = cfg.WITH_POSI_EMBED

#         self.predict_inv_sigmoid = cfg.get("PREDICT_INV_SIGMOID", False)

#         self.feats_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor)
#         self.query_feats_cross_attn = nn.ModuleList()
#         self.query_self_attn = nn.ModuleList()

#         for i in range(self.nblocks):
#             self.query_self_attn.append(ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor_query))
#             self.query_feats_cross_attn.append(
#                 ptTransformerBlock_CrossAttn(self.feat_dim,
#                                              self.transformer_dim,
#                                              self.nneighbor,
#                                              expand_query_dim=False))

#         # self.init_weights()
#         logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

#     def init_weights(self):
#         if self._is_init == True:
#             return

#         # follow the official DETR to init parameters
#         for m in self.modules():
#             if hasattr(m, 'weight') and m.weight.dim() > 1:
#                 xavier_init(m, distribution='uniform')
#         self._is_init = True
#         logger.info(f"{type(self).__name__} init done")

#     def forward(self, pt_xyz, pt_feats, query_xyz, reg_branches, query_feat=None, pt_embed=None, query_emb=None):
#         if pt_embed is not None and self.with_point_embed:
#             pt_feats = pt_feats + pt_embed

#         if query_feat is None:
#             query_feats = query_emb
#         else:
#             query_feats = query_feat + query_emb

#         pt_feats, _ = self.feats_self_attn(pt_xyz, pt_feats)

#         query_xyz_n = []
#         query_feats_n = []

#         # query_feats = query_emb
#         for i in range(self.nblocks):
#             query_feats, _ = self.query_self_attn[i](query_xyz, query_feats)

#             query = torch.cat((query_xyz, query_feats), dim=-1)

#             query_feats, _ = self.query_feats_cross_attn[i](pt_xyz, pt_feats, query)

#             if self.predict_inv_sigmoid:
#                 query_xyz = reg_branches[i](query_feats) + inverse_sigmoid(query_xyz)
#                 query_xyz = query_xyz.sigmoid()
#             else:
#                 query_xyz = reg_branches[i](query_feats) + query_xyz

#             query_xyz_n.append(query_xyz)
#             query_feats_n.append(query_feats)

#         return torch.stack(query_xyz_n)
    

class _Sequential(nn.Sequential):
    """
        A wrapper to allow nn.Sequential to accept multiple inputs
    """

    def forward(self, hand_xyz, hand_feats, obj_xyz, obj_feats, interaction_mode="ho"):
        hand_xyz_n = []
        obj_xyz_n = []
        obj_feats_n = []
        for i, module in enumerate(self._modules.values()):
            hand_xyz, hand_feats, obj_xyz, obj_feats = module(
                hand_xyz, hand_feats, obj_xyz, obj_feats, interaction_mode=interaction_mode
            )
            hand_xyz_n.append(hand_xyz)
            obj_xyz_n.append(obj_xyz)
            obj_feats_n.append(obj_feats)
        hand_xyz_n = torch.stack(hand_xyz_n)
        obj_xyz_n = torch.stack(obj_xyz_n)
        obj_feats_n = torch.stack(obj_feats_n)
        return hand_xyz_n, obj_xyz_n, obj_feats_n


class _HORTRBase(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        # Configuration for METRO part
        self.input_feat_dim = cfg.INPUT_FEAT_DIM
        self.hidden_feat_dim = self.input_feat_dim
        self.output_feat_dim = self.input_feat_dim
        self.dropout = cfg.DROPOUT
        self.num_hidden_layers = cfg.NUM_HIDDEN_LAYERS
        self.num_attention_heads = cfg.NUM_ATTENTION_HEADS
        self.obj_xyz_update_scale = cfg.get("OBJ_XYZ_UPDATE_SCALE", 0.0)

        # Configuration for PT part
        self.nneighbor = cfg.N_NEIGHBOR
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY

        # * load metro block
        self.pt_metro_encoder = []
        self.layer_num = cfg.N_BLOCKS

        # init three transformer-encoder blocks in a loop
        for i in range(self.layer_num):
            config_class, model_class = BertConfig, point_METRO_block

            config = config_class.from_pretrained("config/backbone/bert_cfg.json")
            config.output_attentions = False
            config.hidden_dropout_prob = self.dropout
            config.img_feature_dim = self.input_feat_dim
            config.output_feature_dim = self.output_feat_dim
            self.hidden_size = self.hidden_feat_dim
            self.intermediate_size = self.hidden_size * 4

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for _, param in enumerate(update_params):
                arg_param = getattr(self, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)

            # Required, as default value 512 < 799/4096
            config.max_position_embeddings = 2048

            # Add the PT part to config
            config.n_neighbor = self.nneighbor
            config.n_neighbor_query = self.nneighbor_query
            config.init_block = False
            config.obj_xyz_update_scale = self.obj_xyz_update_scale

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            self.pt_metro_encoder.append(model)

        self.pt_metro_encoder = _Sequential(*self.pt_metro_encoder)

        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def _iter_blocks(self):
        return self.pt_metro_encoder._modules.values()

    @staticmethod
    def _sanitize_tensor(x, abs_max, nan_value=0.0):
        return torch.nan_to_num(x, nan=nan_value, posinf=abs_max, neginf=-abs_max).clamp(-abs_max, abs_max)

    @staticmethod
    def _log_nonfinite_tensor(tag, layer_idx, tensor_name, x):
        finite_mask = torch.isfinite(x)
        if finite_mask.all():
            return
        x_detached = x.detach()
        safe_x = torch.where(finite_mask, x_detached, torch.zeros_like(x_detached))
        logger.warning(
            f"[{tag}] layer={layer_idx} tensor={tensor_name} "
            f"finite_ratio={float(finite_mask.float().mean().item()):.6f} "
            f"nan={int(torch.isnan(x_detached).sum().item())} "
            f"inf={int(torch.isinf(x_detached).sum().item())} "
            f"max_finite_abs={float(safe_x.abs().max().item()):.4f}"
        )

    def _stabilize_tensor(self, x, abs_max, fallback=None, nan_value=0.0):
        if fallback is not None:
            fallback = self._sanitize_tensor(fallback, abs_max, nan_value=nan_value).to(dtype=x.dtype, device=x.device)
            finite_mask = torch.isfinite(x)
            if not finite_mask.all():
                x = torch.where(finite_mask, x, fallback)
        return self._sanitize_tensor(x, abs_max, nan_value=nan_value)

    def _forward_hand_anchor(self, query_xyz, query_feat, pt_xyz, pt_feats, hand_refine_context):
        hand_xyz_norm_abs_max = 25.0
        hand_xyz_master_abs_max = 5.0
        feat_abs_max = 100.0
        current_joint_xyz = self._sanitize_tensor(query_xyz, hand_xyz_norm_abs_max)
        current_joint_feats = self._sanitize_tensor(query_feat, feat_abs_max)
        pt_xyz = self._sanitize_tensor(pt_xyz, hand_xyz_norm_abs_max)
        pt_feats = self._sanitize_tensor(pt_feats, feat_abs_max)
        all_hand_mesh_xyz_master = []
        all_hand_joints_xyz_master = []

        for layer_idx, block in enumerate(self._iter_blocks()):
            next_joint_xyz, next_joint_feats, _, _ = block(
                hand_xyz=current_joint_xyz,
                hand_feats=current_joint_feats,
                obj_xyz=pt_xyz,
                obj_feats=pt_feats,
                interaction_mode="hand_anchor",
            )
            self._log_nonfinite_tensor("NonFiniteHandAnchor", layer_idx, "next_joint_xyz", next_joint_xyz)
            self._log_nonfinite_tensor("NonFiniteHandAnchor", layer_idx, "next_joint_feats", next_joint_feats)
            current_joint_xyz = self._stabilize_tensor(next_joint_xyz, hand_xyz_norm_abs_max, fallback=current_joint_xyz)
            current_joint_feats = self._stabilize_tensor(next_joint_feats, feat_abs_max, fallback=current_joint_feats)
            current_joint_xyz = self._sanitize_tensor(current_joint_xyz, hand_xyz_norm_abs_max)
            current_joint_xyz_master = (current_joint_xyz * 0.2) + hand_refine_context.hand_center_point
            current_joint_xyz_master = self._sanitize_tensor(current_joint_xyz_master, hand_xyz_master_abs_max)
            all_hand_mesh_xyz_master.append(torch.cat((hand_refine_context.anchor_verts_xyz_master, current_joint_xyz_master), dim=1))
            all_hand_joints_xyz_master.append(current_joint_xyz_master)

        final_hand_mesh_xyz_norm = torch.cat(
            (hand_refine_context.anchor_verts_xyz_norm, self._sanitize_tensor(current_joint_xyz, hand_xyz_norm_abs_max)),
            dim=1,
        )
        final_hand_mesh_feats_norm = torch.cat((hand_refine_context.anchor_verts_feats_norm, current_joint_feats), dim=1)
        return HORHandAnchorOutput(
            all_hand_mesh_xyz_master=torch.stack(all_hand_mesh_xyz_master),
            all_hand_joints_xyz_master=torch.stack(all_hand_joints_xyz_master),
            final_hand_mesh_xyz_norm=final_hand_mesh_xyz_norm,
            final_hand_mesh_feats_norm=final_hand_mesh_feats_norm,
        )

    def _forward_mesh_obj(self, query_xyz, query_feat, hand_object_context):
        obj_template = hand_object_context.obj_template
        hand_center_point = hand_object_context.hand_center_point
        current_rot6d = hand_object_context.current_rot6d
        current_trans_norm = hand_object_context.current_trans_norm
        feat_map = hand_object_context.feat_map
        img_metas = hand_object_context.img_metas
        inp_res = hand_object_context.inp_res
        obj_center_feat = hand_object_context.obj_center_feat
        global_mv_feat = hand_object_context.global_mv_feat
        sample_multiview_features_fn = hand_object_context.sample_multiview_features_fn
        project_points_to_views_fn = hand_object_context.project_points_to_views_fn
        build_object_points_from_pose_fn = hand_object_context.build_object_points_from_pose_fn
        build_object_point_view_weights_fn = hand_object_context.build_object_point_view_weights_fn
        update_object_pose_fn = hand_object_context.update_object_pose_fn
        obj_feat_fuser = hand_object_context.obj_feat_fuser
        hand_update_scale = hand_object_context.hand_update_scale
        obj_feat_update_scale = hand_object_context.obj_feat_update_scale
        pose_delta_scale = hand_object_context.pose_delta_scale

        hand_xyz_norm_abs_max = 25.0
        hand_xyz_master_abs_max = 5.0
        feat_abs_max = 100.0
        current_hand_xyz = self._sanitize_tensor(query_xyz, hand_xyz_norm_abs_max)
        current_hand_feats = self._sanitize_tensor(query_feat, feat_abs_max)
        current_obj_feats = None

        all_hand_mesh_xyz_master = []
        all_obj_xyz_master = []
        all_obj_center_xyz_master = []
        all_obj_rot6d = []
        all_obj_trans = []

        batch_size, num_cams = feat_map.shape[:2]

        for layer_idx, block in enumerate(self._iter_blocks()):
            current_obj_xyz_norm, current_obj_xyz_master, current_obj_center_master, _ = build_object_points_from_pose_fn(
                obj_template,
                hand_center_point,
                current_rot6d,
                current_trans_norm,
            )
            obj_query_2d = project_points_to_views_fn(current_obj_xyz_master, img_metas, inp_res)
            obj_point_view_weights = build_object_point_view_weights_fn(
                current_obj_xyz_master,
                current_obj_center_master,
                img_metas,
                inp_res,
                hand_object_context.obj_view_conf,
                hand_object_context.obj_view_conf_xy,
                hand_object_context.obj_occ_map,
            )
            obj_img_feats = sample_multiview_features_fn(
                feat_map,
                obj_query_2d,
                batch_size,
                num_cams,
                current_obj_xyz_master.shape[1],
                view_weights=obj_point_view_weights,
            )
            obj_img_feats = self._sanitize_tensor(obj_img_feats, feat_abs_max)

            if current_obj_feats is None:
                current_obj_feats = obj_img_feats
            else:
                fused_obj_feats = obj_feat_fuser(torch.cat((current_obj_feats, obj_img_feats), dim=-1))
                self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "fused_obj_feats", fused_obj_feats)
                current_obj_feats = self._stabilize_tensor(fused_obj_feats, feat_abs_max, fallback=current_obj_feats)

            prev_hand_xyz = current_hand_xyz
            prev_hand_feats = current_hand_feats
            prev_obj_feats = current_obj_feats
            next_hand_xyz, next_hand_feats, _, next_obj_feats = block(
                hand_xyz=current_hand_xyz,
                hand_feats=current_hand_feats,
                obj_xyz=current_obj_xyz_norm,
                obj_feats=current_obj_feats,
                interaction_mode="mesh_obj",
            )
            self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "next_hand_xyz", next_hand_xyz)
            self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "next_hand_feats", next_hand_feats)
            self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "next_obj_feats", next_obj_feats)
            next_hand_xyz = self._stabilize_tensor(next_hand_xyz, hand_xyz_norm_abs_max, fallback=prev_hand_xyz)
            next_hand_feats = self._stabilize_tensor(next_hand_feats, feat_abs_max, fallback=prev_hand_feats)
            next_obj_feats = self._stabilize_tensor(next_obj_feats, feat_abs_max, fallback=prev_obj_feats)

            current_hand_xyz = prev_hand_xyz + hand_update_scale * (next_hand_xyz - prev_hand_xyz)
            current_hand_feats = prev_hand_feats + hand_update_scale * (next_hand_feats - prev_hand_feats)
            current_obj_feats = prev_obj_feats + obj_feat_update_scale * (next_obj_feats - prev_obj_feats)
            self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "current_hand_xyz", current_hand_xyz)
            self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "current_hand_feats", current_hand_feats)
            self._log_nonfinite_tensor("NonFiniteHandMeshObj", layer_idx, "current_obj_feats", current_obj_feats)
            current_hand_xyz = self._stabilize_tensor(current_hand_xyz, hand_xyz_norm_abs_max, fallback=prev_hand_xyz)
            current_hand_feats = self._stabilize_tensor(current_hand_feats, feat_abs_max, fallback=prev_hand_feats)
            current_obj_feats = self._stabilize_tensor(current_obj_feats, feat_abs_max, fallback=prev_obj_feats)

            current_rot6d, current_trans_norm = update_object_pose_fn(
                current_obj_feats,
                obj_center_feat,
                global_mv_feat,
                current_rot6d,
                current_trans_norm,
                warmup_scale=pose_delta_scale,
            )
            current_obj_xyz_norm, current_obj_xyz_master, current_obj_center_xyz_master, current_obj_trans = build_object_points_from_pose_fn(
                obj_template,
                hand_center_point,
                current_rot6d,
                current_trans_norm,
            )

            all_hand_mesh_xyz_master.append(
                self._sanitize_tensor((current_hand_xyz * 0.2) + hand_center_point, hand_xyz_master_abs_max)
            )
            all_obj_xyz_master.append(current_obj_xyz_master)
            all_obj_center_xyz_master.append(current_obj_center_xyz_master)
            all_obj_rot6d.append(current_rot6d)
            all_obj_trans.append(current_obj_trans)

        return HORMeshObjOutput(
            all_hand_mesh_xyz_master=torch.stack(all_hand_mesh_xyz_master),
            all_obj_xyz_master=torch.stack(all_obj_xyz_master),
            all_obj_center_xyz_master=torch.stack(all_obj_center_xyz_master),
            all_obj_rot6d=torch.stack(all_obj_rot6d),
            all_obj_trans=torch.stack(all_obj_trans),
            final_hand_mesh_xyz_norm=self._sanitize_tensor(current_hand_xyz, hand_xyz_norm_abs_max),
            final_hand_mesh_feats_norm=current_hand_feats,
        )

    def _forward_default(self, query_xyz, query_feat, pt_xyz, pt_feats, interaction_mode="ho"):
        pred_hand_joints, pred_obj_points, pred_obj_feats = self.pt_metro_encoder(
            hand_xyz=query_xyz,
            hand_feats=query_feat,
            obj_xyz=pt_xyz,
            obj_feats=pt_feats,
            interaction_mode=interaction_mode,
        )
        return HORDefaultOutput(
            all_hand_mesh_xyz_master=pred_hand_joints,
            all_obj_xyz_master=pred_obj_points,
            all_obj_feats=pred_obj_feats,
        )


@TRANSFORMER.register_module()
class HORTR_Hand(_HORTRBase):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, query_xyz, query_feat, pt_xyz, pt_feats, hand_refine_context=None, **kwargs):
        if hand_refine_context is None:
            raise ValueError("HORTR_Hand requires hand_refine_context")
        return self._forward_hand_anchor(
            query_xyz=query_xyz,
            query_feat=query_feat,
            pt_xyz=pt_xyz,
            pt_feats=pt_feats,
            hand_refine_context=hand_refine_context,
        )


@TRANSFORMER.register_module()
class HORTR_HO(_HORTRBase):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(
        self,
        query_xyz,
        query_feat,
        pt_xyz=None,
        pt_feats=None,
        interaction_mode="mesh_obj",
        hand_object_context=None,
        **kwargs,
    ):
        if interaction_mode == "mesh_obj":
            if hand_object_context is None:
                raise ValueError("HORTR_HO requires hand_object_context when interaction_mode='mesh_obj'")
            return self._forward_mesh_obj(
                query_xyz=query_xyz,
                query_feat=query_feat,
                hand_object_context=hand_object_context,
            )

        if pt_xyz is None or pt_feats is None:
            raise ValueError("HORTR_HO requires pt_xyz and pt_feats for non-mesh_obj interaction")
        return self._forward_default(query_xyz, query_feat, pt_xyz, pt_feats, interaction_mode=interaction_mode)


@TRANSFORMER.register_module()
class HORTR(HORTR_Hand):

    def __init__(self, cfg):
        super().__init__(cfg)
