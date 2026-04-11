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
class HORStage2Context:
    hand_center_point: torch.Tensor
    anchor_verts_xyz_master: torch.Tensor
    anchor_verts_xyz_norm: torch.Tensor
    anchor_verts_feats_norm: torch.Tensor


@dataclass
class HORStage3Context:
    obj_template: torch.Tensor
    hand_center_point: torch.Tensor
    current_rot6d: torch.Tensor
    current_trans_norm: torch.Tensor
    feat_map: torch.Tensor
    img_metas: dict
    inp_res: torch.Tensor
    obj_center_feat: torch.Tensor
    global_mv_feat: torch.Tensor
    sample_multiview_features_fn: Callable
    project_points_to_views_fn: Callable
    build_object_points_from_pose_fn: Callable
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


@TRANSFORMER.register_module()
class HORTR(nn.Module):

    def __init__(self, cfg):
        super(HORTR, self).__init__()
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

    def _forward_hand_anchor(self, query_xyz, query_feat, pt_xyz, pt_feats, stage2_context):
        current_joint_xyz = query_xyz
        current_joint_feats = query_feat
        all_hand_mesh_xyz_master = []
        all_hand_joints_xyz_master = []

        for block in self._iter_blocks():
            current_joint_xyz, current_joint_feats, _, _ = block(
                hand_xyz=current_joint_xyz,
                hand_feats=current_joint_feats,
                obj_xyz=pt_xyz,
                obj_feats=pt_feats,
                interaction_mode="hand_anchor",
            )
            current_joint_xyz_master = (current_joint_xyz * 0.2) + stage2_context.hand_center_point
            all_hand_mesh_xyz_master.append(torch.cat((stage2_context.anchor_verts_xyz_master, current_joint_xyz_master), dim=1))
            all_hand_joints_xyz_master.append(current_joint_xyz_master)

        final_hand_mesh_xyz_norm = torch.cat((stage2_context.anchor_verts_xyz_norm, current_joint_xyz), dim=1)
        final_hand_mesh_feats_norm = torch.cat((stage2_context.anchor_verts_feats_norm, current_joint_feats), dim=1)
        return HORHandAnchorOutput(
            all_hand_mesh_xyz_master=torch.stack(all_hand_mesh_xyz_master),
            all_hand_joints_xyz_master=torch.stack(all_hand_joints_xyz_master),
            final_hand_mesh_xyz_norm=final_hand_mesh_xyz_norm,
            final_hand_mesh_feats_norm=final_hand_mesh_feats_norm,
        )

    def _forward_mesh_obj(self, query_xyz, query_feat, stage3_context):
        obj_template = stage3_context.obj_template
        hand_center_point = stage3_context.hand_center_point
        current_rot6d = stage3_context.current_rot6d
        current_trans_norm = stage3_context.current_trans_norm
        feat_map = stage3_context.feat_map
        img_metas = stage3_context.img_metas
        inp_res = stage3_context.inp_res
        obj_center_feat = stage3_context.obj_center_feat
        global_mv_feat = stage3_context.global_mv_feat
        sample_multiview_features_fn = stage3_context.sample_multiview_features_fn
        project_points_to_views_fn = stage3_context.project_points_to_views_fn
        build_object_points_from_pose_fn = stage3_context.build_object_points_from_pose_fn
        update_object_pose_fn = stage3_context.update_object_pose_fn
        obj_feat_fuser = stage3_context.obj_feat_fuser
        hand_update_scale = stage3_context.hand_update_scale
        obj_feat_update_scale = stage3_context.obj_feat_update_scale
        pose_delta_scale = stage3_context.pose_delta_scale

        current_hand_xyz = query_xyz
        current_hand_feats = query_feat
        current_obj_feats = None

        all_hand_mesh_xyz_master = []
        all_obj_xyz_master = []
        all_obj_center_xyz_master = []
        all_obj_rot6d = []
        all_obj_trans = []

        batch_size, num_cams = feat_map.shape[:2]

        for block in self._iter_blocks():
            current_obj_xyz_norm, current_obj_xyz_master, _, _ = build_object_points_from_pose_fn(
                obj_template,
                hand_center_point,
                current_rot6d,
                current_trans_norm,
            )
            obj_query_2d = project_points_to_views_fn(current_obj_xyz_master, img_metas, inp_res)
            obj_img_feats = sample_multiview_features_fn(
                feat_map,
                obj_query_2d,
                batch_size,
                num_cams,
                current_obj_xyz_master.shape[1],
            )

            if current_obj_feats is None:
                current_obj_feats = obj_img_feats
            else:
                current_obj_feats = obj_feat_fuser(torch.cat((current_obj_feats, obj_img_feats), dim=-1))

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

            current_hand_xyz = prev_hand_xyz + hand_update_scale * (next_hand_xyz - prev_hand_xyz)
            current_hand_feats = prev_hand_feats + hand_update_scale * (next_hand_feats - prev_hand_feats)
            current_obj_feats = prev_obj_feats + obj_feat_update_scale * (next_obj_feats - prev_obj_feats)

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

            all_hand_mesh_xyz_master.append((current_hand_xyz * 0.2) + hand_center_point)
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
            final_hand_mesh_xyz_norm=current_hand_xyz,
            final_hand_mesh_feats_norm=current_hand_feats,
        )

    def forward(self, query_xyz, query_feat, pt_xyz, pt_feats, interaction_mode="ho", **kwargs):
        if interaction_mode == "hand":
            return self._forward_hand_anchor(
                query_xyz=query_xyz,
                query_feat=query_feat,
                pt_xyz=pt_xyz,
                pt_feats=pt_feats,
                stage2_context=kwargs["stage2_context"],
            )

        if interaction_mode == "mesh_obj":
            return self._forward_mesh_obj(
                query_xyz=query_xyz,
                query_feat=query_feat,
                stage3_context=kwargs["stage3_context"],
            )

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
