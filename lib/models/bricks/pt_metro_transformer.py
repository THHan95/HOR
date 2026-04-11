import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (BertAttention, BertIntermediate, BertOutput, BertPreTrainedModel,
                                                    apply_chunking_to_forward)

from .point_transformers import ptTransformerBlock, ptTransformerBlock_CrossAttn

class pointer_layer(nn.Module):

    def __init__(self, config, is_query_hand=True):
        super(pointer_layer, self).__init__()

        self.nneighbor_cross = config.n_neighbor  # 32
        self.nneighbor_query = config.n_neighbor_query  # 32
        self.nneighbor_decay = True
        self.is_query_hand = is_query_hand
        self.update_xyz = is_query_hand
        self.feat_dim = config.img_feature_dim  # 256
        self.transformer_dim = self.feat_dim  # 256
        self.init_block = config.init_block  # True for first block, False otherwise
        self.obj_xyz_update_scale = getattr(config, "obj_xyz_update_scale", 0.0)

        self.reg_branch = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Linear(self.feat_dim, 3))
        self.query_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor_query,
                                                  self.init_block)
        self.query_cross_attn = ptTransformerBlock_CrossAttn(self.feat_dim,
                                                             self.transformer_dim,
                                                             self.nneighbor_cross,
                                                             expand_query_dim=False,
                                                             IFPS=self.init_block)

    def forward(self, query_xyz, query_feats, key_xyz, key_feats):
        query_feat, _ = self.query_self_attn(query_xyz, query_feats)  # self-attention
        query_cat = torch.cat((query_xyz, query_feat), dim=-1)
        query_feat, _ = self.query_cross_attn(key_xyz, key_feats, query_cat)  # cross-attention
        query_delta = self.reg_branch(query_feat)
        if self.update_xyz:
            query_xyz = query_delta + query_xyz
        else:
            query_xyz = (self.obj_xyz_update_scale * query_delta) + query_xyz

        return query_feat, query_xyz


class point_METRO_layer(nn.Module):

    def __init__(self, config, is_query_hand=True):
        super(point_METRO_layer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

        self.attn = BertAttention(config)
        self.cross_attn = BertAttention(config, position_embedding_type="absolute")
        self.vec_attn = pointer_layer(config, is_query_hand=is_query_hand)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, query_feats, query_xyz, key_feats, key_xyz):
        # 1. 全局特征自注意力
        self_attn_out = self.attn(
            hidden_states=query_feats, attention_mask=None, output_attentions=False
        )[0]

        # 2. 全局特征交叉注意力
        cross_attn_out = self.cross_attn(
            hidden_states=self_attn_out, attention_mask=None, 
            encoder_hidden_states=key_feats, output_attentions=False
        )[0]

        # 3. 3D 几何特征更新 (Point Transformer 模块)
        query_feats, query_xyz = self.vec_attn(
            query_xyz=query_xyz, query_feats=cross_attn_out, 
            key_xyz=key_xyz, key_feats=key_feats
        )

        # 4. 前馈网络 FFN
        query_feats = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, 1, query_feats
        )

        return query_feats, query_xyz

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class point_METRO_block(BertPreTrainedModel):
    """
        Modified the original METRO encoder to provide an interface for key value and query to be different.
    """

    def __init__(self, config):
        super(point_METRO_block, self).__init__(config)
        self.config = config

        # 🌟 实例化双向查询通道
        self.hand_update_layer = point_METRO_layer(config, is_query_hand=True)
        self.obj_update_layer = point_METRO_layer(config, is_query_hand=False)
        self.hand_anchor_update_layer = point_METRO_layer(config, is_query_hand=True)
        self.hand_refine_layer = point_METRO_layer(config, is_query_hand=True)

        self.input_dim = config.img_feature_dim

        self.embedding = nn.Linear(self.input_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif type(module).__name__ == 'BertLMPredictionHead':
            module.bias.data.zero_()

    def _embed_inputs(self, hand_feats, obj_feats):
        hand_feats = self.dropout(self.embedding(hand_feats))
        obj_feats = self.dropout(self.embedding(obj_feats))
        return hand_feats, obj_feats

    def _forward_hand_anchor(self, hand_xyz, hand_feats, obj_xyz, obj_feats):
        updated_hand_feats, pt_updated_hand_xyz = self.hand_update_layer(
            query_feats=hand_feats,
            query_xyz=hand_xyz,
            key_feats=obj_feats,
            key_xyz=obj_xyz,
        )
        updated_obj_feats, updated_obj_xyz = obj_feats, obj_xyz
        return pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats

    def _forward_mesh_obj(self, hand_xyz, hand_feats, obj_xyz, obj_feats):
        updated_hand_feats, pt_updated_hand_xyz = self.hand_update_layer(
            query_feats=hand_feats,
            query_xyz=hand_xyz,
            key_feats=obj_feats,
            key_xyz=obj_xyz,
        )
        updated_obj_feats, _ = self.obj_update_layer(
            query_feats=obj_feats,
            query_xyz=obj_xyz,
            key_feats=updated_hand_feats,
            key_xyz=pt_updated_hand_xyz,
        )
        updated_obj_xyz = obj_xyz
        return pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats

    def _forward_hand(self, hand_xyz, hand_feats, obj_xyz, obj_feats):
        updated_hand_feats, pt_updated_hand_xyz = self.hand_update_layer(
            query_feats=hand_feats,
            query_xyz=hand_xyz,
            key_feats=obj_feats,
            key_xyz=obj_xyz,
        )
        updated_obj_feats, updated_obj_xyz = self.hand_anchor_update_layer(
            query_feats=obj_feats,
            query_xyz=obj_xyz,
            key_feats=updated_hand_feats,
            key_xyz=pt_updated_hand_xyz,
        )
        updated_hand_feats, pt_updated_hand_xyz = self.hand_refine_layer(
            query_feats=updated_hand_feats,
            query_xyz=pt_updated_hand_xyz,
            key_feats=updated_obj_feats,
            key_xyz=updated_obj_xyz,
        )
        return pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats

    def _forward_ho(self, hand_xyz, hand_feats, obj_xyz, obj_feats):
        updated_hand_feats, pt_updated_hand_xyz = self.hand_update_layer(
            query_feats=hand_feats,
            query_xyz=hand_xyz,
            key_feats=obj_feats,
            key_xyz=obj_xyz,
        )
        updated_obj_feats, updated_obj_xyz = self.obj_update_layer(
            query_feats=obj_feats,
            query_xyz=obj_xyz,
            key_feats=updated_hand_feats,
            key_xyz=pt_updated_hand_xyz,
        )
        return pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats

    def forward(self, hand_xyz, hand_feats, obj_xyz, obj_feats, interaction_mode="ho"):
        hand_feats, obj_feats = self._embed_inputs(hand_feats, obj_feats)

        if interaction_mode == "hand_anchor":
            pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats = self._forward_hand_anchor(
                hand_xyz, hand_feats, obj_xyz, obj_feats
            )
        elif interaction_mode == "mesh_obj":
            pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats = self._forward_mesh_obj(
                hand_xyz, hand_feats, obj_xyz, obj_feats
            )
        elif interaction_mode == "hand":
            pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats = self._forward_hand(
                hand_xyz, hand_feats, obj_xyz, obj_feats
            )
        else:
            pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats = self._forward_ho(
                hand_xyz, hand_feats, obj_xyz, obj_feats
            )

        return pt_updated_hand_xyz, updated_hand_feats, updated_obj_xyz, updated_obj_feats
