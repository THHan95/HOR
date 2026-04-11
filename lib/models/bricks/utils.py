from ...utils.triangulation import batch_triangulate_dlt_torch_confidence
from torch import nn

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchgeometry as tgm
from manotorch.manolayer import ManoLayer


def orthgonalProj(xy, scale, transl, img_size=256):
    scale = scale * img_size
    transl = transl * img_size / 2 + img_size / 2
    return xy * scale + transl


class ManoDecoder(nn.Module):
    def __init__(self, root_idx, bbox_3d, input_img_shape):
        super(ManoDecoder, self).__init__()
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=None,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=False,
        )
        self.face_open = self.mano_layer.th_faces
        self.face_closed = self.mano_layer.get_mano_closed_faces()
        self.face = self.face_closed
        self.root_joint_idx = root_idx
        self.bbox_3d_size = bbox_3d
        self.input_img_shape = input_img_shape

    def rot6d_to_rotmat(self, x):
        x = x.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose, shape, cam):
        batch = pose.shape[0]
        # transform rot-6d to angle-axis
        pose = self.rot6d_to_rotmat(pose)
        # pose = kornia.geometry.conversions.rotation_matrix_to_angle_axis(pose).reshape(batch, -1)
        pose = torch.cat([pose, torch.zeros((pose.shape[0], 3, 1)).to(pose.device).float()], 2)
        pose_euler = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch, -1)
        # get coordinates from MANO layer
        self.mano_layer = self.mano_layer.to(pose_euler.device)
        mano_out = self.mano_layer(pose_euler, shape)
        mano_vert_cam = mano_out.verts
        mano_joint_cam = mano_out.joints
        coord_xyz = torch.cat((mano_vert_cam, mano_joint_cam), dim=1)
        # root-relative
        coord_xyz = coord_xyz - mano_joint_cam[:, self.root_joint_idx, None]
        # project xy to uv
        coord_uv = orthgonalProj(coord_xyz[:, :, :2].clone(), cam[:, 0:1].unsqueeze(1), cam[:, 1:].unsqueeze(1))
        # normalization
        coord_xyz = coord_xyz / (self.bbox_3d_size / 2)
        coord_uv = coord_uv / (self.input_img_shape[0] // 2) - 1
        # coord_uv = coord_uv / self.input_img_shape[0]
        # coord_uvd = torch.cat((coord_uv, coord_xyz[:, :, 2:3]), dim=2)
        return coord_xyz, coord_uv, pose_euler, shape, cam
    

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y
    

class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x

class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x, valid=None, mask=False):
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x, valid=None, mask=False):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x), valid, mask)
        x = self.ff(x)

        return x
    

class HOT(nn.Module):
    def __init__(self, in_channels, feature_dim, num_hand_kps=21, num_obj=1):
        super(HOT, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_hand_kps = num_hand_kps
        self.num_obj = num_obj
        self.num_tokens = num_hand_kps + num_obj  # 总共 22 个节点
        
        # =======================================================
        # 绝杀优化 1：空间注意力预测 (计算量极小: 128 -> 22)
        # 用来预测 22 个 Token 在 32x32 画面上的权重分布
        # =======================================================
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, self.num_tokens, kernel_size=1),
            nn.BatchNorm2d(self.num_tokens)
        )
        
        # =======================================================
        # 绝杀优化 2：特征升维投影 (计算量极小: 128 -> 512)
        # =======================================================
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.1)
        )
        
        # 身份编码与类别编码 (保持不变)
        self.identity_embedding = nn.Embedding(self.num_tokens, feature_dim)
        self.type_embedding = nn.Embedding(2, feature_dim) 

    def forward(self, x):
        """
        x 形状: (B, 128, 32, 32)
        返回: (B, 22, 512)
        """
        B, C, H, W = x.shape
        device = x.device

        # --- 步骤 1：预测 22 张空间注意力图 ---
        # 这一步网络会学到：第0张图关注手腕区域，第21张图关注物体区域
        attn_maps = self.spatial_attn(x)  # (B, 22, 32, 32)
        attn_maps = attn_maps.view(B, self.num_tokens, -1)  # 展平空间: (B, 22, 1024)
        
        # 在空间维度做 Softmax，确保每张图的权重总和为 1
        attn_maps = F.softmax(attn_maps, dim=-1) 

        # --- 步骤 2：注意力加权提取局部特征 ---
        x_flat = x.view(B, C, -1).transpose(1, 2)  # 展平特征: (B, 1024, 128)
        
        # 核心魔法：矩阵乘法。22个区域分布 乘以 1024个像素特征
        # 瞬间得到 22 个精准的局部特征！
        tokens = torch.bmm(attn_maps, x_flat)  # (B, 22, 128)

        # --- 步骤 3：映射到目标维度 (512) ---
        tokens = self.feature_proj(tokens)  # (B, 22, 512)

        # --- 步骤 4：注入身份与类别编码 ---
        identity_ids = torch.arange(self.num_tokens, dtype=torch.long, device=device)
        identities = self.identity_embedding(identity_ids)  # (22, 512)

        type_ids = torch.zeros(self.num_tokens, dtype=torch.long, device=device)
        type_ids[self.num_hand_kps:] = 1  
        types = self.type_embedding(type_ids)  # (22, 512)

        # 最终融合
        out_tokens = tokens + identities + types

        return out_tokens
    

class GraphConv(nn.Module):
    def __init__(self, num_joint, in_features, out_features, use_dynamic=True):
        super(GraphConv, self).__init__()
        self.use_dynamic = use_dynamic
        self.fc = nn.Linear(in_features, out_features)
        
        # 静态图先验：初始化不再用严苛的 eye，而是加入微小噪声打破对称性
        self.adj = nn.Parameter(torch.eye(num_joint) + torch.randn(num_joint, num_joint) * 0.01)
        
        if self.use_dynamic:
            # 用于计算动态图的特征变换，降低维度减少计算量
            self.theta = nn.Linear(in_features, in_features // 2, bias=False)
            self.phi = nn.Linear(in_features, in_features // 2, bias=False)

    def forward(self, x):
        # x shape: (B, 22, C)
        batch = x.size(0)
        
        # 1. 静态图 (Static Graph Prior)
        # 用 Softmax 保证权重为正且和为 1，彻底杜绝除 0 导致的 NaN
        A_static = F.softmax(self.adj, dim=-1).unsqueeze(0).repeat(batch, 1, 1) # (B, 22, 22)
        
        # 2. 动态图 (Dynamic Graph) - 实例级别自适应
        if self.use_dynamic:
            query = self.theta(x)  # (B, 22, C/2)
            key = self.phi(x)      # (B, 22, C/2)
            # 计算节点间的特征相似度作为动态边权重
            A_dynamic = torch.einsum('bni,bmi->bnm', query, key) # (B, 22, 22)
            A_dynamic = F.softmax(A_dynamic / (query.size(-1) ** 0.5), dim=-1)
            
            # 融合静态结构先验与动态实例特征
            A_hat = A_static + A_dynamic
        else:
            A_hat = A_static
            
        # 3. 聚合与更新
        # 先聚合特征 (A_hat @ x)，再进行线性变换 (FC)
        out = self.fc(torch.matmul(A_hat, x))
        return out


class GraphRegression(nn.Module):
    def __init__(self, node_num, in_dim, out_dim, layer_num=2, last=True):
        super(GraphRegression, self).__init__()
        self.num_node = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = nn.LeakyReLU(0.1)
        self.reg = nn.Sequential()
        
        # 首层归一化
        self.reg.add_module('ln_in', nn.LayerNorm(self.in_dim))
        
        # 修复 Bug：正确使用 f-string，并为每一层 GCN 加入独立的 LayerNorm
        for i in range(layer_num - 1):
            self.reg.add_module(f'gcn_{i}', GraphConv(node_num, self.in_dim, self.in_dim))
            self.reg.add_module(f'ln_{i}', nn.LayerNorm(self.in_dim)) # GCN 极度依赖层间归一化
            self.reg.add_module(f'activate_{i}', self.activation)
            self.reg.add_module(f'dp_{i}', nn.Dropout(0.1))
            
        # 最后一层
        self.reg.add_module(f'gcn_{layer_num-1}', GraphConv(node_num, self.in_dim, self.out_dim))
        if not last:
            self.reg.add_module(f'activate_{layer_num-1}', self.activation)

    def forward(self, graph, shortcut=False):
        in_graph = graph
        out_graph = self.reg(graph)
        if shortcut:
            assert in_graph.shape[2] == out_graph.shape[2], "Shortcut requires in_dim == out_dim"
            return out_graph + in_graph
        else:
            return out_graph
        

class Proj2World(nn.Module):

    def __init__(self, num_hand_joints, num_obj_joints, img_size):
        super(Proj2World, self).__init__()
        self.name = type(self).__name__
        self.num_hand_joints = num_hand_joints
        self.num_obj_joints = num_obj_joints
        self.img_size = img_size

    def forward(self, hand_coord_pred, obj_coord_pred, hand_confidence, obj_confidence, K, T_c2m, batch_size, num_cams):
        W, H = self.img_size
        hand_coord_pred = (hand_coord_pred + 1) / 2  # Normalize to [0, 1]
        hand_coord_im = torch.einsum("bij, j->bij", hand_coord_pred, torch.tensor([W, H]).to(hand_coord_pred.device))  # range 0~W,H
        hand_coord_im = hand_coord_im.view(batch_size, num_cams, self.num_hand_joints, 2)  # (B, N, 21, 2)
        hand_confidence_im = hand_confidence.view(batch_size, num_cams, self.num_hand_joints)  # (B, N, 21)

        ref_hand = batch_triangulate_dlt_torch_confidence(hand_coord_im, K, T_c2m, hand_confidence_im)  # (B, 21, 3)

        obj_coord_pred = (obj_coord_pred + 1) / 2  # Normalize to [0, 1]
        obj_coord_im = torch.einsum("bij, j->bij", obj_coord_pred, torch.tensor([W, H]).to(obj_coord_pred.device))  # range 0~W,H
        obj_coord_im = obj_coord_im.view(batch_size, num_cams, self.num_obj_joints, 2)  # (B, N, 1, 2)
        obj_confidence_im = obj_confidence.view(batch_size, num_cams, self.num_obj_joints)  # (B, N, 1)

        ref_obj = batch_triangulate_dlt_torch_confidence(obj_coord_im, K, T_c2m, obj_confidence_im)  # (B, 1, 3)

        return ref_hand, ref_obj
