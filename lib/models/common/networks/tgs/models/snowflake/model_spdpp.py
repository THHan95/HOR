import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.base import BaseModule
from ...utils.typing import *
from dataclasses import dataclass, field

from pytorch3d.renderer import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from lib.models.common.utils.transform import homoify, dehomoify

from .utils import fps_subsample
from typing import List, Optional

from .utils import MLP_CONV
from .SPD import SPD
from .SPD_crossattn import SPD_crossattn
from .SPD_pp import SPD_pp

SPD_BLOCK = {
    'SPD': SPD,
    'SPD_crossattn': SPD_crossattn,
    'SPD_PP': SPD_pp,
}


def mask_generation(points: Float[Tensor, "B Np 3"],
                      intrinsics: Float[Tensor, "B 3 3"],
                      input_img: Float[Tensor, "B C H W"],
                      raster_point_radius: float = 0.01,  # point size
                      raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                      bin_size: int = 0):
    """
    points: (B, Np, 3)
    """
    B, C, H, W = input_img.shape
    device = intrinsics.device

    cam_R = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
    cam_t = torch.zeros(3).to(device).unsqueeze(0).repeat(B, 1)

    raster_settings = PointsRasterizationSettings(image_size=(H, W), radius=raster_point_radius, points_per_pixel=raster_points_per_pixel, bin_size=bin_size)

    image_size = torch.as_tensor([H, W]).view(1, 2).expand(B, -1).to(device)
    cameras = cameras_from_opencv_projection(cam_R, cam_t, intrinsics, image_size)

    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))

    fragments_idx: Tensor = fragments.idx.long()
    mask = (fragments_idx[..., 0] > -1)

    return mask.float()


def points_projection(points: Float[Tensor, "B Np 3"],
                      intrinsics: Float[Tensor, "B 3 3"],
                      local_features: Float[Tensor, "B C H W"],
                      raster_point_radius: float = 0.0075,  # point size
                      raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                      bin_size: int = 0):
    """
    points: (B, Np, 3)
    """
    B, C, H, W = local_features.shape
    device = local_features.device
    cam_R = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
    cam_t = torch.zeros(3).to(device).unsqueeze(0).repeat(B, 1)

    raster_settings = PointsRasterizationSettings(image_size=(H, W), radius=raster_point_radius, points_per_pixel=raster_points_per_pixel, bin_size=bin_size)
    Np = points.shape[1]
    R = raster_settings.points_per_pixel
    image_size = torch.as_tensor([H, W]).view(1, 2).expand(B, -1).to(device)
    cameras = cameras_from_opencv_projection(cam_R, cam_t, intrinsics, image_size)
    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]
    # Reshape local features to (B, H, W, R, C)
    local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)
    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * Np, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, Np, C)
    return local_features_proj


def points_projection_v2(input_xyz_points, cam_intr, feature_maps):
    """
    Project 3D world points to 2D image coordinates and sample features.

    Args:
        input_xyz_points: (B, Np, 3) - 3D points in world coordinates
        cam_intr: (B, 3, 3) - camera intrinsic matrix
        feature_maps: (B, C, H, W) - CNN feature maps

    Returns:
        sample_feat: (B, Np, C) - sampled features for each point
    """
    input_points = input_xyz_points.clone()
    batch_size = input_points.shape[0]
    xyz = input_points[:, :, :3]  # (B, Np, 3)

    # Project to image plane: P_2d = K @ P_3d
    # cam_intr is (B, 3, 3), xyz is (B, Np, 3)
    # We need to compute: P_2d = K @ P_3d for each point
    homo_xyz_2d = torch.matmul(cam_intr.unsqueeze(1), xyz.unsqueeze(-1)).squeeze(-1)  # (B, Np, 3)
    xyz_2d = (homo_xyz_2d[:, :, :2] / homo_xyz_2d[:, :, [2]]).unsqueeze(2)  # (B, Np, 1, 2)

    # Get feature map dimensions
    _, _, H, W = feature_maps.shape

    # Normalize to [-1, 1] for grid_sample
    uv_2d = xyz_2d / torch.tensor([W, H], device=xyz_2d.device, dtype=xyz_2d.dtype).view(1, 1, 1, 2) * 2 - 1

    # Sample features using grid_sample
    sample_feat = torch.nn.functional.grid_sample(feature_maps, uv_2d, align_corners=False)[:, :, :, 0].transpose(1, 2)  # (B, Np, C)

    # Check validity: points should be within image bounds
    uv_2d_flat = uv_2d.squeeze(2).reshape((-1, 2))
    validity = (uv_2d_flat[:, 0] >= -1.0) & (uv_2d_flat[:, 0] <= 1.0) & (uv_2d_flat[:, 1] >= -1.0) & (uv_2d_flat[:, 1] <= 1.0)
    validity = validity.unsqueeze(1)

    return sample_feat


class Decoder(nn.Module):
    def __init__(self, input_channels=256, dim_feat=512, num_p0=512,
                 radius=1, bounding=True, up_factors=None,
                 SPD_type='SPD',
                 token_type='image_token'
                 ):
        super(Decoder, self).__init__()
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors
        uppers = []
        self.num_p0 = num_p0
        self.mlp_feat_cond = MLP_CONV(in_channel=input_channels,
                                      layer_dims=[dim_feat*2, dim_feat])

        for i, factor in enumerate(up_factors):
            uppers.append(
                SPD_BLOCK[SPD_type](dim_feat=dim_feat, up_factor=factor,
                                    i=i, bounding=bounding, radius=radius))
        self.uppers = nn.ModuleList(uppers)
        self.token_type = token_type

    def calculate_pcl_token(self, pcl_token, up_factor):
        up_token =  F.interpolate(pcl_token, scale_factor=up_factor, mode='nearest')
        return up_token

    def calculate_image_token(self, pcd, hand_pcd, input_image_tokens, cam_intr):
        """
        Args:
            pcd: object points in world coordinates (B, 3, Np)
            hand_pcd: hand points in world coordinates (B, 3, Nh)
            input_image_tokens: CNN feature maps (B, C, H, W)
            cam_intr: camera intrinsic matrix (B, 3, 3)

        Returns:
            local_features_proj: (B, C, Np) - sampled features for object points
            local_features_proj_hand: (B, C, Nh) - sampled features for hand points
        """
        B, C, H, W = input_image_tokens.shape
        device = input_image_tokens.device

        # Convert points from (B, 3, N) to (B, N, 3)
        pcd_xyz = pcd.permute(0, 2, 1).contiguous()  # (B, Np, 3)
        hand_pcd_xyz = hand_pcd.permute(0, 2, 1).contiguous()  # (B, Nh, 3)

        # Project to image plane: P_2d = K @ P_3d
        # cam_intr is (B, 3, 3), xyz is (B, N, 3)
        pcd_proj_2d = torch.matmul(cam_intr.unsqueeze(1), pcd_xyz.unsqueeze(-1)).squeeze(-1)  # (B, Np, 3)
        hand_proj_2d = torch.matmul(cam_intr.unsqueeze(1), hand_pcd_xyz.unsqueeze(-1)).squeeze(-1)  # (B, Nh, 3)

        # Normalize by depth and get 2D coordinates
        pcd_uv = pcd_proj_2d[:, :, :2] / pcd_proj_2d[:, :, [2]]  # (B, Np, 2)
        hand_uv = hand_proj_2d[:, :, :2] / hand_proj_2d[:, :, [2]]  # (B, Nh, 2)

        # Normalize to [-1, 1] for grid_sample
        pcd_uv_norm = pcd_uv / torch.tensor([W, H], device=device, dtype=pcd_uv.dtype).view(1, 1, 2) * 2 - 1
        hand_uv_norm = hand_uv / torch.tensor([W, H], device=device, dtype=hand_uv.dtype).view(1, 1, 2) * 2 - 1

        # Add spatial dimension for grid_sample: (B, N, 2) -> (B, N, 1, 2)
        pcd_uv_norm = pcd_uv_norm.unsqueeze(2)  # (B, Np, 1, 2)
        hand_uv_norm = hand_uv_norm.unsqueeze(2)  # (B, Nh, 1, 2)

        # Sample features using grid_sample
        pcd_feats = F.grid_sample(input_image_tokens, pcd_uv_norm, align_corners=False)  # (B, C, Np, 1)
        hand_feats = F.grid_sample(input_image_tokens, hand_uv_norm, align_corners=False)  # (B, C, Nh, 1)

        # Remove spatial dimensions and reshape to (B, C, N)
        local_features_proj = pcd_feats.squeeze(-1)  # (B, C, Np)
        local_features_proj_hand = hand_feats.squeeze(-1)  # (B, C, Nh)

        return local_features_proj, local_features_proj_hand

    def forward(self, x):
        """
        Args:
            x['points']: object point cloud in world coordinates (B, num_p0, 3)
            x['hand_points']: hand vertices in world coordinates (B, num_hand, 3)
            x['mlvl_feat']: CNN feature maps (B, C, H, W)
            x['cam_intr']: camera intrinsic matrix (B, 3, 3)
        """
        points = x['points']  # (B, num_p0, 3) - world coordinates
        hand_points = x['hand_points']  # (B, num_hand, 3) - world coordinates

        if self.token_type == 'pcl_token':
            feat_cond = x['pcl_token']
            feat_cond = self.mlp_feat_cond(feat_cond)
        elif self.token_type == 'image_token':
            feat_cond = x['mlvl_feat']  # (B, C, H, W) - CNN feature maps, no MLP needed

        arr_pcd = []
        feat_prev = None

        # Convert to (B, 3, num_p0) format for processing
        pcd = torch.permute(points, (0, 2, 1)).contiguous()
        hand_pcd = torch.permute(hand_points, (0, 2, 1)).contiguous()

        pcl_up_scale = 1
        for upper in self.uppers:
            if self.token_type == 'pcl_token':
                up_cond = self.calculate_pcl_token(feat_cond, pcl_up_scale)
                up_cond_hand = up_cond  # Same condition for hand
                pcl_up_scale *= upper.up_factor
            elif self.token_type == 'image_token':
                # Project both object and hand points to image features
                up_cond, up_cond_hand = self.calculate_image_token(pcd, hand_pcd, feat_cond, x['cam_intr'])

            pcd, feat_prev = upper(pcd, hand_pcd, up_cond_hand, up_cond, feat_prev)
            points = torch.permute(pcd, (0, 2, 1)).contiguous()
            arr_pcd.append(points)

        return arr_pcd


# class SnowflakeModelSPDPP(BaseModule):
#     """
#     apply PC^2 / PCL token to decoder
#     """
#     @dataclass
#     class Config(BaseModule.Config):
#         input_channels: int = 1152
#         dim_feat: int = 128
#         num_p0: int = 512
#         radius: float = 1
#         bounding: bool = True
#         use_fps: bool = True
#         up_factors: List[int] = field(default_factory=lambda: [2, 2])
#         image_full_token_cond: bool = False
#         SPD_type: str = 'SPD_PP'
#         token_type: str = 'pcl_token'
#     cfg: Config

#     def configure(self) -> None:
#         super().configure()
#         self.decoder = Decoder(input_channels=self.cfg.input_channels,
#                                dim_feat=self.cfg.dim_feat, num_p0=self.cfg.num_p0,
#                                radius=self.cfg.radius, up_factors=self.cfg.up_factors, bounding=self.cfg.bounding,
#                                SPD_type=self.cfg.SPD_type,
#                                token_type=self.cfg.token_type
#                                )

#     def forward(self, x):
#         results = self.decoder(x)
#         return results

class SnowflakeModelSPDPP(nn.Module):
    """
    Point cloud upsampler for object reconstruction.
    Now supports CNN feature maps (mlvl_feat) instead of image tokens.
    Both object and hand points are in world coordinates.
    """
    def __init__(
        self,
        input_channels: int = 256,  # CNN feature channels (e.g., from ResNet backbone)
        dim_feat: int = 128,
        num_p0: int = 2048,
        radius: float = 1,
        bounding: bool = True,
        use_fps: bool = True,
        up_factors: Optional[List[int]] = None,
        image_full_token_cond: bool = False,
        SPD_type: str = 'SPD_PP',
        token_type: str = 'image_token'
    ):
        super(SnowflakeModelSPDPP, self).__init__()

        if up_factors is None:
            up_factors = [2, 4]

        self.input_channels = input_channels
        self.dim_feat = dim_feat
        self.num_p0 = num_p0
        self.radius = radius
        self.bounding = bounding
        self.use_fps = use_fps
        self.up_factors = up_factors
        self.image_full_token_cond = image_full_token_cond
        self.SPD_type = SPD_type
        self.token_type = token_type

        self.decoder = Decoder(
            input_channels=self.input_channels,
            dim_feat=self.dim_feat,
            num_p0=self.num_p0,
            radius=self.radius,
            up_factors=self.up_factors,
            bounding=self.bounding,
            SPD_type=self.SPD_type,
            token_type=self.token_type
        )

    def forward(self, x):
        """
        Args:
            x['points']: object point cloud in world coordinates (B, num_p0, 3)
            x['hand_points']: hand vertices in world coordinates (B, num_hand, 3)
            x['mlvl_feat']: CNN feature maps (B, C, H, W)
            x['cam_intr']: camera intrinsic matrix (B, 3, 3)

        Returns:
            list of upsampled point clouds at different scales
        """
        results = self.decoder(x)
        return results
