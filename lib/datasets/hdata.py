import traceback
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from ..utils.builder import build_transform
from ..utils.logger import logger
from ..utils.misc import CONST
from ..utils.transform import fit_ortho_param, ortho_project, rotmat_to_rot6d

# https://github.com/lmb-freiburg/freihand/blob/master/utils/mano_utils.py
kpId2vertices = {
    4: [744],  #ThumbT
    8: [320],  #IndexT
    12: [443],  #MiddleT
    16: [555],  #RingT
    20: [672]  #PinkT
}

OPENPOSE_JOINTS_NAME = [
    "loc_bn_palm_L", "loc_bn_thumb_L_01", "loc_bn_thumb_L_02", "loc_bn_thumb_L_03", "loc_bn_thumb_L_04",
    "loc_bn_index_L_01", "loc_bn_index_L_02", "loc_bn_index_L_03", "loc_bn_index_L_04", "loc_bn_mid_L_01",
    "loc_bn_mid_L_02", "loc_bn_mid_L_03", "loc_bn_mid_L_04", "loc_bn_ring_L_01", "loc_bn_ring_L_02", "loc_bn_ring_L_03",
    "loc_bn_ring_L_04", "loc_bn_pinky_L_01", "loc_bn_pinky_L_02", "loc_bn_pinky_L_03", "loc_bn_pinky_L_04"
]
OPENPOSE_JOINTS_NAME_TO_ID = {w: i for i, w in enumerate(OPENPOSE_JOINTS_NAME)}


class HDataset(torch.utils.data.Dataset, ABC):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = "HDataset"
        self.data_mode = cfg.DATA_MODE
        self.data_root = cfg.DATA_ROOT
        self.data_split = cfg.DATA_SPLIT
        self.use_cache = cfg.DATA_PRESET.USE_CACHE
        self.bbox_expand_ratio = float(cfg.DATA_PRESET.BBOX_EXPAND_RATIO)
        self.image_size = cfg.DATA_PRESET.IMAGE_SIZE  # (W, H)
        self.data_preset = cfg.DATA_PRESET

        self.center_idx = int(cfg.DATA_PRESET.CENTER_IDX)
        self.sides = CONST.SIDE
        self.njoints = CONST.NUM_JOINTS
        self.current_stage_name = "stage2"
        self.stage12_bbox_expand_ratio = float(cfg.DATA_PRESET.get("STAGE12_BBOX_EXPAND_RATIO", 1.5))
        self.stage12_use_hand_obj_bbox = bool(cfg.DATA_PRESET.get("STAGE12_USE_HAND_OBJ_BBOX", False))
        # Keep the old bbox key as a fallback so existing data presets do not break.
        self.stage2_hand_obj_scale = float(
            cfg.DATA_PRESET.get("STAGE2_HAND_OBJ_SCALE", cfg.DATA_PRESET.get("STAGE3_HAND_OBJ_SCALE", 1.25))
        )
        self.stage12_hand_obj_scale = float(
            cfg.DATA_PRESET.get("STAGE12_HAND_OBJ_SCALE", self.stage2_hand_obj_scale)
        )

        self.transform = build_transform(cfg=cfg.TRANSFORM,
                                         data_preset=self.data_preset,
                                         is_train="train" in self.data_split)
        logger.info(f"Initialized abstract class: {self.name}")

    def __len__(self):
        return len(self)

    @staticmethod
    def flip_2d(annot_2d, centerX):
        annot_2d = annot_2d.copy()
        annot_2d[:, 0] = centerX - annot_2d[:, 0]
        return annot_2d

    @staticmethod
    def flip_3d(annot_3d):
        annot_3d = annot_3d.copy()
        annot_3d[:, 0] = -annot_3d[:, 0]
        return annot_3d

    @staticmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_image(self, idx):
        pass

    @abstractmethod
    def get_image_mask(self, idx):
        pass

    @abstractmethod
    def get_image_path(self, idx):
        pass

    @abstractmethod
    def get_joints_3d(self, idx):
        pass

    @abstractmethod
    def get_verts_3d(self, idx):
        pass

    @abstractmethod
    def get_verts_uvd(self, idx):
        pass

    @abstractmethod
    def get_bone_scale(self, idx):
        pass

    @abstractmethod
    def get_joints_2d(self, idx):
        pass

    @abstractmethod
    def get_joints_uvd(self, idx):
        pass

    @abstractmethod
    def get_cam_intr(self, idx):
        pass

    @abstractmethod
    def get_cam_center(self, idx):
        pass

    @abstractmethod
    def get_sides(self, idx):
        pass

    @abstractmethod
    def get_bbox_center_scale(self, idx):
        pass

    @abstractmethod
    def get_sample_identifier(self, idx):
        pass

    @abstractmethod
    def get_mano_pose(self, idx):
        pass

    @abstractmethod
    def get_mano_shape(self, idx):
        pass

    @abstractmethod
    def get_rawimage_size(self, idx):
        pass

    # 1. 默认接口
    def get_obj_corners(self, idx):
        return None
    
    def get_obj_pose_and_points(self, idx):
        return {"sparse": None, "dense": None}

    def get_obj_info(self, idx):
        return None, None
    
    def get_bbox_hand_obj(self, idx, scale_factor):
        return None, None

    def set_stage(self, stage_name: str):
        self.current_stage_name = stage_name
        if hasattr(self.transform, "set_stage"):
            self.transform.set_stage(stage_name)

    def _get_3d_bbox(self, idx):
        if self.current_stage_name in ["stage1", "stage2"]:
            if self.stage12_use_hand_obj_bbox:
                bbox_center, bbox_scale = self.get_bbox_hand_obj(idx, scale_factor=self.stage12_hand_obj_scale)
                if bbox_center is not None and bbox_scale is not None:
                    return bbox_center, bbox_scale
            bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
            bbox_scale = bbox_scale * self.stage12_bbox_expand_ratio
            return bbox_center, bbox_scale
        return self.get_bbox_hand_obj(idx, scale_factor=self.stage2_hand_obj_scale)

    # visible in raw image
    def get_joints_2d_vis(self, joints_2d=None, raw_size=None, **kwargs):
        joints_vis = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < raw_size[0])) & \
                     ((joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < raw_size[1]))
        return joints_vis.astype(np.float32)

    def get_obj_center_2d_vis(self, obj_center_2d=None, raw_size=None, **kwargs):
        """Check if object center is visible in raw image"""
        if obj_center_2d is None or raw_size is None:
            return None
        obj_center_vis = ((obj_center_2d[:, 0] >= 0) & (obj_center_2d[:, 0] < raw_size[0])) & \
                         ((obj_center_2d[:, 1] >= 0) & (obj_center_2d[:, 1] < raw_size[1]))
        return obj_center_vis.astype(np.float32)

    def getitem_2d(self, idx):
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        joints_2d = self.get_joints_2d(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        flip_hand = True if hand_side != self.sides else False

        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_2d = self.flip_2d(joints_2d, raw_size[0])
            image = image[:, ::-1, :]

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "joints_2d": joints_2d,
            "joints_vis": joints_vis,
            "image_path": image_path,
            "raw_size": raw_size,
        }

        if self.data_preset.get("WITH_MASK", False):
            image_mask = self.get_image_mask(idx)
            if flip_hand:
                image_mask = image_mask[:, ::-1]
            label["image_mask"] = image_mask

        results = self.transform(image, label)
        return results

    def getitem_uvd(self, idx):
        # Support FreiHAND, HO3D, DexYCB, YT3D, TMANO,
        hand_side = self.get_sides(idx)

        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        verts_uvd = self.get_verts_uvd(idx)
        joints_uvd = self.get_joints_uvd(idx)
        joints_2d = self.get_joints_2d(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        flip_hand = True if hand_side != self.sides else False

        ## Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_2d = self.flip_2d(joints_2d, raw_size[0])
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            image = image[:, ::-1, :]

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "joints_2d": joints_2d,
            "verts_uvd": verts_uvd,
            "joints_uvd": joints_uvd,
            "joints_vis": joints_vis,
            "image_path": image_path,
            "raw_size": raw_size,
        }

        if self.data_preset.get("WITH_MASK", False):
            image_mask = self.get_image_mask(idx)
            if flip_hand:
                image_mask = image_mask[:, ::-1]
            label["image_mask"] = image_mask

        results = self.transform(image, label)
        return results

    def getitem_3d(self, idx):
        # Support FreiHAND, HO3D, DexYCB
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self._get_3d_bbox(idx)
        cam_intr = self.get_cam_intr(idx)
        cam_center = self.get_cam_center(idx)
        joints_3d = self.get_joints_3d(idx)
        verts_3d = self.get_verts_3d(idx)
        joints_2d = self.get_joints_2d(idx)
        verts_uvd = self.get_verts_uvd(idx)
        joints_uvd = self.get_joints_uvd(idx)

        # 🌟 1. 替换为接住新的 6D 位姿与点云字典
        obj_data = self.get_obj_pose_and_points(idx)

        obj_id = obj_data["obj_id"]
        pc_sparse_rest = obj_data["sparse_rest"]
        pc_sparse_cam = obj_data["sparse_cam"]
        pc_dense_cam = obj_data["dense_cam"]
        R_label = obj_data["R_label"]           # (3, 3) 相机系下的绝对旋转
        t_label_rel = obj_data["t_label_rel"]   # (3,) 相对于手腕的平移
        rot6d_label = obj_data["rot6d_label"]   # (6,) 

        # 🚨 修复 1：提供全零占位符，消灭 None
        if pc_sparse_cam is not None:
            obj_center_3d = np.mean(pc_sparse_cam, axis=0, keepdims=True).astype(np.float32)
        else:
            obj_center_3d = np.zeros((1, 3), dtype=np.float32)

        _, obj_transform = self.get_obj_info(idx)
        
        # 🚨 修复 2：绝对不能给 None，给安全的占位符
        if obj_transform is not None:
            obj_trans_3d = obj_transform[:3, 3:4].T
        else:
            obj_transform = np.eye(4, dtype=np.float32)
            obj_trans_3d = np.zeros((1, 3), dtype=np.float32)

        image_path = self.get_image_path(idx)
        mano_pose = self.get_mano_pose(idx)
        mano_shape = self.get_mano_shape(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)
        flip_hand = True if hand_side != self.sides else False

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            joints_2d = self.flip_2d(joints_2d, centerX=raw_size[0])
            image = image[:, ::-1, :]
            
            # 翻转点云和中心 (前提是它们存在)
            if pc_sparse_cam is not None:
                pc_sparse_cam = self.flip_3d(pc_sparse_cam)
                pc_dense_cam = self.flip_3d(pc_dense_cam)
                
            obj_center_3d = self.flip_3d(obj_center_3d)
            
            # 翻转相对平移 t_rel
            t_label_rel = self.flip_3d(t_label_rel.reshape(1, 3)).flatten()

            # 翻转 3x3 旋转矩阵
            flip_mat_3x3 = np.array([[-1, 0, 0],
                                     [ 0, 1, 0],
                                     [ 0, 0, 1]], dtype=np.float32)
            R_label = flip_mat_3x3 @ R_label @ flip_mat_3x3
            
            # 重新生成 6D 表征
            rot6d_label = rotmat_to_rot6d(torch.from_numpy(R_label).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
            
            if pc_sparse_rest is not None:
                pc_sparse_rest = self.flip_3d(pc_sparse_rest)

            # 翻转 obj_transform
            flip_mat = np.array([[-1, 0, 0, 0],
                                 [ 0, 1, 0, 0],
                                 [ 0, 0, 1, 0],
                                 [ 0, 0, 0, 1]], dtype=np.float32)
            obj_transform = flip_mat @ obj_transform @ flip_mat
            obj_trans_3d = self.flip_3d(obj_trans_3d)

        label = {
            "idx": idx,
            "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_intr": cam_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "mano_pose": mano_pose,
            "mano_shape": mano_shape,
            "image_path": image_path,
            "raw_size": raw_size,
            "obj_id": obj_id,
            "obj_pc_sparse_rest": pc_sparse_rest,
            "obj_pc_sparse": pc_sparse_cam,       
            "obj_pc_dense": pc_dense_cam,
            "obj_center_3d": obj_center_3d,
            "obj_transform": obj_transform,
            "obj_trans_3d": obj_trans_3d,
            "R_label": R_label,
            "t_label_rel": t_label_rel,
            "rot6d_label": rot6d_label,
        }
        
        if self.data_preset.get("WITH_MASK", False):
            image_mask = self.get_image_mask(idx)
            if flip_hand:
                image_mask = image_mask[:, ::-1]
            label["image_mask"] = image_mask

        results = self.transform(image, label)
        results.update(label)
        
        # 🌟 🚨 终极护盾：在送给 DataLoader 前，强制踢掉所有意外产生的 None
        results = {k: v for k, v in results.items() if v is not None}
        
        return results

    def getitem_uvd_ortho(self, idx):
        # idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle

        joints_3d = self.get_joints_3d(idx)
        joints_2d = self.get_joints_2d(idx)
        joints_uvd = np.concatenate((joints_2d, joints_3d[:, 2:]), axis=1)

        CID = self.cfg.DATA_PRESET.CENTER_IDX
        ortho_intr = fit_ortho_param((joints_3d - joints_3d[CID, :]), joints_2d)

        verts_3d = self.get_verts_3d(idx)
        verts_uv = ortho_project(verts_3d - joints_3d[CID, :], ortho_intr)
        verts_d = verts_3d[:, 2:]
        verts_uvd = np.concatenate((verts_uv, verts_d), axis=1)

        image_path = self.get_image_path(idx)
        mano_pose = self.get_mano_pose(idx)
        mano_shape = self.get_mano_shape(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)

        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)
        flip_hand = True if hand_side != self.sides else False

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            joints_2d = self.flip_2d(joints_2d, centerX=raw_size[0])
            image = image[:, ::-1, :]

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "ortho_intr": ortho_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "mano_pose": mano_pose,
            "mano_shape": mano_shape,
            "image_path": image_path,
            "raw_size": raw_size,
        }

        if self.data_preset.get("WITH_MASK", False):
            image_mask = self.get_image_mask(idx)
            if flip_hand:
                image_mask = image_mask[:, ::-1]
            label["image_mask"] = image_mask

        results = self.transform(image, label)

        return results

    def __getitem__(self, idx):
        if self.data_mode not in ["2D", "3D", "UVD", "UVD_ortho"]:
            raise NotImplementedError(f"Unknown data mode: {self.data_mode}")

        if self.data_mode == "2D":
            return self.getitem_2d(idx)
        elif self.data_mode == "UVD":
            return self.getitem_uvd(idx)
        elif self.data_mode == "3D":
            return self.getitem_3d(idx)
        elif self.data_mode == "UVD_ortho":
            return self.getitem_uvd_ortho(idx)
