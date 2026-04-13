import math
import random
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from pytorch3d.transforms import (axis_angle_to_matrix, axis_angle_to_quaternion, euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_axis_angle, quaternion_to_matrix, rotation_6d_to_matrix)

from .heatmap import generate_heatmap
from .logger import logger
from .misc import CONST
from .builder import TRANSFORM
from .config import CN


class RandomOcclusion:
    """Add random occlusion based on occlusion probability.

    Args:
        occlusion_prob (float): probability of the image having
        occlusion. Default: 0.5
    """

    def __init__(self, occlusion_prob=0.5):
        self.occlusion_prob = occlusion_prob

    def __call__(self, results):

        if np.random.rand() > self.occlusion_prob:
            return results

        xmin, ymin, xmax, ymax = results["bbox"]
        imgwidth = results["width"]
        imgheight = results["height"]
        img = results["image"]

        area_min = 0.0
        area_max = 0.2
        synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

        ratio_min = 0.5
        ratio_max = 1 / 0.5
        synth_ratio = random.random() * (ratio_max - ratio_min) + ratio_min

        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        if (synth_xmin >= 0 and synth_ymin >= 0 and \
            synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imgheight):

            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            img[synth_ymin:synth_ymin + synth_h,
                synth_xmin:synth_xmin + synth_w, :] = (np.random.rand(synth_h, synth_w, 3) * 255)

        results["image"] = img
        return results


@TRANSFORM.register_module()
class SimpleTransform2D:

    def __init__(self, cfg: CN) -> None:
        super().__init__()
        self._output_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._train = cfg.IS_TRAIN
        self._aug = cfg.AUG

        self._center_jit_factor = cfg.get("CENTER_JIT", 0 if self._aug else 0)
        self._scale_jit_factor = cfg.get("SCALE_JIT", 0.04 if self._aug else 0)
        self._color_jit_factor = cfg.get("COLOR_JIT", 0.3 if self._aug else 0)
        self._rot_jit_factor = cfg.get("ROT_JIT", 10 if self._aug else 0)
        self._rot_prob = cfg.get("ROT_PROB", 1.0 if self._aug else 0)
        self._occlusion = cfg.get("OCCLUSION", True if self._aug else False)
        self._occlusion_prob = cfg.get("OCCLUSION_PROB", 0.1 if self._aug else 0)
        self._base_scale_jit_factor = self._scale_jit_factor
        self._base_rot_jit_factor = self._rot_jit_factor
        self._base_rot_prob = self._rot_prob
        self._base_occlusion = self._occlusion
        self._base_occlusion_prob = self._occlusion_prob
        self._stage12_scale_jit_factor = cfg.get("STAGE12_SCALE_JIT", min(self._base_scale_jit_factor, 0.02))
        self._stage12_rot_jit_factor = cfg.get("STAGE12_ROT_JIT", 5 if self._aug else 0)
        self._stage12_rot_prob = cfg.get("STAGE12_ROT_PROB", min(self._base_rot_prob, 0.3))
        self._stage12_occlusion = cfg.get("STAGE12_OCCLUSION", False)
        self._stage12_occlusion_prob = cfg.get("STAGE12_OCCLUSION_PROB", 0.0)
        self._min_joints_after_crop = int(cfg.get("MIN_JOINTS_AFTER_CROP", 3))
        self._min_joints_in_raw = int(cfg.get("MIN_JOINTS_IN_RAW", 3))
        self.current_stage_name = "stage2"

        self._with_heatmap = cfg.DATA_PRESET.get("WITH_HEATMAP", False)
        self._with_mask = cfg.DATA_PRESET.get("WITH_MASK", False)
        self._heatmap_size = cfg.DATA_PRESET.get("HEATMAP_SIZE", (64, 64))
        self._heatmap_sigma = cfg.DATA_PRESET.get("HEATMAP_SIGMA", 2.0)
        self._with_obj_occ = cfg.DATA_PRESET.get("WITH_OBJ_OCC", self._with_heatmap)
        self._obj_occ_size = cfg.DATA_PRESET.get("OBJ_OCC_SIZE", self._heatmap_size)
        self._obj_occ_radius = int(cfg.DATA_PRESET.get("OBJ_OCC_RADIUS", 1))
        self._obj_occ_blur_sigma = float(cfg.DATA_PRESET.get("OBJ_OCC_BLUR_SIGMA", 1.0))
        self._mask_scale_to_heatmap = cfg.DATA_PRESET.get("MASK_SCALE_TO_HEATMAP", False)

        if self._occlusion:
            self.occlusion_op = RandomOcclusion(self._occlusion_prob)

    def _generate_target_heatmap(self, coords_2d, visibility=None):
        coords_2d = np.asarray(coords_2d, dtype=np.float32)
        num_points = coords_2d.shape[0]
        target_heatmap = np.zeros((num_points, self._heatmap_size[1], self._heatmap_size[0]), dtype=np.float32)
        imsize = np.array(self._output_size, dtype=np.float32)
        hmsize = np.array(self._heatmap_size, dtype=np.float32)

        for i in range(num_points):
            if visibility is not None and visibility[i] <= 0:
                continue
            coord_hm = ((coords_2d[i] / imsize) * hmsize).astype(np.int32)
            target_heatmap[i], _ = generate_heatmap(target_heatmap[i], coord_hm, self._heatmap_sigma)

        return target_heatmap

    def _generate_point_occupancy_map(self, coords_2d, depth=None):
        occ_h, occ_w = self._obj_occ_size[1], self._obj_occ_size[0]
        occupancy = np.zeros((occ_h, occ_w), dtype=np.float32)
        if coords_2d is None:
            return occupancy

        coords_2d = np.asarray(coords_2d, dtype=np.float32)
        if coords_2d.size == 0:
            return occupancy

        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32).reshape(-1)
            valid = depth > 1e-6
            coords_2d = coords_2d[valid]
        if coords_2d.size == 0:
            return occupancy

        imsize = np.array(self._output_size, dtype=np.float32)
        occsize = np.array(self._obj_occ_size, dtype=np.float32)
        coords_occ = (coords_2d / imsize) * occsize
        inside = (
            (coords_occ[:, 0] >= 0.0)
            & (coords_occ[:, 0] < occ_w)
            & (coords_occ[:, 1] >= 0.0)
            & (coords_occ[:, 1] < occ_h)
        )
        coords_occ = coords_occ[inside]
        if coords_occ.size == 0:
            return occupancy

        radius = max(1, self._obj_occ_radius)
        for x_f, y_f in coords_occ:
            x_i = int(np.clip(round(float(x_f)), 0, occ_w - 1))
            y_i = int(np.clip(round(float(y_f)), 0, occ_h - 1))
            cv2.circle(occupancy, (x_i, y_i), radius=radius, color=1.0, thickness=-1)

        if self._obj_occ_blur_sigma > 0:
            occupancy = cv2.GaussianBlur(
                occupancy,
                ksize=(0, 0),
                sigmaX=self._obj_occ_blur_sigma,
                sigmaY=self._obj_occ_blur_sigma,
            )

        max_value = float(occupancy.max())
        if max_value > 0:
            occupancy /= max_value
        return occupancy.astype(np.float32)

    def set_stage(self, stage_name: str):
        self.current_stage_name = stage_name
        if not self._aug:
            return

        if stage_name == "stage1":
            self._scale_jit_factor = self._stage12_scale_jit_factor
            self._rot_jit_factor = self._stage12_rot_jit_factor
            self._rot_prob = self._stage12_rot_prob
            self._occlusion = self._stage12_occlusion
            self._occlusion_prob = self._stage12_occlusion_prob
        else:
            self._scale_jit_factor = self._base_scale_jit_factor
            self._rot_jit_factor = self._base_rot_jit_factor
            self._rot_prob = self._base_rot_prob
            self._occlusion = self._base_occlusion
            self._occlusion_prob = self._base_occlusion_prob

        self.occlusion_op = RandomOcclusion(self._occlusion_prob)

    def __call__(self, image, label):
        if self._aug:
            cf = self._center_jit_factor
            sf = self._scale_jit_factor
            rf = self._rot_jit_factor
            c_factor = np.clip(np.random.normal(loc=0, scale=cf, size=2), -3 * cf, 3 * cf)
            bbox_center = label["bbox_center"] + c_factor * label["bbox_scale"]
            s_factor = np.clip(np.random.normal(loc=1, scale=sf), 1 - 3 * sf, 1 + 3 * sf)
            bbox_scale = label["bbox_scale"] * s_factor

            r_factor = np.clip(np.random.normal(loc=0, scale=rf), -3 * rf, 3 * rf)
            rot = np.deg2rad(r_factor) if np.random.rand() <= self._rot_prob else 0.0
            if self._occlusion:  # apply occlu
                occlu_inp = {}
                occlu_inp["bbox"] = center_scale_to_box(bbox_center, bbox_scale)
                occlu_inp["width"], occlu_inp["height"] = image.shape[1], image.shape[0]
                occlu_inp["image"] = image
                image = self.occlusion_op(occlu_inp)["image"]
        else:
            bbox_scale = label["bbox_scale"]
            bbox_center = label["bbox_center"]
            rot = 0.0

        rot_mat3d = _construct_rotation_matrix(rot)
        affine = _affine_transform(center=bbox_center, scale=bbox_scale, out_res=self._output_size, rot=rot)

        target_joints_2d = _transform_coords(label["joints_2d"], affine).astype(np.float32)

        jv = label["joints_vis"]
        if not self._train:
            target_joints_vis = np.full(CONST.NUM_JOINTS, 1.0, dtype=np.float32)
        elif jv.sum() < self._min_joints_in_raw:
            target_joints_vis = np.full(CONST.NUM_JOINTS, 0.0, dtype=np.float32)
        else:
            tj2d = target_joints_2d
            target_joints_vis = (((tj2d[:, 0] >= 0) & (tj2d[:, 0] < self._output_size[0])) &
                                 ((tj2d[:, 1] >= 0) & (tj2d[:, 1] < self._output_size[1]))).astype(np.float32)
            if target_joints_vis.sum() < self._min_joints_after_crop:
                target_joints_vis = np.full(CONST.NUM_JOINTS, 0.0, dtype=np.float32)

        affine_2x3 = affine[:2, :]
        image = cv2.warpAffine(image,
                               affine_2x3, (int(self._output_size[0]), int(self._output_size[1])),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)

        if self._aug:
            c_high = 1 + self._color_jit_factor
            c_low = 1 - self._color_jit_factor
            image[:, :, 0] = np.clip(image[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            image[:, :, 1] = np.clip(image[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        image_np = image
        image = tvF.to_tensor(image)
        assert image.shape[0] == 3
        image = tvF.normalize(image, [0.5, 0.5, 0.5], [1, 1, 1])

        results = {
            "rot_rad": rot,
            "rot_mat3d": rot_mat3d,
            "affine": affine,
            "image": image,
            "target_bbox_center": bbox_center,
            "target_bbox_scale": bbox_scale,
            "target_joints_2d": target_joints_2d,
            "target_joints_vis": target_joints_vis,
            "image_path": label["image_path"]
        }

        if self._with_mask:
            mask = cv2.warpAffine(label["image_mask"],
                                  affine_2x3, (int(self._output_size[0]), int(self._output_size[1])),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            mask = mask.astype(np.float32) / 255.0
            results["mask"] = mask

        if self._with_heatmap:
            results["target_joints_heatmap"] = self._generate_target_heatmap(target_joints_2d, target_joints_vis)
            if self._mask_scale_to_heatmap:
                mask = cv2.resize(results["mask"], self._heatmap_size, interpolation=cv2.INTER_LINEAR)
                results["mask"] = mask

        return results


@TRANSFORM.register_module()
class SimpleTransformUVD(SimpleTransform2D):

    def __init__(self, cfg: CN) -> None:
        super().__init__(cfg)
        self._center_idx = cfg.DATA_PRESET.CENTER_IDX

    def __call__(self, image, label):
        results = super().__call__(image, label)
        raw_size = label["raw_size"]
        affine = results["affine"]
        joints_uvd = label["joints_uvd"]
        verts_uvd = label["verts_uvd"]

        joints_uv = joints_uvd[:, :2]  # (21, 2)
        verts_uv = verts_uvd[:, :2]  # (778, 2)
        joints_uv = _transform_coords(joints_uv, affine).astype(np.float32)
        verts_uv = _transform_coords(verts_uv, affine).astype(np.float32)

        joints_d = joints_uvd[:, 2:]  # (21, 1)
        verts_d = verts_uvd[:, 2:]  # (778, 1)
        root_joint_d = joints_d[self._center_idx].copy()  # (1)

        joints_d_rel = joints_d - root_joint_d
        verts_d_rel = verts_d - root_joint_d

        # tj_uv = joints_uv / np.array([self._output_size[0], self._output_size[1]])  # (21, 2)
        tj_uv = joints_uv / (np.array([self._output_size[0], self._output_size[1]]) / 2) - 1  # (21, 2)
        tj_d = 0.5 + (joints_d_rel[:, :1] / CONST.UVD_DEPTH_RANGE)  # (21, 1)
        target_joints_uvd = np.concatenate([tj_uv, tj_d], axis=1).astype(np.float32)

        # tv_uv = verts_uv / np.array([self._output_size[0], self._output_size[1]])  # (778, 2)
        tv_uv = verts_uv / (np.array([self._output_size[0], self._output_size[1]]) / 2) - 1  # (778, 2)
        tv_d = 0.5 + (verts_d_rel[:, :1] / CONST.UVD_DEPTH_RANGE)  # (21, 1)
        target_verts_uvd = np.concatenate([tv_uv, tv_d], axis=1).astype(np.float32)

        results["target_root_d"] = root_joint_d
        results["target_joints_uvd"] = target_joints_uvd
        results["target_verts_uvd"] = target_verts_uvd

        return results


@TRANSFORM.register_module()
class SimpleTransform3DMultiView(SimpleTransformUVD):

    def __init__(self, cfg: CN) -> None:
        super().__init__(cfg)

    def __call__(self, image, label):
        results = super().__call__(image, label)
        rot = results["rot_rad"]
        rot_mat = results["rot_mat3d"]
        center = results["target_bbox_center"]
        scale = results["target_bbox_scale"]
        cc = label["cam_center"]
        affine_postrot = _affine_transform_post_rot(center=center,
                                                    scale=scale,
                                                    optical_center=cc,
                                                    out_res=self._output_size,
                                                    rot=rot)

        target_cam_intr = affine_postrot.dot(label["cam_intr"])
        
        # 提前安全获取，避免 get() 报错
        target_joints_3d_no_rot = label.get("joints_3d")
        target_verts_3d_no_rot = label.get("verts_3d")
        target_obj_sparse_no_rot = label.get("obj_pc_sparse")
        target_obj_dense_no_rot = label.get("obj_pc_dense")
        target_obj_center_3d_no_rot = label.get("obj_center_3d")
        target_obj_trans_3d_no_rot = label.get("obj_trans_3d")

        target_joints_3d = rot_mat.dot(label["joints_3d"].transpose(1, 0)).transpose()
        target_verts_3d = rot_mat.dot(label["verts_3d"].transpose(1, 0)).transpose()
        target_root_joint = target_joints_3d[self._center_idx]
        target_joints_3d_rel = target_joints_3d - target_root_joint
        target_verts_3d_rel = target_verts_3d - target_root_joint
        
        if label.get("obj_pc_sparse") is not None:
            # 旋转绝对点云 (相机系)
            target_obj_pc_sparse = rot_mat.dot(label["obj_pc_sparse"].transpose(1, 0)).transpose()
            target_obj_pc_dense = rot_mat.dot(label["obj_pc_dense"].transpose(1, 0)).transpose()
            results["target_obj_pc_sparse"] = target_obj_pc_sparse
            results["target_obj_pc_dense"] = target_obj_pc_dense
            if self._with_obj_occ:
                obj_sparse_2d_h = target_cam_intr.dot(target_obj_pc_sparse.T).T
                obj_sparse_z = obj_sparse_2d_h[:, 2:3]
                obj_sparse_z[np.abs(obj_sparse_z) < 1e-6] = 1e-6
                obj_sparse_uv = np.concatenate(
                    [obj_sparse_2d_h[:, 0:1] / obj_sparse_z, obj_sparse_2d_h[:, 1:2] / obj_sparse_z],
                    axis=-1,
                ).astype(np.float32)
                results["target_obj_occupancy"] = self._generate_point_occupancy_map(
                    obj_sparse_uv,
                    depth=target_obj_pc_sparse[:, 2],
                )[None, ...]

        # 🌟 核心：处理 6D Pose 标签的数据增强旋转
        if label.get("R_label") is not None:
            # 1. 旋转相对平移向量 (3x3 @ 3x1)
            t_rel_gt = label["t_label_rel"].reshape(3, 1)
            target_t_label_rel = rot_mat @ t_rel_gt
            results["target_t_label_rel"] = target_t_label_rel.flatten()

            # 2. 旋转物体的绝对朝向矩阵 (R_new = rot_mat @ R_old)
            target_R_label = rot_mat @ label["R_label"]
            results["target_R_label"] = target_R_label
            
            # 3. 重新提取 6D 连续表征
            results["target_rot6d_label"] = rotmat_to_rot6d(
                torch.from_numpy(target_R_label).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)
            
            # 传递不需要被增强改变的标准空间模板
            results["target_obj_pc_sparse_rest"] = label["obj_pc_sparse_rest"]
            if label.get("obj_pc_eval_rest") is not None:
                results["target_obj_pc_eval_rest"] = label["obj_pc_eval_rest"]
            results["obj_id"] = label["obj_id"]

        if label.get("obj_center_3d") is not None:
            # 1. 施加 3D 数据增强旋转
            target_obj_center_3d = rot_mat.dot(label["obj_center_3d"].transpose(1, 0)).transpose()
            
            # 2. 利用增强后的完美相机内参，将 3D 中心点直接透视投影为 2D 坐标！
            obj_pts_2d = target_cam_intr.dot(target_obj_center_3d.T).T
            
            # 透视除法取出 U, V，并保留 Z 深度 (组合成 U, V, D)
            Z = obj_pts_2d[:, 2:3]
            Z[np.abs(Z) < 1e-6] = 1e-6
            
            U = obj_pts_2d[:, 0:1] / Z
            V = obj_pts_2d[:, 1:2] / Z
            
            obj_uv = np.concatenate([U, V], axis=-1).astype(np.float32)
            target_obj_center_uv = obj_uv / (np.array([self._output_size[0], self._output_size[1]], dtype=np.float32) / 2) - 1
            
            # 计算 obj_center 的可见性
            if not self._train:
                target_obj_center_vis = np.ones(1, dtype=np.float32)
            else:
                obj_center_2d = obj_uv 
                target_obj_center_vis = (((obj_center_2d[:, 0] >= 0) & (obj_center_2d[:, 0] < self._output_size[0])) &
                                         ((obj_center_2d[:, 1] >= 0) & (obj_center_2d[:, 1] < self._output_size[1]))).astype(np.float32)

            results["target_obj_center_3d"] = target_obj_center_3d
            results["target_obj_center_uv"] = target_obj_center_uv
            results["target_obj_center_vis"] = target_obj_center_vis
            if self._with_heatmap:
                results["target_obj_center_heatmap"] = self._generate_target_heatmap(obj_uv, target_obj_center_vis)

        # 🌟 同步旋转 4x4 物体位姿矩阵
        if label.get("obj_transform") is not None:
            rot_mat_4x4 = np.eye(4, dtype=np.float32)
            rot_mat_4x4[:3, :3] = rot_mat
            target_obj_transform = rot_mat_4x4 @ label["obj_transform"]
            results["target_obj_transform"] = target_obj_transform

        results["affine_postrot"] = affine_postrot
        results["extr_prerot"] = rot_mat
        results["target_cam_intr"] = target_cam_intr
        results["target_joints_3d"] = target_joints_3d
        results["target_verts_3d"] = target_verts_3d
        results["target_joints_3d_rel"] = target_joints_3d_rel
        results["target_verts_3d_rel"] = target_verts_3d_rel
        results["target_root_joint"] = target_root_joint

        # 🚨 修复 3：条件赋值！仅当值不为 None 时才放入 results，绝不塞入 NoneType
        if target_joints_3d_no_rot is not None:
            results["target_joints_3d_no_rot"] = target_joints_3d_no_rot
        if target_verts_3d_no_rot is not None:
            results["target_verts_3d_no_rot"] = target_verts_3d_no_rot
        if target_obj_sparse_no_rot is not None:
            results["target_obj_sparse_no_rot"] = target_obj_sparse_no_rot
        if target_obj_dense_no_rot is not None:
            results["target_obj_dense_no_rot"] = target_obj_dense_no_rot
        if target_obj_center_3d_no_rot is not None:
            results["target_obj_center_3d_no_rot"] = target_obj_center_3d_no_rot
        if target_obj_trans_3d_no_rot is not None:
            results["target_obj_trans_3d_no_rot"] = target_obj_trans_3d_no_rot

        return results


@TRANSFORM.register_module()
class SimpleTransform3D(SimpleTransformUVD):
    """Apply 2D and 3D transform to images and labels, including:
    1. scale and color jittering;
    2. random ratation on image, 2d/3d keypoints, MANO global rotation;
    3. random occlusion on image;
    """

    def __init__(self, cfg: CN) -> None:
        super().__init__(cfg)

    def __call__(self, image, label):
        results = super().__call__(image, label)
        rot = results["rot_rad"]
        rot_mat = results["rot_mat3d"]

        center = results["target_bbox_center"]
        scale = results["target_bbox_scale"]
        cc = label["cam_center"]
        affine_postrot = _affine_transform_post_rot(center=center,
                                                    scale=scale,
                                                    optical_center=cc,
                                                    out_res=self._output_size,
                                                    rot=rot)

        target_cam_intr = affine_postrot.dot(label["cam_intr"])
        target_joints_3d = rot_mat.dot(label["joints_3d"].transpose(1, 0)).transpose()
        target_verts_3d = rot_mat.dot(label["verts_3d"].transpose(1, 0)).transpose()
        target_root_joint = target_joints_3d[self._center_idx]
        target_joints_3d_rel = target_joints_3d - target_root_joint
        target_verts_3d_rel = target_verts_3d - target_root_joint

        results["affine_postrot"] = affine_postrot
        results["target_cam_intr"] = target_cam_intr
        results["target_joints_3d"] = target_joints_3d
        results["target_verts_3d"] = target_verts_3d
        results["target_joints_3d_rel"] = target_joints_3d_rel
        results["target_verts_3d_rel"] = target_verts_3d_rel
        results["target_root_joint"] = target_root_joint

        return results


@TRANSFORM.register_module()
class SimpleTransform3DMANO(SimpleTransform3D):

    def __init__(self, cfg: CN) -> None:
        super().__init__(cfg)

    def __call__(self, image, label):
        results = super().__call__(image, label)
        rot = results["rot_rad"]
        target_mano_pose = _rotate_smpl_pose(label["mano_pose"], rot).reshape(-1, 3)

        results["target_mano_pose"] = target_mano_pose
        results["target_mano_shape"] = label["mano_shape"]

        return results


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript. 

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = 'xyz', **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, np.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError('Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def rotmat_to_aa(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def aa_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles f{axis_angle.shape}.')
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def aa_to_rot6d(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)


def quat_to_aa(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def rot6d_to_rotmat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def rotmat_to_rot6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def rotmat_to_quat(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions shape f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_quat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d shape f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


def _rotate_smpl_pose(pose, rot):
    """Rotate SMPL pose parameters.

    SMPL (https://smpl.is.tue.mpg.de/) is a 3D human model.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation rad.
    Returns:
        pose_rotated
    """
    rot_mat = _construct_rotation_matrix(rot)
    pose_rotated = pose.copy()
    orient = pose[:3]
    orient_mat = aa_to_rotmat(orient)

    new_orient_mat = np.matmul(rot_mat, orient_mat)
    new_orient = rotmat_to_aa(new_orient_mat)
    pose_rotated[:3] = new_orient

    return pose_rotated


def _construct_rotation_matrix(rot, size=3):
    """Construct the in-plane rotation matrix.

    Args:
        rot (float): Rotation rad.
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.
    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        sn, cs = np.sin(rot), np.cos(rot)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    return rot_mat


def _transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows


def _get_affine_transform(center, scale, optical_center, out_res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -optical_center[0]
    t_mat[1, 2] = -optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    post_rot_trans = _get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = _get_affine_trans_no_rot(transformed_center[:2], scale, out_res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(np.float32)


def _affine_transform(center, scale, out_res, rot=0):
    rotmat = _construct_rotation_matrix(rot=rot, size=3)
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = (rotmat.dot(np.concatenate([center, np.ones(1)])))[:2]

    post_rot_trans = _get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rotmat)
    return total_trans.astype(np.float32)


def _affine_transform_post_rot(center, scale, optical_center, out_res, rot=0):
    rotmat = _construct_rotation_matrix(rot=rot, size=3)
    t_mat = np.eye(3)
    t_mat[0, 2] = -optical_center[0]
    t_mat[1, 2] = -optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rotmat).dot(t_mat).dot(np.concatenate([center, np.ones(1)]))
    affine_trans_post_rot = _get_affine_trans_no_rot(transformed_center[:2], scale, out_res)

    return affine_trans_post_rot.astype(np.float32)


def _get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    scale_ratio = float(res[0]) / float(res[1])
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale * scale_ratio
    affinet[0, 2] = res[0] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[1] * (-float(center[1]) / scale * scale_ratio + 0.5)
    affinet[2, 2] = 1
    return affinet


def fit_ortho_param(joints3d: np.ndarray, joints2d: np.ndarray) -> np.ndarray:
    """Fit the orthographic projection parameters.

    Args:
        joints3d (np.ndarray): 3D joints. (N, 3)
        joints2d (np.ndarray): 2D joints. (N, 2)

    Returns:
        np.ndarray: The orthographic projection parameters (f, tx, ty). (3,) 
    """
    joints3d_xy = joints3d[:, :2]  # (21, 2)
    joints3d_xy = joints3d_xy.reshape(-1)[:, np.newaxis]
    joints2d = joints2d.reshape(-1)[:, np.newaxis]
    pad2 = np.array(range(joints2d.shape[0]))
    pad2 = (pad2 % 2)[:, np.newaxis]
    pad1 = 1 - pad2
    jM = np.concatenate([joints3d_xy, pad1, pad2], axis=1)  # (42, 3)
    jMT = jM.transpose()  # (3, 42)
    jMTjM = np.matmul(jMT, jM)
    jMTb = np.matmul(jMT, joints2d)
    ortho_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
    ortho_param = ortho_param.reshape(-1)
    return ortho_param  # [f, tx, ty]


def batch_xyz2uvd(xyz: torch.Tensor,
                  root_joint: torch.Tensor,
                  intr: torch.Tensor,
                  inp_res: List[int],
                  depth_range=0.4,
                  ref_bone_len: Optional[torch.Tensor] = None,
                  camera_mode="persp") -> torch.Tensor:

    inp_res = torch.Tensor(inp_res).to(xyz.device)  # TENSOR (2,)
    batch_size = xyz.shape[0]
    if ref_bone_len is None:
        ref_bone_len = torch.ones((batch_size, 1)).to(xyz.device)  # TENSOR (B, 1)

    if camera_mode == "persp":
        assert intr.dim() == 3, f"Unexpected dim, expect intr has shape (B, 3, 3), got {intr.shape}"
        #  1. normalize depth : root_relative, scale_invariant
        z = xyz[:, :, 2]  # TENSOR (B, NKP)
        xy = xyz[:, :, :2]  # TENSOR (B, NKP, 2)
        xy_ = xy / z.unsqueeze(-1).expand_as(xy)  # TENSOR (B, NKP, 2)
        root_joint_z = root_joint[:, -1].unsqueeze(-1)  # TENSOR (B, 1)
        z_ = (z - root_joint_z.expand_as(z)) / ref_bone_len.expand_as(z)  # TENSOR (B, NKP)

        #  2. xy_ -> uv
        fx = intr[:, 0, 0].unsqueeze(-1)  # TENSOR (B, 1)
        fy = intr[:, 1, 1].unsqueeze(-1)
        cx = intr[:, 0, 2].unsqueeze(-1)
        cy = intr[:, 1, 2].unsqueeze(-1)
        # cat 4 TENSOR (B, 1)
        camparam = torch.cat((fx, fy, cx, cy), dim=1)  # TENSOR (B, 4)
        camparam = camparam.unsqueeze(1).expand(-1, xyz.shape[1], -1)  # TENSOR (B, NKP, 4)
        uv = (xy_ * camparam[:, :, :2]) + camparam[:, :, 2:4]  # TENSOR (B, NKP, 2)

        #  3. normalize uvd to 0~1
        uv = torch.einsum("bij, j->bij", uv, 1.0 / inp_res)  # TENSOR (B, NKP, 2), [0 ~ 1]
        d = z_ / depth_range + 0.5  # TENSOR (B, NKP), [0 ~ 1]
        uvd = torch.cat((uv, d.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)
    elif camera_mode == "ortho":
        assert intr.dim() == 2, f"Unexpected dim, expect intr has shape (B, 3), got {intr.shape}"
        # root_relative
        xyz = xyz - root_joint.unsqueeze(1)  # TENSOR (B, NKP, 3)

        xy = xyz[:, :, :2]  # TENSOR (B, NKP, 2)
        z = xyz[:, :, 2]  # TENSOR (B, NKP)
        z_ = z / ref_bone_len.expand_as(z)  # TENSOR (B, NKP)
        d = z_ / depth_range + 0.5  # TENSOR (B, NKP), [0 ~ 1]

        scale = intr[:, :1].unsqueeze(1)  # TENSOR (B, 1, 1)
        shift = intr[:, 1:].unsqueeze(1)  # TENSOR (B, 1, 2)
        uv = xy * scale + shift  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
        uv = torch.einsum("bij,j->bij", uv, 1.0 / inp_res)  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
        uvd = torch.cat((uv, d.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)

    return uvd


def batch_uvd2xyz(uvd: torch.Tensor,
                  root_joint: torch.Tensor,
                  intr: torch.Tensor,
                  inp_res: List[int],
                  depth_range: float = 0.4,
                  ref_bone_len: Optional[torch.Tensor] = None,
                  camera_mode="persp"):

    inp_res = torch.Tensor(inp_res).to(uvd.device)
    batch_size = uvd.shape[0]
    if ref_bone_len is None:
        ref_bone_len = torch.ones((batch_size, 1)).to(uvd.device)

    #  1. denormalized uvd
    uv = torch.einsum("bij,j->bij", uvd[:, :, :2], inp_res)  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
    d = (uvd[:, :, 2] - 0.5) * depth_range  # TENSOR (B, NKP), [-0.2 ~ 0.2]

    if camera_mode == "persp":
        assert intr.dim() == 3, f"Unexpected dim, expect intr has shape (B, 3, 3), got {intr.shape}"
        root_joint_z = root_joint[:, -1].unsqueeze(-1)  # TENSOR (B, 1)
        z = d * ref_bone_len + root_joint_z.expand_as(uvd[:, :, 2])  # TENSOR (B, NKP)

        #  2. uvd->xyz
        # camparam = torch.zeros((batch_size, 4)).float().to(uvd.device)  # TENSOR (B, 4)
        fx = intr[:, 0, 0].unsqueeze(-1)  # TENSOR (B, 1)
        fy = intr[:, 1, 1].unsqueeze(-1)
        cx = intr[:, 0, 2].unsqueeze(-1)
        cy = intr[:, 1, 2].unsqueeze(-1)
        # cat 4 TENSOR (B, 1)
        camparam = torch.cat((fx, fy, cx, cy), dim=1)  # TENSOR (B, 4)
        camparam = camparam.unsqueeze(1).expand(-1, uvd.shape[1], -1)  # TENSOR (B, NKP, 4)
        xy_ = (uv - camparam[:, :, 2:4]) / camparam[:, :, :2]  # TENSOR (B, NKP, 2)
        xy = xy_ * z.unsqueeze(-1).expand_as(uv)  # TENSOR (B, NKP, 2)
        xyz = torch.cat((xy, z.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)
    elif camera_mode == "ortho":
        assert intr.dim() == 2, f"Unexpected dim, expect intr has shape (B, 3), got {intr.shape}"
        scale = intr[:, :1].unsqueeze(1)  # TENSOR (B, 1, 1)
        shift = intr[:, 1:].unsqueeze(1)  # TENSOR (B, 1, 2)
        xy = (uv - shift) / scale
        z = d * ref_bone_len
        xyz = torch.cat((xy, z.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)

        # add root back
        xyz = xyz + root_joint.unsqueeze(1)  # TENSOR (B, NKP, 3)

    return xyz


def mano_to_openpose(J_regressor, mano_verts):
    """  Convert MANO vertices to OpenPose joints
    NOTE: the MANO Joints' order (with tips)

                       16-15-14-13-\
                                    \
                 17 --3 --2 --1------0
               18 --6 --5 --4-------/
               19 -12 -11 --10-----/
                 20 --9 --8 --7---/

    NOTE: the OpenPose Joints' order (with tips)

                       4 -3 -2 -1 -\
                                    \
                  8 --7 --6 --5------0
               12 --11--10--9-------/
               16 -15 -14 --13-----/
                 20--19--18--17---/

    Args:
        J_regressor (torch.Tensor): MANO's th_J_regressor, shape (16, 778)
        mano_verts (torch.Tensor): MANO hand's vertices, shape (B, 778, 3)

    Returns:
        torch.Tensor: hand joints of OpenPose Joints' order, shape (B, 21, 3)
    """
    mano_joints = torch.matmul(J_regressor, mano_verts)
    kpId2vertices = CONST.MANO_KPID_2_VERTICES
    tipsId = [v[0] for k, v in kpId2vertices.items()]
    tips = mano_verts[:, tipsId]
    openpose_joints = torch.cat([mano_joints, tips], dim=1)
    # Reorder joints to match OpenPose definition
    openpose_joints = openpose_joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    # NOTE: if you want to reorder back, use: reorder_idx =
    #       [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
    return openpose_joints


def batch_ref_bone_len(joint: Union[np.ndarray, torch.Tensor], ref_bone_link=None) -> Union[np.ndarray, torch.Tensor]:

    if ref_bone_link is None:
        ref_bone_link = (0, 9)

    if not torch.is_tensor(joint) and not isinstance(joint, np.ndarray):
        raise TypeError("joint should be ndarray or torch tensor. Got {}".format(type(joint)))
    if len(joint.shape) != 3 or joint.shape[1] != 21 or joint.shape[2] != 3:
        raise TypeError("joint should have shape (B, njoint, 3), Got {}".format(joint.shape))

    batch_size = joint.shape[0]
    bone = 0
    if torch.is_tensor(joint):
        bone = torch.zeros((batch_size, 1)).to(joint.device)
        for jid, nextjid in zip(ref_bone_link[:-1], ref_bone_link[1:]):
            bone += torch.norm(joint[:, jid, :] - joint[:, nextjid, :], dim=1, keepdim=True)  # (B, 1)
    elif isinstance(joint, np.ndarray):
        bone = np.zeros((batch_size, 1))
        for jid, nextjid in zip(ref_bone_link[:-1], ref_bone_link[1:]):
            bone += np.linalg.norm((joint[:, jid, :] - joint[:, nextjid, :]), ord=2, axis=1, keepdims=True)  # (B, 1)
    return bone


def batch_cam_extr_transf(batch_cam_extr, batch_joints):
    """apply batch camera extrinsic transformation on batch joints

    Args:
        batch_cam_extr (torch.Tensor): shape (BATCH, NPERSP, 4, 4)
        batch_joints (torch.Tensor): shape (BATCH, NPERSP, NJOINTS, 3)

    Returns:
        torch.Tensor: shape (BATCH, NPERSP, NJOINTS, 3)
    """
    res = (batch_cam_extr[..., :3, :3] @ batch_joints.transpose(2, 3)).transpose(2, 3)
    # [B, NPERSP, 3, 3] @ [B, NPERSP, 3, 21] => [B, NPERSP, 3, 21] => [B, NPERSP, 21, 3]
    res = res + batch_cam_extr[..., :3, 3].unsqueeze(2)
    return res


def batch_cam_intr_projection(batch_cam_intr, batch_joints, eps=1e-7):
    """apply camera projection on batch joints with batch intrinsics

    Args:
        batch_cam_intr (torch.Tensor): shape (BATCH, NPERSP, 3, 3)
        batch_joints (torch.Tensor): shape (BATCH, NPERSP, NJOINTS, 3)
        eps (float, optional): avoid divided by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: shape (BATCH, NPERSP, NJOINTS, 2)
    """
    res = (batch_cam_intr @ batch_joints.transpose(2, 3)).transpose(2, 3)  # [B, NPERSP, 21, 3]
    res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
    xy = res[..., 0:2]
    z = res[..., 2:]
    z_sign = torch.where(z >= 0.0, torch.ones_like(z), -torch.ones_like(z))
    safe_z = torch.where(torch.abs(z) < eps, z_sign * eps, z)
    uv = xy / safe_z
    uv = torch.nan_to_num(uv, nan=0.0, posinf=1e6, neginf=-1e6)
    return uv


def batch_persp_project(verts: torch.Tensor, camintr: torch.Tensor):
    """Batch apply perspective procjection on points

    Args:
        verts (torch.Tensor): 3D points with shape (B, N, 3)
        camintr (torch.Tensor): intrinsic matrix with shape (B, 3, 3)

    Returns:
        torch.Tensor: shape (B, N, 2)
    """
    # Project 3d vertices on image plane
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def persp_project(points3d, cam_intr):
    """Apply perspective camera projection on a 3D point

    Args:
        points3d (np.ndarray): shape (N, 3)
        cam_intr (np.ndarray): shape (3, 3)

    Returns:
        np.ndarray: shape (N, 2)
    """
    hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
    points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
    return points2d.astype(np.float32)


def SE3_transform(points3d, transform):
    """Apply SE3 transform on a 3D point

    Args:
        points3d (np.ndarray): shape (N, 3)
        transform (np.ndarray): shape (4, 4)

    Returns:
        np.ndarray: shape (N, 3)
    """
    return (transform[:3, :3] @ points3d.T).T + transform[:3, 3][None, :]


def ortho_project(points3d, ortho_cam):
    """Apply orthographic camera projection on a 3D point

    Args:
        points3d (np.ndarray): shape (N, 3)
        ortho_cam (np.ndarray): shape (3, 3)

    Returns:
        np.ndarray: shape (N, 2)
    """
    x, y = points3d[:, 0], points3d[:, 1]
    u = ortho_cam[0] * x + ortho_cam[1]
    v = ortho_cam[0] * y + ortho_cam[2]
    u_, v_ = u[:, np.newaxis], v[:, np.newaxis]
    return np.concatenate([u_, v_], axis=1)


def get_annot_scale(annots, visibility=None, scale_factor=1.0):
    """
    Retreives the size of the square we want to crop by taking the
    maximum of vertical and horizontal span of the hand and multiplying
    it by the scale_factor to add some padding around the hand
    """
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    s = max_delta * scale_factor
    return s


def get_annot_center(annots, visibility=None):
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    return np.asarray([c_x, c_y])


def process_bbox(bbox, factor=1.25):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = 1.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * factor
    bbox[3] = h * factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError('Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError('Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def center_scale_to_box(center, scale):
    """Convert bbox center scale to bbox xyxy

    Args:
        center (np.array): center of the bbox (x, y)
        scale (np.float_): side length of the bbox (bbox must be square)

    Returns:
        list: list of 4 elms, containing bbox' s xmin, ymin, xmax, ymax.
    """
    pixel_std = 1.0
    w = scale * pixel_std
    h = scale * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def denormalize(tensor, mean, std, inplace=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).sub_(-1 * mean)
    return tensor


def bhwc_2_bchw(tensor):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError("invalid tensor or tensor channel is not BCHW")
    return tensor.unsqueeze(1).transpose(1, -1).squeeze(-1)


def bchw_2_bhwc(tensor):
    """
    :param x: torch tensor, B x C x H x W
    :return:  torch tensor, B x H x W x C
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError("invalid tensor or tensor channel is not BCHW")
    return tensor.unsqueeze(-1).transpose(1, -1).squeeze(1)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def cal_transform_mean(batch_extr: Union[List[np.ndarray], np.ndarray]):
    """Calculate the mean of the transform matrix.
    The rotation mean is calculated thorugh quaternion mean.

    Args:
        batch_extr (Union[List[np.ndarray], np.ndarray]): The transform matrix list.

    Returns:
        np.ndarray: The mean of the rotation matrix. (3, 3)
        np.ndarray: The mean of the translation matrix. (3, )
    """

    if isinstance(batch_extr, list):
        batch_extr = np.array(batch_extr)  # (N, 4, 4)

    n_transform = batch_extr.shape[0]
    batch_transl = batch_extr[:, :3, 3]  # (N, 3)
    batch_rot = batch_extr[:, :3, :3]  # (N, 3, 3)
    batch_quat = rotmat_to_quat(batch_rot)  # (N, 4)
    mean_quat = np.mean(batch_quat, axis=0)  # (4,)
    mean_quat = mean_quat / (np.linalg.norm(mean_quat) + 1e-7)
    mean_rot = quat_to_rotmat(mean_quat)  # (3, 3)

    mean_trasl_center = np.mean(batch_transl, axis=0)  # (3)

    return mean_rot, mean_trasl_center
