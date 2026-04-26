import hashlib
import json
import os
import pickle
import random
import time
import warnings
from collections import defaultdict
from typing import List

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import trimesh
import yaml
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from termcolor import colored

from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import (SE3_transform, aa_to_rotmat, batch_ref_bone_len, cal_transform_mean, denormalize,
                               get_annot_center, get_annot_scale, persp_project, rotmat_to_aa, rotmat_to_rot6d)
from .hdata import HDataset, kpId2vertices


def _bbox21_from_bounds(bounds):
    bounds = np.asarray(bounds, dtype=np.float32)
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    p_blb = np.array([x_min, y_min, z_min], dtype=np.float32)
    p_brb = np.array([x_max, y_min, z_min], dtype=np.float32)
    p_blf = np.array([x_min, y_max, z_min], dtype=np.float32)
    p_brf = np.array([x_max, y_max, z_min], dtype=np.float32)
    p_tlb = np.array([x_min, y_min, z_max], dtype=np.float32)
    p_trb = np.array([x_max, y_min, z_max], dtype=np.float32)
    p_tlf = np.array([x_min, y_max, z_max], dtype=np.float32)
    p_trf = np.array([x_max, y_max, z_max], dtype=np.float32)
    p_center = (p_tlb + p_brf) * 0.5
    p_ble = (p_blb + p_blf) * 0.5
    p_bre = (p_brb + p_brf) * 0.5
    p_bfe = (p_blf + p_brf) * 0.5
    p_bbe = (p_blb + p_brb) * 0.5
    p_tle = (p_tlb + p_tlf) * 0.5
    p_tre = (p_trb + p_trf) * 0.5
    p_tfe = (p_tlf + p_trf) * 0.5
    p_tbe = (p_tlb + p_trb) * 0.5
    p_lfe = (p_tlf + p_blf) * 0.5
    p_lbe = (p_tlb + p_blb) * 0.5
    p_rfe = (p_trf + p_brf) * 0.5
    p_rbe = (p_trb + p_brb) * 0.5
    return np.stack(
        (
            p_blb, p_brb, p_blf, p_brf,
            p_tlb, p_trb, p_tlf, p_trf,
            p_ble, p_bre, p_bfe, p_bbe,
            p_tle, p_tre, p_tfe, p_tbe,
            p_lfe, p_lbe, p_rfe, p_rbe,
            p_center,
        ),
        axis=0,
    ).astype(np.float32)


@DATASET.register_module()
class HO3D(HDataset):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.mini_factor_of_dataset = float(cfg.get("MINI_FACTOR", 1.0))

        self.use_gt_from_multiview = cfg.get("USE_GT_FROM_MULTIVIEW", False)
        self.use_official_eval_gt_json = bool(cfg.get("USE_OFFICIAL_EVAL_GT_JSON", False))
        self.use_test_gt_root = os.path.join(self.data_root, "HO3D_v3_manual_test_gt")

        # ======== HO3D params >>>>>>>>>>>>>>>>>>
        self.split_mode = cfg.SPLIT_MODE  # paper
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ======== HO3D default >>>>>>>>>>>>>>>>>
        self.raw_size = (640, 480)
        self.reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        # this camera extrinsic has no translation
        # and this is the reason transforms in following code just use rotation part
        self.cam_extr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).astype(np.float32)

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,
            flat_hand_mean=True,
        )

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.load_dataset()
        self._init_object_assets()

    def _preload(self):
        # deal with all the naming and path convention
        self.name = "HO3D_v3"
        self.root = os.path.join(self.data_root, self.name)
        self.root_extra_info = os.path.normpath("assets")

        self.cache_identifier_dict = {
            "data_split": self.data_split,
            "split_mode": self.split_mode,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        self._preload()

        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if self.split_mode == "paper":  # official paper split V2 and V3
            # For details:
            # V2: https://competitions.codalab.org/competitions/22485
            # V3: https://competitions.codalab.org/competitions/33267
            seq_frames, subfolder = self._load_seq_frames()
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}")
        else:
            raise NotImplementedError()

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")

        else:
            annot_mapping, seq_idx = self._load_annots(seq_frames=seq_frames, subfolder=subfolder)

            annotations = {"seq_idx": seq_idx, "annot_mapping": annot_mapping}

            with open(self.cache_path, "wb") as p_f:
                pickle.dump(annotations, p_f)
            logger.info(f"Wrote cache for {self.name}_{self.data_split}_{self.split_mode} to {self.cache_path}")

        self.seq_idx = annotations["seq_idx"]
        self.annot_mapping = annotations["annot_mapping"]
        self.sample_idxs = list(range(len(self.seq_idx)))
        self._load_official_eval_gt()

        if self.mini_factor_of_dataset != float(1):
            random.Random(1).shuffle(self.sample_idxs)
            self.sample_idxs = self.sample_idxs[:int(self.mini_factor_of_dataset * len(self.sample_idxs))]

        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{len(self.seq_idx)} samples for data_split {self.data_split}")

    def _load_official_eval_gt(self):
        self.official_eval_xyz = None
        self.official_eval_verts = None
        self.official_eval_key_to_idx = {}
        self._test_hand_gt_cache = {}

        if not (self.data_split == "test" and self.use_official_eval_gt_json):
            return

        eval_txt_path = os.path.join(self.root, "evaluation.txt")
        xyz_path = os.path.join(self.root, "evaluation_xyz.json")
        verts_path = os.path.join(self.root, "evaluation_verts.json")
        cache_path = os.path.join("common", "cache", self.name, "official_eval_hand_gt_v1.npz")

        if not (os.path.isfile(eval_txt_path) and os.path.isfile(xyz_path) and os.path.isfile(verts_path)):
            logger.warning(
                f"{self.name} official evaluation GT JSON files are missing; "
                f"expected {xyz_path} and {verts_path}"
            )
            return

        with open(eval_txt_path, "r") as f:
            frame_keys = [line.strip() for line in f if line.strip()]

        xyz, verts = None, None
        if os.path.isfile(cache_path):
            try:
                cache = np.load(cache_path, allow_pickle=False)
                cached_keys = cache["frame_keys"].astype(str).tolist()
                if cached_keys == frame_keys:
                    xyz = cache["xyz"].astype(np.float32)
                    verts = cache["verts"].astype(np.float32)
                    logger.info(f"Loaded official HO3D eval GT cache from {cache_path}")
            except Exception as exc:
                logger.warning(f"Failed to load official HO3D eval GT cache {cache_path}: {exc}")

        if xyz is None or verts is None:
            logger.info("Loading official HO3D eval joints/verts GT from JSON files...")
            with open(xyz_path, "r") as f:
                xyz = np.asarray(json.load(f), dtype=np.float32)
            with open(verts_path, "r") as f:
                verts = np.asarray(json.load(f), dtype=np.float32)

            if len(xyz) != len(frame_keys) or len(verts) != len(frame_keys):
                raise ValueError(
                    f"Official HO3D eval GT length mismatch: "
                    f"len(xyz)={len(xyz)}, len(verts)={len(verts)}, len(keys)={len(frame_keys)}"
                )

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, xyz=xyz, verts=verts, frame_keys=np.asarray(frame_keys))
            logger.info(f"Cached official HO3D eval GT to {cache_path}")

        self.official_eval_xyz = xyz
        self.official_eval_verts = verts
        self.official_eval_key_to_idx = {key: idx for idx, key in enumerate(frame_keys)}

    def _get_test_gt_entry(self, seq, img_idx):
        cache_key = (seq, int(img_idx))
        if cache_key in self._test_hand_gt_cache:
            return self._test_hand_gt_cache[cache_key]

        annot = self.annot_mapping[seq][img_idx]
        frame_idx = str(annot["frame_idx"])
        entry = None

        if self.use_gt_from_multiview:
            test_gt_root = os.path.join(self.use_test_gt_root, seq, "meta", f"{frame_idx}.pkl")
            if os.path.exists(test_gt_root):
                with open(test_gt_root, "rb") as p_f:
                    manual_gt = pickle.load(p_f)
                entry = {
                    "source": "manual_test_gt",
                    "target_joints_3d": manual_gt["target_joints_3d"].astype(np.float32),
                    "target_verts_3d": manual_gt["target_verts_3d"].astype(np.float32),
                    "mano_pose": manual_gt["mano_pose"].astype(np.float32),
                    "mano_shape": manual_gt["mano_shape"].astype(np.float32),
                }

        if entry is None and self.official_eval_xyz is not None and self.official_eval_verts is not None:
            official_idx = self.official_eval_key_to_idx.get(f"{seq}/{frame_idx}", None)
            if official_idx is not None:
                entry = {
                    "source": "official_eval_json",
                    "official_idx": int(official_idx),
                }

        self._test_hand_gt_cache[cache_key] = entry
        return entry

    def _resolve_ycb_model_root(self):
        candidates = [
            os.path.join(self.data_root, "DexYCB", "models"),
            os.path.join(self.data_root, "YCB", "models"),
            os.path.join("data", "DexYCB", "models"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find YCB object models for HO3D object supervision. "
            "Expected a directory like DATA_ROOT/DexYCB/models."
        )

    @staticmethod
    def _pick_mesh_path(model_root, obj_name, obj_id):
        candidates = [
            os.path.join(model_root, obj_name, "textured_simple.obj"),
            os.path.join(model_root, obj_name, "textured.obj"),
            os.path.join(os.path.dirname(model_root), "bop", "models_eval", f"obj_{obj_id:06d}.ply"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find a mesh for HO3D object {obj_name} (id={obj_id})")

    def _iter_unique_objects(self):
        unique = {}
        for seq_annots in self.annot_mapping.values():
            for annot in seq_annots:
                obj_name = annot.get("objName", None)
                obj_label = annot.get("objLabel", None)
                if obj_name is None or obj_label is None:
                    continue
                unique[str(obj_name)] = int(obj_label)
        return sorted(unique.items(), key=lambda kv: kv[1])

    def _init_object_assets(self):
        self.obj_rest_corners = {}
        self.obj_kp21_rest = {}
        self.obj_pc_sparse = {}
        self.obj_pc_dense = {}
        self.obj_pc_eval = {}
        self.obj_name_to_id = {}
        self.obj_id_to_name = {}
        self.obj_mesh_path = {}
        self.ycb_model_root = self._resolve_ycb_model_root()

        unique_objects = self._iter_unique_objects()
        if not self.quiet_init:
            logger.info(
                f"Pre-sampling HO3D YCB objects from {self.ycb_model_root} "
                f"(Sparse: 2048, Dense: 16384, Eval: 30000)..."
            )

        for obj_name, obj_id in unique_objects:
            mesh_path = self._pick_mesh_path(self.ycb_model_root, obj_name, obj_id)
            object_mesh = trimesh.load(mesh_path, process=False)

            verts = torch.from_numpy(np.asarray(object_mesh.vertices, dtype=np.float32)).float()
            faces = torch.from_numpy(np.asarray(object_mesh.faces, dtype=np.int64)).long()
            pt3d_mesh = Meshes(verts=[verts], faces=[faces])

            pc_sparse = sample_points_from_meshes(pt3d_mesh, 2048)[0].numpy()
            pc_dense = sample_points_from_meshes(pt3d_mesh, 16384)[0].numpy()
            pc_eval = sample_points_from_meshes(pt3d_mesh, 30000)[0].numpy()

            self.obj_name_to_id[obj_name] = obj_id
            self.obj_id_to_name[obj_id] = obj_name
            self.obj_mesh_path[obj_id] = mesh_path
            self.obj_rest_corners[obj_id] = trimesh.bounds.corners(object_mesh.bounds).astype(np.float32)
            self.obj_kp21_rest[obj_id] = _bbox21_from_bounds(object_mesh.bounds)
            self.obj_pc_sparse[obj_id] = pc_sparse.astype(np.float32)
            self.obj_pc_dense[obj_id] = pc_dense.astype(np.float32)
            self.obj_pc_eval[obj_id] = pc_eval.astype(np.float32)

        if not self.quiet_init:
            logger.info(f"HO3D object assets ready for {len(unique_objects)} YCB objects")

    def _load_seq_frames(self, subfolder=None, seqs=None, trainval_idx=6000):
        """
        trainval_idx (int): How many frames to include in training split when
                using trainval/val/test split
        """
        if self.split_mode == "paper":
            if self.data_split == "train":
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                info_path = os.path.join(self.root, "evaluation.txt")
                subfolder = "evaluation"
            else:
                assert False
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
        else:
            assert False
        return seq_frames, subfolder

    def _load_annots(self, seq_frames=[], subfolder="train", **kwargs):
        seq_idx = []
        annot_mapping = defaultdict(list)
        seq_counts = defaultdict(int)
        for idx_count, (seq, frame_idx) in enumerate(etqdm(seq_frames)):
            seq_folder = os.path.join(self.root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")

            with open(meta_path, "rb") as p_f:
                annot = pickle.load(p_f)
                if annot["handJoints3D"].size == 3:
                    annot["handTrans"] = annot["handJoints3D"]
                    annot["handJoints3D"] = annot["handJoints3D"][np.newaxis, :].repeat(21, 0)
                    annot["handPose"] = np.zeros(48, dtype=np.float32)
                    annot["handBeta"] = np.zeros(10, dtype=np.float32)

            img_path = os.path.join(rgb_folder, f"{frame_idx}.jpg")
            annot["img"] = img_path
            annot["frame_idx"] = frame_idx

            annot_mapping[seq].append(annot)
            seq_idx.append((seq, seq_counts[seq]))
            seq_counts[seq] += 1

        return annot_mapping, seq_idx

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idxs(self) -> List[int]:
        return self.sample_idxs

    def get_seq_frame(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        frame_idx = annot["frame_idx"]
        return seq, frame_idx

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = np.array(imageio.imread(img_path, pilmode="RGB"), dtype=np.uint8)
        return img

    def get_image_path(self, idx):
        seq, img_idx = self.seq_idx[idx]
        img_path = self.annot_mapping[seq][img_idx]["img"]
        return img_path

    def get_rawimage_size(self, idx):
        return (640, 480)

    def get_sides(self, idx):
        return "right"

    def get_image_mask(self, idx):
        seq, img_idx = self.seq_idx[idx]
        # image_mask not found
        zeroo = np.zeros((640, 480))
        return np.array(zeroo, dtype=np.uint8)

    def get_obj_corners(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_corners = np.asarray(annot["objCorners3D"], dtype=np.float32)
        obj_corners = self.cam_extr[:3, :3].dot(obj_corners.transpose()).transpose()
        return obj_corners.astype(np.float32)

    def get_bbox_hand_obj(self, idx, scale_factor=1.25):
        hand_joints_2d = self.get_joints_2d(idx)
        obj_corners = self.get_obj_corners(idx)
        cam_intr = self.get_cam_intr(idx)
        obj_corners_2d = persp_project(obj_corners, cam_intr)
        if self.crop_model == "root_obj":
            crop_points_2d = np.concatenate([hand_joints_2d[[self.center_idx]], obj_corners_2d], axis=0)
        else:
            crop_points_2d = np.concatenate([hand_joints_2d, obj_corners_2d], axis=0)

        center = get_annot_center(crop_points_2d)
        scale = get_annot_scale(crop_points_2d) * scale_factor
        return center.astype(np.float32), float(scale)

    def get_joints_2d(self, idx):
        joints_3d = self.get_joints_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(joints_3d, cam_intr)

    def get_joints_3d(self, idx):
        seq, img_idx = self.seq_idx[idx]
        if self.data_split == "train":
            annot = self.annot_mapping[seq][img_idx]
            joints_3d = annot["handJoints3D"]
            joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()
            joints_3d = joints_3d[self.reorder_idxs]
            return joints_3d.astype(np.float32)

        if self.data_split == "test":
            test_gt = self._get_test_gt_entry(seq, img_idx)
            if test_gt is not None:
                if test_gt["source"] == "manual_test_gt":
                    return test_gt["target_joints_3d"].astype(np.float32)
                joints_3d = self.official_eval_xyz[test_gt["official_idx"]]
                joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()
                joints_3d = joints_3d[self.reorder_idxs]
                return joints_3d.astype(np.float32)

            annot = self.annot_mapping[seq][img_idx]
            joints_3d = annot["handJoints3D"]
            joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()
            joints_3d = joints_3d[self.reorder_idxs]
            return joints_3d.astype(np.float32)

        raise RuntimeError(f"Unsupported data split for HO3D joints: {self.data_split}")

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_cam_intr(self, idx):
        seq, img_idx = self.seq_idx[idx]
        cam_intr = self.annot_mapping[seq][img_idx]["camMat"]
        return cam_intr.astype(np.float32)

    def _get_obj_pose_cam(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_rot_aa = np.asarray(annot["objRot"], dtype=np.float32).reshape(1, 3)
        obj_trans = np.asarray(annot["objTrans"], dtype=np.float32).reshape(3, 1)
        rot_raw = np.asarray(aa_to_rotmat(obj_rot_aa), dtype=np.float32).reshape(3, 3)
        rot_cam = self.cam_extr[:3, :3] @ rot_raw
        trans_cam = self.cam_extr[:3, :3] @ obj_trans
        return rot_cam.astype(np.float32), trans_cam.astype(np.float32)

    def get_obj_pose_and_points(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_name = str(annot["objName"])
        obj_id = int(annot["objLabel"])

        if obj_id not in self.obj_pc_sparse:
            obj_id = self.obj_name_to_id[obj_name]

        rot_cam, trans_cam = self._get_obj_pose_cam(idx)
        hand_joints_3d = self.get_joints_3d(idx)
        hand_root_cam = hand_joints_3d[self.center_idx].reshape(3, 1)
        t_label_rel = (trans_cam - hand_root_cam).astype(np.float32)
        rot6d_label = rotmat_to_rot6d(torch.from_numpy(rot_cam).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)

        pc_sparse_rest = self.obj_pc_sparse[obj_id]
        pc_dense_rest = self.obj_pc_dense[obj_id]
        pc_eval_rest = self.obj_pc_eval[obj_id]
        kp21_rest = self.obj_kp21_rest[obj_id]

        pc_sparse_cam = (rot_cam @ pc_sparse_rest.T).T + trans_cam.flatten()
        pc_dense_cam = (rot_cam @ pc_dense_rest.T).T + trans_cam.flatten()
        kp21_cam = (rot_cam @ kp21_rest.T).T + trans_cam.flatten()

        return {
            "obj_id": obj_id,
            "R_label": rot_cam,
            "t_label_rel": t_label_rel.flatten(),
            "rot6d_label": rot6d_label,
            "sparse_rest": pc_sparse_rest,
            "eval_rest": pc_eval_rest,
            "kp21_rest": kp21_rest,
            "kp21_cam": kp21_cam.astype(np.float32),
            "sparse_cam": pc_sparse_cam.astype(np.float32),
            "dense_cam": pc_dense_cam.astype(np.float32),
        }

    def get_obj_info(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_name = str(annot["objName"])
        obj_id = int(annot["objLabel"])
        if obj_id not in self.obj_pc_sparse:
            obj_id = self.obj_name_to_id[obj_name]

        rot_cam, trans_cam = self._get_obj_pose_cam(idx)
        obj_transform = np.eye(4, dtype=np.float32)
        obj_transform[:3, :3] = rot_cam
        obj_transform[:3, 3] = trans_cam.flatten()
        return obj_id, obj_transform

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_mano_pose(self, idx):
        seq, img_idx = self.seq_idx[idx]
        if self.data_split == "train":
            annot = self.annot_mapping[seq][img_idx]
            handpose = annot["handPose"]
            root, remains = handpose[:3], handpose[3:]
            root = rotmat_to_aa(self.cam_extr[:3, :3] @ aa_to_rotmat(root))
            handpose_transformed = np.concatenate((root, remains), axis=0)
            return handpose_transformed.astype(np.float32)

        if self.data_split == "test":
            test_gt = self._get_test_gt_entry(seq, img_idx)
            if test_gt is not None:
                if test_gt["source"] == "manual_test_gt":
                    return test_gt["mano_pose"].astype(np.float32)
                return np.zeros(48, dtype=np.float32)

            annot = self.annot_mapping[seq][img_idx]
            handpose = annot["handPose"]
            root, remains = handpose[:3], handpose[3:]
            root = rotmat_to_aa(self.cam_extr[:3, :3] @ aa_to_rotmat(root))
            handpose_transformed = np.concatenate((root, remains), axis=0)
            return handpose_transformed.astype(np.float32)

        raise RuntimeError(f"Unsupported data split for HO3D MANO pose: {self.data_split}")

    def get_mano_shape(self, idx):
        seq, img_idx = self.seq_idx[idx]
        if self.data_split == "train":
            annot = self.annot_mapping[seq][img_idx]
            shape = annot["handBeta"]
            shape = np.array(shape, dtype=np.float32)
            return shape

        if self.data_split == "test":
            test_gt = self._get_test_gt_entry(seq, img_idx)
            if test_gt is not None:
                if test_gt["source"] == "manual_test_gt":
                    return test_gt["mano_shape"].astype(np.float32)
                return np.zeros(10, dtype=np.float32)

            annot = self.annot_mapping[seq][img_idx]
            shape = annot["handBeta"]
            shape = np.array(shape, dtype=np.float32)
            return shape

        raise RuntimeError(f"Unsupported data split for HO3D MANO shape: {self.data_split}")

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_bone_scale(self, idx):
        joints_3d = self.get_joints_3d(idx)
        bone_len = batch_ref_bone_len(np.expand_dims(joints_3d, axis=0)).squeeze(0)
        return bone_len.astype(np.float32)

    def get_verts_3d(self, idx):
        if self.data_split == "train":
            _handpose, _handtsl, _handshape = self._ho3d_get_hand_info(idx)
            mano_out = self.mano_layer(
                torch.from_numpy(_handpose).unsqueeze(0),
                torch.from_numpy(_handshape).unsqueeze(0),
            )
            # important modify!!!!
            handverts = mano_out.verts[0].numpy() + _handtsl
            transf_handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
            return transf_handverts.astype(np.float32)

        if self.data_split == "test":
            seq, img_idx = self.seq_idx[idx]
            test_gt = self._get_test_gt_entry(seq, img_idx)
            if test_gt is not None:
                if test_gt["source"] == "manual_test_gt":
                    return test_gt["target_verts_3d"].astype(np.float32)
                handverts = self.official_eval_verts[test_gt["official_idx"]]
                handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
                return handverts.astype(np.float32)

            _handpose, _handtsl, _handshape = self._ho3d_get_hand_info(idx)
            mano_out = self.mano_layer(
                torch.from_numpy(_handpose).unsqueeze(0),
                torch.from_numpy(_handshape).unsqueeze(0),
            )
            handverts = mano_out.verts[0].numpy() + _handtsl
            transf_handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
            return transf_handverts.astype(np.float32)

        raise RuntimeError(f"Unsupported data split for HO3D verts: {self.data_split}")

    def get_verts_2d(self, idx):
        verts_3d = self.get_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(verts_3d, cam_intr)

    def get_hand_faces(self, idx):
        faces = np.array(self.mano_layer.th_faces).astype(np.long)
        return faces

    def get_bbox_center_scale(self, idx):
        # Only use hand joints or hand bbox
        if self.data_split == "train" or (self.data_split == "test" and self.split_mode == "v1"):
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            center = get_annot_center(joints2d)
            scale = get_annot_scale(joints2d)
            return center, scale
        elif self.data_split == "test":  # No gt joints annot, using handBoundingBox
            seq, img_idx = self.seq_idx[idx]
            if self._get_test_gt_entry(seq, img_idx) is not None:
                joints2d = self.get_joints_2d(idx)
                center = get_annot_center(joints2d)
                scale = get_annot_scale(joints2d)
                return center, scale
            annot = self.annot_mapping[seq][img_idx]
            hand_bbox_coord = annot["handBoundingBox"]  # (x0, y0, x1, y1)
            hand_bbox_2d = np.array(
                [
                    [hand_bbox_coord[0], hand_bbox_coord[1]],
                    [hand_bbox_coord[2], hand_bbox_coord[3]],
                ],
                dtype=np.float32,
            )
            center = get_annot_center(hand_bbox_2d)
            scale = get_annot_scale(hand_bbox_2d)
            scale = scale
            return center, scale
        else:
            raise RuntimeError()

    def _ho3d_get_hand_info(self, idx):
        """
        Get the hand annotation in the raw ho3d datasets.
        !!! This Mehthods shoudln't be called outside.
        :param idx:
        :return: raw hand pose, translate and shape coefficients
        """
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        # Retrieve hand info
        handpose = annot["handPose"]
        handtsl = annot["handTrans"]
        handshape = annot["handBeta"]
        handpose = handpose.astype(np.float32)
        return handpose, handtsl, handshape

    def get_hand_vis2d(self, idx):
        handvis = np.ones_like(self.get_verts_2d(idx)[:, 0])
        return handvis

    def get_annot(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        return annot

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    # The following functions only used in offline eval
    def get_hand_pose_wrt_cam(self, idx):  # pose = root_rot + ...
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        handpose = annot["handPose"]
        # only the first 3 dimension needs to be transformed by cam_extr
        root, remains = handpose[:3], handpose[3:]
        root = rotmat_to_aa(self.cam_extr[:3, :3] @ aa_to_rotmat(root))
        handpose_transformed = np.concatenate((root, remains), axis=0)
        return handpose_transformed.astype(np.float32)

    def get_hand_tsl_wrt_cam(self, idx):
        hand_pose = torch.from_numpy(self.get_hand_pose_wrt_cam(idx)).unsqueeze(0)
        hand_shape = torch.from_numpy(self.get_hand_shape(idx)).unsqueeze(0)

        mano_out = self.mano_layer(hand_pose, hand_shape)
        hand_verts = np.array(mano_out.verts.squeeze(0))
        tsl = self.get_verts_3d(idx) - hand_verts
        return tsl[0]

    def get_hand_axisang_wrt_cam(self, idx):
        rootRot = self.get_hand_rot_wrt_cam(idx)
        root = rotmat_to_aa(rootRot)
        return root.astype(np.float32)

    def get_hand_rot_wrt_cam(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        handpose = annot["handPose"]
        # only the first 3 dimension needs to be transformed by cam_extr
        root = handpose[:3]
        rootRot = self.cam_extr[:3, :3] @ aa_to_rotmat(root)
        return rootRot.astype(np.float32)

    def get_hand_shape(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        handshape = annot["handBeta"]
        return handshape.astype(np.float32)

    def get_hand_pose(self, idx):
        return self.get_hand_pose_wrt_cam(idx)

    def get_hand_tsl(self, idx):
        return self.get_hand_tsl_wrt_cam(idx)


# new content in HO3DV3 has been add to HO3D, while HO3DV3 can still be used
@DATASET.register_module()
class HO3DV3(HO3D):

    def __init__(self, *args, **kwargs):
        warnings.warn("New content in HO3DV3 has been add to HO3D, please use HO3D instead")
        super(HO3DV3, self).__init__(*args, **kwargs)

    def _preload(self):
        # deal with all the naming and path convention
        self.name = "HO3D_v3"
        self.root = os.path.join(self.data_root, self.name)
        self.root_extra_info = os.path.normpath("assets")
        assert self.split_mode == "paper", "HO3D_v3 only support paper split."

        self.cache_identifier_dict = {
            "data_split": self.data_split,
            "split_mode": self.split_mode,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def _load_annots(self, obj_meshes={}, seq_frames=[], subfolder="train"):
        annot_mapping, seq_idx = super()._load_annots(obj_meshes=obj_meshes, seq_frames=seq_frames, subfolder=subfolder)
        for seq in annot_mapping.values():
            for annot in seq:
                annot["img"] = annot["img"].replace(".png", ".jpg")  # In HO3D V3, image is given in the form of jpg
        return annot_mapping, seq_idx

    def _ho3d_get_hand_info(self, idx):
        handpose, handtsl, handshape = super()._ho3d_get_hand_info(idx)
        return (
            np.asfarray(handpose, dtype=np.float32),
            np.asfarray(handtsl, dtype=np.float32),
            np.asfarray(handshape, dtype=np.float32),
        )


@DATASET.register_module()
class HO3Dv3MultiView(torch.utils.data.Dataset):

    def __init__(self, cfg):

        self.name = type(self).__name__
        self.cfg = cfg
        self.n_views = cfg.N_VIEWS
        self.data_split = cfg.DATA_SPLIT
        self.mv_split_mode = str(cfg.get("MV_SPLIT_MODE", "legacy")).lower()
        self.add_evalset_train = cfg.get("ADD_EVALSET_TRAIN", True)
        assert self.data_split in ["train", "val", "test"], f"{self.name} unsupport data split {self.data_split}"

        self.master_system = cfg.MASTER_SYSTEM
        assert self.master_system == "as_constant_camera", f"{self.name} only support as_constant_camera master system"
        self.const_cam_id = cfg.CONST_CAM_ID

        self.data_mode = cfg.DATA_MODE
        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.split_mode = cfg.SPLIT_MODE
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        _trainset, _testset = self._single_view_ho3d()

        self.set_mappings = {f"{cfg.SPLIT_MODE}_train": _trainset, f"{cfg.SPLIT_MODE}_test": _testset}
        self.root = _trainset.root
        self.seq_multiview_dict, self.seq_eval = self._build_mv_split()

        # 10: side_view_facing_whiteboard
        # 11: top_view_facing_desk
        # 12: front_view_facing_wall
        # 13: side_view_facing_screens
        # 14: ego_view_facing_desk
        self.view_name = [
            'side_view_facing_whiteboard',
            'top_view_facing_desk',
            'front_view_facing_wall',
            'side_view_facing_screens',
            'ego_view_facing_desk',
        ]

        self.multivew_mapping = {}
        self.multiview_sample_idxs = []
        self.multiview_sample_infos = []

        seq_frames, subfolder = self._load_seq_frames_multiview()

        if self.split_mode in ['paper', 'v2', 'v3']:  # full view mode
            source_set_name = self._resolve_mv_source_set_name()
            self._mapping_multiview(seq_frames=seq_frames)

            if self.data_split == "test" and self.mv_split_mode == "legacy" and self.add_evalset_train:
                info_path_eval = os.path.join(self.root, "evaluation.txt")
                with open(info_path_eval, "r") as f:
                    lines = f.readlines()
                seq_frames_eval = [line.strip().split("/") for line in lines]
                self._mapping_multiview(seq_frames=seq_frames_eval)

            self._mapping_idxs_infos(source_set_name=source_set_name, subfolder=subfolder)

        else:
            raise ValueError(f"{self.split_mode} is not supported")

        logger.warning(
            f"{self.name} {self.mv_split_mode}_{self.split_mode}_{self.data_split} Init Done. "
            f"{len(self.multiview_sample_idxs)} samples")

    def _build_mv_split(self):
        legacy_train = {
            "ABF10": 0, "ABF11": 1, "ABF12": 2, "ABF13": 3, "ABF14": 4,
            "BB10": 5, "BB11": 6, "BB12": 7, "BB13": 8, "BB14": 9,
            "GSF10": 15, "GSF11": 16, "GSF12": 17, "GSF13": 18, "GSF14": 19,
            "MDF10": 20, "MDF11": 21, "MDF12": 22, "MDF13": 23, "MDF14": 24,
            "SiBF10": 25, "SiBF11": 26, "SiBF12": 27, "SiBF13": 28, "SiBF14": 29,
        }
        legacy_test = {
            "GPMF10": 10, "GPMF11": 11, "GPMF12": 12, "GPMF13": 13, "GPMF14": 14,
            "SB10": 30, "SB11": 31, "SB12": 32, "SB13": 33, "SB14": 34,
        }
        artiboost_like_train = {
            "ABF10": 0, "ABF11": 1, "ABF12": 2, "ABF13": 3, "ABF14": 4,
            "BB10": 5, "BB11": 6, "BB12": 7, "BB13": 8, "BB14": 9,
            "GPMF10": 10, "GPMF11": 11, "GPMF12": 12, "GPMF13": 13, "GPMF14": 14,
            "GSF10": 15, "GSF11": 16, "GSF12": 17, "GSF13": 18, "GSF14": 19,
            "MDF10": 20, "MDF11": 21, "MDF12": 22, "MDF13": 23, "MDF14": 24,
            "SiBF10": 25, "SiBF11": 26, "SiBF12": 27, "SiBF13": 28, "SiBF14": 29,
        }
        artiboost_like_test = {
            "AP10": 30, "AP11": 31, "AP12": 32, "AP13": 33, "AP14": 34,
            "MPM10": 35, "MPM11": 36, "MPM12": 37, "MPM13": 38, "MPM14": 39,
        }

        if self.mv_split_mode in ["legacy", "poem"]:
            seq_multiview_dict = legacy_train if self.data_split == "train" else legacy_test
            seq_eval = {"SB11": 31, "SB13": 33} if self.add_evalset_train else {}
            return seq_multiview_dict, seq_eval

        if self.mv_split_mode == "artiboost_like":
            seq_multiview_dict = artiboost_like_train if self.data_split == "train" else artiboost_like_test
            return seq_multiview_dict, {}

        raise ValueError(f"Unsupported MV_SPLIT_MODE: {self.mv_split_mode}")

    def _resolve_mv_source_set_name(self):
        if self.mv_split_mode == "artiboost_like":
            split_name = "train" if self.data_split == "train" else "test"
            return f"{self.split_mode}_{split_name}"
        return f"{self.split_mode}_train"

    def _single_view_ho3d(self):
        cfg_train = dict(
            TYPE="HO3D",
            DATA_SPLIT="train",
            DATA_MODE=self.data_mode,
            SPLIT_MODE=self.split_mode,
            DATA_ROOT=self.cfg.DATA_ROOT,
            TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET,
            USE_OFFICIAL_EVAL_GT_JSON=bool(self.cfg.get("USE_OFFICIAL_EVAL_GT_JSON", False)),
        )

        cfg_test = cfg_train.copy()
        cfg_test["DATA_SPLIT"] = "test"
        if self.add_evalset_train:
            cfg_test["USE_GT_FROM_MULTIVIEW"] = True

        ho3d_train = HO3D(CN(cfg_train))
        ho3d_test = HO3D(CN(cfg_test))

        return ho3d_train, ho3d_test

    def _mapping_multiview(self, seq_frames):
        for i, (seq, frame_idx) in enumerate(etqdm(seq_frames)):
            if seq in self.seq_multiview_dict.keys():
                seq_id = self.seq_multiview_dict[seq]
                cam_id = seq_id % self.n_views
                seq_id_main = (seq_id // self.n_views) * self.n_views
                if (seq_id_main, frame_idx) not in self.multivew_mapping:
                    self.multivew_mapping[(seq_id_main, frame_idx)] = [(cam_id, i)]
                else:
                    self.multivew_mapping[(seq_id_main, frame_idx)].append((cam_id, i))

    def _mapping_idxs_infos(self, source_set_name, subfolder):
        # 10: side_view_facing_whiteboard
        # 11: top_view_facing_desk
        # 12: front_view_facing_wall
        # 13: side_view_facing_screens
        # 14: ego_view_facing_desk
        for key, value in self.multivew_mapping.items():
            seq_id_main, frame_idx = key
            for k, v in self.seq_multiview_dict.items():
                if seq_id_main == v:
                    seq_name_main = k[:-1]
            self.multiview_sample_idxs.append([i for (_, i) in value])
            self.multiview_sample_infos.append([{
                "set_name": source_set_name if not (seq_id_main + cam_id) in self.seq_eval.values() else "paper_test",
                "seq_id_main": seq_id_main,
                "seq_name": seq_name_main + f"{cam_id}",
                "seq_name_main": seq_name_main,
                "cam_id": cam_id,
                "frame_idx": frame_idx,
                "subfolder": subfolder if not (seq_id_main + cam_id) in self.seq_eval.values() else "evaluation",
                "seq_id": seq_id_main + cam_id,
                "view_name": self.view_name[cam_id]
            } for (cam_id, _) in value])

    def _load_seq_frames_multiview(self, subfolder=None, seqs=None, trainval_idx=6000):
        """
        trainval_idx (int): How many frames to include in training split when
                using trainval/val/test split
        """
        if self.split_mode == "paper":
            if self.data_split in ["train", "trainval", "val"]:
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                if self.mv_split_mode == "artiboost_like":
                    info_path = os.path.join(self.root, "evaluation.txt")
                    subfolder = "evaluation"
                else:
                    info_path = os.path.join(self.root, "train.txt")
                    subfolder = "train"
            else:
                assert False
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
            if self.data_split == "trainval":
                seq_frames = seq_frames[:trainval_idx]
            elif self.data_split == "val":
                seq_frames = seq_frames[trainval_idx:]
        else:
            assert False
        return seq_frames, subfolder

    def __len__(self):
        return len(self.multiview_sample_idxs)

    def _testing(self, pose, joints, verts, shape, image, cam_intr):
        # this funciton just for testing in developing
        # this function shoundn't be used in normal
        # you can ues this function to visualization verts
        shape = torch.tensor(shape).unsqueeze(0)  # (1, 10)
        pose = torch.tensor(pose).unsqueeze(0)  # (1, 48)
        self.mano_layer = ManoLayer(
            side="right",
            rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=0,
            flat_hand_mean=True,
        )
        mano_results: MANOOutput = self.mano_layer(pose, shape)
        verts_pre = mano_results.verts
        joints_pre = mano_results.joints
        verts_pre = torch.tensor(verts_pre).squeeze(0)  # (778, 3)
        joints_pre = torch.tensor(joints_pre).squeeze(0)  # (21, 3)
        joints_root = torch.tensor(joints[0])
        joints_root = joints_root.view(1, 3).repeat(778, 1)  # (778, 3)

        verts_3d = verts_pre + joints_root
        verts_3d = np.array(verts_3d)
        joints_2d_proj = persp_project(verts_3d, cam_intr)
        verts_proj = persp_project(verts, cam_intr)

        image = denormalize(image, [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
        image = (image * 255.0).astype(np.uint8)
        frame_0 = image.copy()
        frame_1 = image.copy()
        for i in range(verts_proj.shape[0]):
            cx = int(verts_proj[i, 0])
            cy = int(verts_proj[i, 1])
            cv2.circle(frame_0, (cx, cy), radius=2, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)
        for i in range(joints_2d_proj.shape[0]):
            cx = int(joints_2d_proj[i, 0])
            cy = int(joints_2d_proj[i, 1])
            cv2.circle(frame_1, (cx, cy), radius=2, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)
        img_list = [image, frame_0, frame_1]
        comb_image = np.hstack(img_list)
        imageio.imwrite("tmp/img.png", comb_image)
        time.sleep(2)

    def _calculate_save_eval_from_train(self, sample, multiview_info_list):
        # this funciton only used in developing
        # this function shoundn't be used in normal
        # dataset in "evaluation" need calculation manually
        if multiview_info_list[0]["seq_name_main"] == "SB1":
            i = 0
            while i < self.n_views:
                if sample["cam_serial"][i] not in self.seq_eval.keys():
                    break
                i += 1
            tempo_master_joints_3d = sample["target_joints_3d"][i]
            tempo_master_verts_3d = sample["target_verts_3d"][i]
            tempo_master_mano_pose = sample["mano_pose"][i]

            for j in range(self.n_views):
                if sample["cam_serial"][j] in self.seq_eval.keys():
                    curr_T_m2c = np.linalg.inv(sample["target_cam_extr"][i]) @ sample["target_cam_extr"][j]
                    curr_T_c2m = np.linalg.inv(curr_T_m2c)
                    curr_cam_intr = sample["target_cam_intr"][j]
                    sample["target_joints_3d"][j] = SE3_transform(tempo_master_joints_3d, curr_T_c2m)
                    sample["target_joints_2d"][j] = persp_project(sample["target_joints_3d"][j], curr_cam_intr)
                    sample["target_joints_uvd"][j] = np.concatenate(
                        (sample["target_joints_2d"][j], sample["target_joints_3d"][j][:, 2:]), axis=1)
                    sample["target_verts_3d"][j] = SE3_transform(tempo_master_verts_3d, curr_T_c2m)
                    v3d = sample["target_verts_3d"][j]
                    v_uv = persp_project(v3d, curr_cam_intr)[:, :2]
                    sample["target_verts_uvd"][j] = np.concatenate((v_uv, v3d[:, 2:]), axis=1)
                    sample["mano_shape"][j] = sample["mano_shape"][i]

                    # print(sample["mano_pose"][j][:3],0)
                    r_pose = curr_T_c2m[:3, :3]
                    handpose = sample["mano_pose"][i]
                    root, remains = handpose[:3], handpose[3:]
                    root = rotmat_to_aa(r_pose @ aa_to_rotmat(root))
                    sample["mano_pose"][j] = np.concatenate((root, remains), axis=0)
                    sample["mano_pose"][j].astype(np.float32)
                    # print(sample["mano_pose"][j][:3],1)

                    meta_map = {}
                    meta_map["target_joints_3d"] = sample["target_joints_3d"][j]
                    meta_map["target_verts_3d"] = sample["target_verts_3d"][j]
                    meta_map["mano_pose"] = sample["mano_pose"][j]
                    meta_map["mano_shape"] = sample["mano_shape"][j]
                    id_frame = multiview_info_list[j]["frame_idx"]

                    meta_root = os.path.join("./tmp/HO3Dv3_test_gt", sample["cam_serial"][j], "meta",
                                             "{}.pkl".format(id_frame))
                    meta_dir = os.path.dirname(meta_root)
                    os.makedirs(meta_dir, exist_ok=True)

                    with open(meta_root, "wb") as p_f:
                        pickle.dump(meta_map, p_f)

    def __getitem__(self, idx):
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]

        seq_name_main = multiview_info_list[0]["seq_name_main"]
        extr_mapping = {}

        for i in range(len(multiview_id_list)):
            cam_id = int(multiview_info_list[i]["cam_id"])
            extr_seq_dir = os.path.join(self.root, "calibration", seq_name_main, "calibration", f"trans_{cam_id}.txt")
            with open(extr_seq_dir) as f:
                extr = np.loadtxt(f, dtype=np.float32)
            extr_mapping[cam_id] = extr

        # get true cam_id, the cam_id above just for convenient index
        true_cam_order_dir = os.path.join(self.root, "calibration", seq_name_main, "calibration", "cam_orders.txt")
        true_cam_orders = [int(float(number)) for line in open(true_cam_order_dir, 'r') for number in line.split()]

        sample = dict()
        sample["sample_idx"] = multiview_id_list
        sample["target_cam_extr"] = list()
        sample["cam_serial"] = list()

        available_cam_ids = sorted({int(info["cam_id"]) for info in multiview_info_list})
        idx_list = [str(cam_id) for cam_id in available_cam_ids]
        const_v_id = -101
        for vi, info in enumerate(multiview_info_list):
            if info["cam_id"] == self.const_cam_id:
                const_v_id = vi
                break
        assert const_v_id != -101, f"Cannot find the constant camera serial {self.const_cam_id} in the dataset"
        idx_list.remove(str(self.const_cam_id))
        idx_list.insert(0, str(self.const_cam_id))

        for idx in idx_list:
            for i, info in zip(multiview_id_list, multiview_info_list):
                if info["seq_name"][-1] == idx:
                    if idx == '2':
                        assert info["view_name"] == "front_view_facing_wall", \
                            "the cam of id 2 must be 'front_view_facing_wall'"
                    # get the source set -- one of the  HO3D s#_train, s#_val and s#_test.
                    source_set = self.set_mappings[info["set_name"]]
                    cam_id = int(info["seq_name"][-1])
                    # change 2 the true cam_id
                    if cam_id == 0:
                        cam_id = true_cam_orders[0]
                    elif cam_id == 1:
                        cam_id = true_cam_orders[1]
                    elif cam_id == 2:
                        cam_id = true_cam_orders[2]
                    elif cam_id == 3:
                        cam_id = true_cam_orders[3]
                    elif cam_id == 4:
                        cam_id = true_cam_orders[4]

                    T_master_2_cam = extr_mapping.get(cam_id, extr_mapping[int(info["cam_id"])])
                    sample["target_cam_extr"].append(T_master_2_cam)

                    cam_serial = info["seq_name"]
                    sample["cam_serial"].append(cam_serial)

                    # get sample from the source set. (HO3D's getitem)
                    src_sample = source_set[i]
                    for query, value in src_sample.items():
                        if query in sample:
                            sample[query].append(value)
                        else:
                            sample[query] = [value]

        # >>>>>>>>>>>
        # you can use self._testing() to test here
        # <<<<<<<<<<<

        # set a new master
        if self.master_system == "as_first_camera":
            new_master_id = 0
            new_master_serial = sample["cam_serial"][new_master_id]
            T_master_2_new_master = sample["target_cam_extr"][new_master_id]
            master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
            master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]
            master_obj_kp21 = sample["target_obj_kp21_no_rot"][new_master_id]
            master_obj_kp21_rest = sample["target_obj_kp21_rest"][new_master_id]
            master_obj_sparse = sample["target_obj_sparse_no_rot"][new_master_id]
            master_obj_sparse_rest = sample["target_obj_pc_sparse_rest"][new_master_id]
            master_obj_eval_rest = sample["target_obj_pc_eval_rest"][new_master_id]
            master_obj_rot6d_label = sample["target_rot6d_label"][new_master_id]
            master_obj_t_label_rel = sample["target_t_label_rel"][new_master_id]

        elif self.master_system == "as_constant_camera":
            new_master_serial = sample["cam_serial"][0][:-1] + f"{self.const_cam_id}"
            new_master_id = sample["cam_serial"].index(new_master_serial)
            T_master_2_new_master = sample["target_cam_extr"][new_master_id]
            master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
            master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]
            master_obj_kp21 = sample["target_obj_kp21_no_rot"][new_master_id]
            master_obj_kp21_rest = sample["target_obj_kp21_rest"][new_master_id]
            master_obj_sparse = sample["target_obj_sparse_no_rot"][new_master_id]
            master_obj_sparse_rest = sample["target_obj_pc_sparse_rest"][new_master_id]
            master_obj_eval_rest = sample["target_obj_pc_eval_rest"][new_master_id]
            master_obj_rot6d_label = sample["target_rot6d_label"][new_master_id]
            master_obj_t_label_rel = sample["target_t_label_rel"][new_master_id]

        for i, T_m2c in enumerate(sample["target_cam_extr"]):
            T_new_master_2_cam = np.linalg.inv(T_master_2_new_master) @ T_m2c
            extr_prerot = sample["extr_prerot"][i]  # (3, 3)
            extr_pre_transf = np.concatenate([extr_prerot, np.zeros((3, 1))], axis=1)
            extr_pre_transf = np.concatenate([extr_pre_transf, np.array([[0, 0, 0, 1]])], axis=0)

            T_new_master_2_cam = np.linalg.inv(extr_pre_transf @ np.linalg.inv(T_new_master_2_cam))
            sample["target_cam_extr"][i] = T_new_master_2_cam.astype(np.float32)

        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        sample["master_id"] = new_master_id
        sample["master_serial"] = new_master_serial
        sample["master_joints_3d"] = master_joints_3d
        sample["master_verts_3d"] = master_verts_3d
        sample["master_obj_kp21"] = master_obj_kp21
        sample["master_obj_kp21_rest"] = master_obj_kp21_rest
        sample["master_obj_sparse"] = master_obj_sparse
        sample["master_obj_sparse_rest"] = master_obj_sparse_rest
        sample["master_obj_eval_rest"] = master_obj_eval_rest
        sample["master_obj_rot6d_label"] = master_obj_rot6d_label
        sample["master_obj_t_label_rel"] = master_obj_t_label_rel
        if "target_obj_pc_eval_rest" in sample:
            del sample["target_obj_pc_eval_rest"]

        for query, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[query] = np.ascontiguousarray(value)

        return sample


@DATASET.register_module()
class HO3Dv3MultiView_Video(HO3Dv3MultiView):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.name = type(self).__name__
        # self.cfg = cfg
        self.seq_len = cfg.SEQ_LEN
        self.drop_last_frames = cfg.get("DROP_LAST_FRAMES", True)
        self.interval_frames = cfg.get("INTERVAL_FRAMES", 0)

        self._load_video_frames()

        assert self.master_system == "as_constant_camera", f"{self.name} only support master system mode 'as_constant_camera' "
        logger.warning(
            f"{self.name} {self.split_mode}_{self.data_split} Init Done. {len(self.multiview_video_sample_idxs)} samples"
        )

    def _load_video_frames(self):
        if self.data_split == 'train':
            with open('./assets/video_task/ho3dv3_multiview_video_idxs_train.pkl', 'rb') as f_idx:
                all_multiview_samples = pickle.load(f_idx)
        elif self.data_split == 'test':
            with open('./assets/video_task/ho3dv3_multiview_video_idxs_test.pkl', 'rb') as f_idx:
                all_multiview_samples = pickle.load(f_idx)
        else:
            raise ValueError(f"Don't supported data_split: {self.data_split}")
        if self.interval_frames != 0:
            all_multiview_samples = all_multiview_samples[::self.interval_frames]

        self.multiview_video_sample_idxs = []
        for i in range(len(all_multiview_samples)):
            if all_multiview_samples[i][-1] == all_multiview_samples[i + self.seq_len - 1][-1]:
                tmp_list = []
                for j in range(i, i + self.seq_len):
                    tmp_list.append(all_multiview_samples[j])
                self.multiview_video_sample_idxs.append(tmp_list)
            if i + self.seq_len == len(all_multiview_samples):
                break

    def __len__(self):
        return len(self.multiview_video_sample_idxs)

    def _save_tmp(self, idx):
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]
        tmp = []
        for i in multiview_info_list:
            tmp.append(i['seq_name'])
        save_list = []
        save_list.append(idx)
        save_list.append(multiview_id_list)
        save_list.append(tmp)

        return save_list

    def __getitem__(self, idx):
        multiview_video_id_list = self.multiview_video_sample_idxs[idx]  # [[idx,[single_idxs],[seq_names]] * seq_len]
        sample = dict()
        for i in range(self.seq_len):
            # get sample from the source set. (HO3Dv3MultiView's getitem)
            multiview_idx = multiview_video_id_list[i][0]
            i_sample = super().__getitem__(multiview_idx)
            for query, value in i_sample.items():
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]
        return sample
