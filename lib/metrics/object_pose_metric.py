from typing import Dict

import torch

from ..utils.transform import rot6d_to_rotmat
from .basic_metric import AverageMeter, Metric


class ObjectPoseMetric(Metric):

    def __init__(self, cfg, name="Obj") -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.rot_deg = AverageMeter(f"{name}_rot_deg")
        self.trans_epe = AverageMeter(f"{name}_trans_epe")
        self.add = AverageMeter(f"{name}_add")
        self.adds = AverageMeter(f"{name}_adds")
        self.reset()

    def reset(self):
        self.count = 0
        self.rot_deg.reset()
        self.trans_epe.reset()
        self.add.reset()
        self.adds.reset()

    @staticmethod
    def _apply_pose(points, rot6d, trans):
        rotmat = rot6d_to_rotmat(rot6d)  # (B, 3, 3)
        posed = torch.matmul(rotmat, points.transpose(1, 2)).transpose(1, 2)
        posed = posed + trans.unsqueeze(1)
        return posed

    def feed(self,
             pred_rot6d,
             pred_trans,
             gt_rot6d,
             gt_trans,
             obj_points_rest,
             **kwargs):
        pred_rot6d = pred_rot6d.detach()
        pred_trans = pred_trans.detach()
        gt_rot6d = gt_rot6d.detach()
        gt_trans = gt_trans.detach()
        obj_points_rest = obj_points_rest.detach()
        batch_size = pred_rot6d.shape[0]

        pred_rotmat = rot6d_to_rotmat(pred_rot6d)
        gt_rotmat = rot6d_to_rotmat(gt_rot6d)
        rel_rotmat = torch.matmul(pred_rotmat, gt_rotmat.transpose(1, 2))
        trace = rel_rotmat[:, 0, 0] + rel_rotmat[:, 1, 1] + rel_rotmat[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        rot_deg = torch.rad2deg(torch.acos(cos_theta))

        trans_epe = torch.norm(pred_trans - gt_trans, dim=-1)

        pred_points = self._apply_pose(obj_points_rest, pred_rot6d, pred_trans)
        gt_points = self._apply_pose(obj_points_rest, gt_rot6d, gt_trans)
        add = torch.norm(pred_points - gt_points, dim=-1).mean(dim=-1)

        pairwise = torch.cdist(pred_points.float(), gt_points.float())
        adds = pairwise.min(dim=-1)[0].mean(dim=-1)

        self.rot_deg.update(rot_deg.sum().item(), batch_size)
        self.trans_epe.update(trans_epe.sum().item(), batch_size)
        self.add.update(add.sum().item(), batch_size)
        self.adds.update(adds.sum().item(), batch_size)
        self.count += batch_size

    def get_measures(self, **kwargs) -> Dict[str, float]:
        return {
            f"{self.name}_rot_deg": self.rot_deg.avg,
            f"{self.name}_trans_epe": self.trans_epe.avg,
            f"{self.name}_add": self.add.avg,
            f"{self.name}_adds": self.adds.avg,
        }

    def get_result(self):
        return self.adds.avg

    def __str__(self) -> str:
        return (
            f"{self.name} ADD/ADDS {self.add.avg * 1000.0:.1f}/{self.adds.avg * 1000.0:.1f} mm | "
            f"T {self.trans_epe.avg * 1000.0:.1f} mm | "
            f"R {self.rot_deg.avg:.1f} deg"
        )
