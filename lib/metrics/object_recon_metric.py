from typing import Dict

import torch
from pytorch3d.ops import knn_points

from .basic_metric import AverageMeter, Metric


class ObjectReconMetric(Metric):

    def __init__(self, cfg, name="ObjRec") -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.cd = AverageMeter(f"{name}_cd")
        self.fs_5 = AverageMeter(f"{name}_fs_5")
        self.fs_10 = AverageMeter(f"{name}_fs_10")
        self.chunk_size = 4096
        self.chunk_trigger = 8192
        self.reset()

    def reset(self):
        self.count = 0
        self.cd.reset()
        self.fs_5.reset()
        self.fs_10.reset()

    @staticmethod
    def _sanitize_points(points):
        points = points.detach().float()
        points = torch.nan_to_num(points, nan=0.0, posinf=10.0, neginf=-10.0)
        return points

    @staticmethod
    def _compute_fscore(dist_sq_pred_to_gt, dist_sq_gt_to_pred, threshold_m):
        threshold_sq = float(threshold_m) * float(threshold_m)
        precision = (dist_sq_pred_to_gt < threshold_sq).float().mean(dim=1)
        recall = (dist_sq_gt_to_pred < threshold_sq).float().mean(dim=1)
        return 2.0 * precision * recall / (precision + recall + 1e-7)

    def _knn_min_dist_sq(self, src_points, dst_points):
        if src_points.shape[1] <= self.chunk_trigger and dst_points.shape[1] <= self.chunk_trigger:
            return knn_points(src_points, dst_points, K=1, return_nn=False).dists[..., 0]

        batch_dists = []
        for batch_idx in range(src_points.shape[0]):
            src_single = src_points[batch_idx:batch_idx + 1]
            dst_single = dst_points[batch_idx:batch_idx + 1]
            chunk_dists = []
            for start_idx in range(0, src_single.shape[1], self.chunk_size):
                src_chunk = src_single[:, start_idx:start_idx + self.chunk_size]
                chunk_dist = knn_points(src_chunk, dst_single, K=1, return_nn=False).dists[..., 0]
                chunk_dists.append(chunk_dist.squeeze(0))
            batch_dists.append(torch.cat(chunk_dists, dim=0))
        return torch.stack(batch_dists, dim=0)

    def feed(self, pred_points, gt_points, **kwargs):
        pred_points = self._sanitize_points(pred_points)
        gt_points = self._sanitize_points(gt_points)

        valid_mask = torch.isfinite(pred_points).all(dim=2).all(dim=1) & torch.isfinite(gt_points).all(dim=2).all(dim=1)
        if not torch.any(valid_mask):
            return

        pred_points = pred_points[valid_mask]
        gt_points = gt_points[valid_mask]
        batch_size = pred_points.shape[0]

        pred_to_gt = self._knn_min_dist_sq(pred_points, gt_points)
        gt_to_pred = self._knn_min_dist_sq(gt_points, pred_points)

        # Points are in meters. Convert squared distances to cm^2 for HORT-style CD reporting.
        cd_cm2 = (pred_to_gt.mean(dim=1) + gt_to_pred.mean(dim=1)) * 10000.0
        fs_5 = self._compute_fscore(pred_to_gt, gt_to_pred, threshold_m=0.005)
        fs_10 = self._compute_fscore(pred_to_gt, gt_to_pred, threshold_m=0.010)

        self.cd.update(cd_cm2.sum().item(), batch_size)
        self.fs_5.update(fs_5.sum().item(), batch_size)
        self.fs_10.update(fs_10.sum().item(), batch_size)
        self.count += batch_size

    def get_measures(self, **kwargs) -> Dict[str, float]:
        return {
            f"{self.name}_cd": self.cd.avg,
            f"{self.name}_fs_5": self.fs_5.avg,
            f"{self.name}_fs_10": self.fs_10.avg,
        }

    def get_result(self):
        return self.cd.avg

    def __str__(self) -> str:
        return f"{self.name} FS@5/10 {self.fs_5.avg:.4f}/{self.fs_10.avg:.4f} | CD {self.cd.avg:.4f}"
