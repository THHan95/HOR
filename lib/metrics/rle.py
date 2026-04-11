import math

import torch
import torch.nn as nn


class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, nf_loss, pred_jts, sigma, target_uv, vis=None):
        gt_uv = target_uv.reshape(pred_jts.shape)

        Q_logprob = self.logQ(gt_uv, pred_jts, sigma)
        loss = nf_loss + Q_logprob

        # Apply visibility mask if provided
        if vis is not None:
            # vis shape: (B*N, num_joints) or (B*N, 1)
            # loss shape: (B*N, num_joints, 2) or similar
            vis = vis.reshape(loss.shape[0], -1, 1)  # (B*N, num_joints, 1)
            loss = loss * vis

        if self.size_average:
            if vis is not None:
                # Average only over visible points
                valid_elements = vis.expand_as(loss).sum() 
                return loss.sum() / (valid_elements + 1e-9)
            else:
                return loss.mean() # 等价于 loss.sum() / loss.numel()
        else:
            return loss.sum()