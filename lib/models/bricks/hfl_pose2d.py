import torch
import torch.nn as nn


class HFLPose2DHead(nn.Module):
    def __init__(self, channels=256, inter_channels=None, joint_nb=21):
        super().__init__()
        if inter_channels is None:
            inter_channels = channels // 2

        self.conv1_1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1_1 = nn.BatchNorm2d(inter_channels)
        self.conv1_2 = nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(channels)

        self.conv2_1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2_1 = nn.BatchNorm2d(inter_channels)
        self.conv2_2 = nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(channels)

        self.conv3_1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3_1 = nn.BatchNorm2d(inter_channels)
        self.conv3_2 = nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(channels)

        self.out_conv = nn.Conv2d(channels, joint_nb * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        for layer in [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, self.conv3_1, self.conv3_2, self.out_conv]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.leaky_relu(self.bn1_1(self.conv1_1(x)))
        out = self.leaky_relu(self.bn1_2(self.conv1_2(out)))
        out = self.leaky_relu(self.bn2_1(self.conv2_1(out)))
        out = self.leaky_relu(self.bn2_2(self.conv2_2(out)))
        out = self.leaky_relu(self.bn3_1(self.conv3_1(out)))
        out = self.leaky_relu(self.bn3_2(self.conv3_2(out)))
        return self.out_conv(out)


class HFLPose2DDecoder(nn.Module):
    def __init__(self, joint_nb=21, coord_norm_factor=10.0):
        super().__init__()
        self.num_keypoints = joint_nb
        self.coord_norm_factor = float(coord_norm_factor)

    def forward(self, output):
        nB, _, nH, nW = output.shape
        nV = self.num_keypoints

        output = output.view(nB, 3 * nV, nH * nW).permute(1, 0, 2).contiguous().view(3 * nV, nB * nH * nW)

        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nH, nW, nV))
        x = output[nV:2 * nV].transpose(0, 1).view(nB, nH, nW, nV)
        y = output[2 * nV:3 * nV].transpose(0, 1).view(nB, nH, nW, nV)

        grid_x = ((torch.linspace(0, nW - 1, nW, device=output.device, dtype=output.dtype)
                   .repeat(nH, 1)
                   .repeat(nB * nV, 1, 1)
                   .view(nB, nV, nH, nW) + 0.5) / nW) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH, device=output.device, dtype=output.dtype)
                   .repeat(nW, 1).t()
                   .repeat(nB * nV, 1, 1)
                   .view(nB, nV, nH, nW) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 2, 3, 1).contiguous()
        grid_y = grid_y.permute(0, 2, 3, 1).contiguous()

        predx = (x + grid_x) / self.coord_norm_factor
        predy = (y + grid_y) / self.coord_norm_factor
        return predx, predy, conf


def select_pose2d_from_conf(predx, predy, conf):
    batch_size, feat_h, feat_w, num_points = conf.shape
    flat_size = feat_h * feat_w
    conf_flat = conf.view(batch_size, flat_size, num_points)
    best_idx = conf_flat.argmax(dim=1)

    predx_flat = predx.view(batch_size, flat_size, num_points)
    predy_flat = predy.view(batch_size, flat_size, num_points)

    gather_idx = best_idx.unsqueeze(1)
    best_x = torch.gather(predx_flat, 1, gather_idx).squeeze(1)
    best_y = torch.gather(predy_flat, 1, gather_idx).squeeze(1)
    best_conf = torch.gather(conf_flat, 1, gather_idx).squeeze(1)
    coords = torch.stack((best_x, best_y), dim=-1)
    return coords, best_conf


def dense_pose2d_hfl_loss(gt_uv, valid_mask, pred_maps, spatial_mask=None, loss_fn=None):
    predx, predy, pred_conf = pred_maps
    if loss_fn is None:
        loss_fn = nn.L1Loss()

    gt_uv = torch.nan_to_num(gt_uv, nan=0.0, posinf=1.0, neginf=0.0)
    valid_mask = valid_mask.to(dtype=predx.dtype)

    feat_h, feat_w = predx.shape[1:3]
    gt_x = gt_uv[..., 0].unsqueeze(1).unsqueeze(1).expand(-1, feat_h, feat_w, -1)
    gt_y = gt_uv[..., 1].unsqueeze(1).unsqueeze(1).expand(-1, feat_h, feat_w, -1)
    point_mask = valid_mask.unsqueeze(1).unsqueeze(1).expand(-1, feat_h, feat_w, -1)

    if point_mask.sum().item() <= 0:
        zero = predx.new_tensor(0.0)
        conf_target = pred_conf.new_zeros(pred_conf.shape)
        return zero, zero, conf_target

    reg_px = predx * point_mask
    reg_py = predy * point_mask
    reg_label_x = gt_x * point_mask
    reg_label_y = gt_y * point_mask

    bias = torch.sqrt((reg_py - reg_label_y) ** 2 + (reg_px - reg_label_x) ** 2 + 1e-9)
    conf_target = (torch.exp(-bias) * point_mask).detach()

    if spatial_mask is not None:
        if spatial_mask.dim() == 3:
            spatial_mask = spatial_mask.unsqueeze(-1)
        spatial_mask = spatial_mask.to(dtype=predx.dtype)
        reg_px = reg_px * spatial_mask
        reg_py = reg_py * spatial_mask
        reg_label_x = reg_label_x * spatial_mask
        reg_label_y = reg_label_y * spatial_mask
        pred_conf = pred_conf * spatial_mask
        conf_target = conf_target * spatial_mask

    reg_loss = loss_fn(reg_px, reg_label_x) + loss_fn(reg_py, reg_label_y)
    conf_loss = loss_fn(pred_conf, conf_target)
    return reg_loss, conf_loss, conf_target
