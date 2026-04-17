import torch
import torch.nn as nn
import torch.nn.functional as F


class HFLHandBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HFLHandResidual(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.bn = nn.BatchNorm2d(self.num_in)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.num_in, self.num_out // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.num_out // 2)
        self.conv2 = nn.Conv2d(self.num_out // 2, self.num_out // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_out // 2)
        self.conv3 = nn.Conv2d(self.num_out // 2, self.num_out, bias=True, kernel_size=1)

        if self.num_in != self.num_out:
            self.conv4 = nn.Conv2d(self.num_in, self.num_out, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.num_in != self.num_out:
            residual = self.conv4(x)

        return out + residual


class HFLHandBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, skip=None, groups=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True, groups=groups)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.skip = skip

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.skip is not None:
            residual = self.skip(x)

        return out + residual


class HFLHourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super().__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        return nn.Sequential(*[block(planes * block.expansion, planes) for _ in range(num_blocks)])

    def _make_hourglass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2, mode="bilinear", align_corners=False)
        return up1 + up2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HFLHandHeatmapHead(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=128, blocks=1):
        super().__init__()
        self.out_res = int(roi_res)
        self.joint_nb = int(joint_nb)
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.stacks = int(stacks)

        self.betas = nn.Parameter(torch.ones((self.joint_nb, 1), dtype=torch.float32))

        center_offset = 0.5
        vv, uu = torch.meshgrid(
            torch.arange(self.out_res).float(),
            torch.arange(self.out_res).float(),
            indexing="ij",
        )
        uu, vv = uu + center_offset, vv + center_offset
        self.register_buffer("uu", uu / self.out_res)
        self.register_buffer("vv", vv / self.out_res)

        self.softmax = nn.Softmax(dim=2)
        block = HFLHandBottleneck
        self.features = self.channels // block.expansion

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.stacks):
            hg.append(HFLHourglass(block, self.blocks, self.features, 4))
            res.append(self.make_residual(block, self.channels, self.features, self.blocks))
            fc.append(HFLHandBasicBlock(self.channels, self.channels, kernel_size=1))
            score.append(nn.Conv2d(self.channels, self.joint_nb, kernel_size=1, bias=True))
            if i < self.stacks - 1:
                fc_.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(self.joint_nb, self.channels, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    @staticmethod
    def _init_conv_bias(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def make_residual(self, block, inplanes, planes, blocks, stride=1):
        skip = None
        if stride != 1 or inplanes != planes * block.expansion:
            skip = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True)
            )
        layers = [block(inplanes, planes, stride, skip)]
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def spatial_softmax(self, latents):
        latents = latents.view((-1, self.joint_nb, self.out_res ** 2))
        latents = latents * self.betas
        heatmaps = self.softmax(latents)
        heatmaps = heatmaps.view(-1, self.joint_nb, self.out_res, self.out_res)
        return heatmaps

    def generate_output(self, heatmaps):
        predictions = torch.stack(
            (
                torch.sum(torch.sum(heatmaps * self.uu, dim=2), dim=2),
                torch.sum(torch.sum(heatmaps * self.vv, dim=2), dim=2),
            ),
            dim=2,
        )
        return predictions

    def forward(self, x):
        out, encoding, preds = [], [], []
        for i in range(self.stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            latents = self.score[i](y)
            heatmaps = self.spatial_softmax(latents)
            out.append(heatmaps)
            preds.append(self.generate_output(heatmaps))
            if i < self.stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](heatmaps)
                x = x + fc_ + score_
                encoding.append(x)
            else:
                encoding.append(y)
        return out, encoding, preds


class HFLHandEncoder(nn.Module):
    def __init__(self, num_heatmap_chan, num_feat_chan, size_input_feature=(32, 32), n_reg_block=4, n_reg_modules=2):
        super().__init__()
        self.num_heatmap_chan = int(num_heatmap_chan)
        self.num_feat_chan = int(num_feat_chan)
        self.size_input_feature = size_input_feature
        self.n_reg_block = int(n_reg_block)
        self.n_reg_modules = int(n_reg_modules)

        self.heatmap_conv = nn.Conv2d(self.num_heatmap_chan, self.num_feat_chan, bias=True, kernel_size=1, stride=1)
        self.encoding_conv = nn.Conv2d(self.num_feat_chan, self.num_feat_chan, bias=True, kernel_size=1, stride=1)

        reg = []
        for _ in range(self.n_reg_block):
            for _ in range(self.n_reg_modules):
                reg.append(HFLHandResidual(self.num_feat_chan, self.num_feat_chan))
        self.reg = nn.ModuleList(reg)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_scale = 2 ** self.n_reg_block
        self.num_feat_out = self.num_feat_chan * (
            size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2)
        )

    def forward(self, hm_list, encoding_list):
        x = self.heatmap_conv(hm_list[-1]) + self.encoding_conv(encoding_list[-1])
        if len(encoding_list) > 1:
            x = x + encoding_list[-2]

        for i in range(self.n_reg_block):
            for j in range(self.n_reg_modules):
                x = self.reg[i * self.n_reg_modules + j](x)
            x = self.maxpool(x)

        return x.view(x.size(0), -1)
