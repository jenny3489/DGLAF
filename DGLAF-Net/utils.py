import numpy as np
import torch
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn






class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()
        # 注意：拼接后通道数翻倍，需用1x1卷积降维
        self.local_conv = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.ea = External_attention(c=in_channel)

    def forward(self, x):
        # 局部注意力（并联版：CA和SA并行，结果拼接）
        x0, x1 = x.chunk(2, dim=2)
        x0_left, x0_right = x0.chunk(2, dim=3)
        x1_left, x1_right = x1.chunk(2, dim=3)

        def parallel_attention(patch):
            ca_out = self.ca(patch) * patch  # 通道注意力分支
            sa_out = self.sa(patch) * patch  # 空间注意力分支
            t_out = torch.cat([ca_out, sa_out], dim=1)
            # print(t_out)
            return t_out  # 并联融合（拼接）

        x0_left = self.local_conv(parallel_attention(x0_left))  # 拼接后降维
        x0_right = self.local_conv(parallel_attention(x0_right))
        x1_left = self.local_conv(parallel_attention(x1_left))
        x1_right = self.local_conv(parallel_attention(x1_right))

        # 重组局部区域
        x0 = torch.cat([x0_left, x0_right], dim=3)
        x1 = torch.cat([x1_left, x1_right], dim=3)
        x_local = torch.cat([x0, x1], dim=2)
        # print(x_local)

        # 全局注意力（保持原设计）
        # x_global = self.ca(x) * x
        # x_global = self.sa(x_global) * x_global
        x3 = self.ca(x) * x
        x4 = self.sa(x) * x
        x_global = torch.cat([x3, x4], dim=1)
        x_global = self.local_conv(x_global)
        x_global = self.ea(x_global)

        # 局部+全局融合
        x = x_local + x_global
        x = self.conv(x)
        return x









class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number. 官方的代码中设为512
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        norm_layer = nn.BatchNorm2d

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h * w
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        # linear_0是第一个memory unit
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n

        x = self.linear_1(attn)  # b, c, n
        # linear_1是第二个memory unit
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x