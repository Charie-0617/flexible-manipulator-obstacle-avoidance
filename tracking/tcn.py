# -*- coding: utf-8 -*-
"""
TCN 时序分支，用于输出动作均值 μ。

采用多层因果膨胀卷积堆叠残差块，最后通过全局平均池化和线性头输出。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """单个 TCN 因果膨胀卷积残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_total = (kernel_size - 1) * dilation
        self.pad = (self.pad_total, 0)

        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False)
        )

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x_pad = F.pad(x, self.pad, mode='constant', value=0.0)
        out = self.conv(x_pad)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class TCN(nn.Module):
    """多层因果膨胀卷积 TCN，输出动作均值 μ。"""

    def __init__(
        self,
        in_channels: int = 4,
        seq_len: int = 16,
        channels: list = None,
        kernel_size: int = 3,
        dilations: list = None,
        dropout: float = 0.1,
        activation: str = 'ReLU',
        output_dim: int = 64,
        actor_dim: int = 2,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 64]
        if dilations is None:
            dilations = [1, 2, 4]

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.output_dim = output_dim

        blocks = []
        ch_in = in_channels
        for ch_out, dil in zip(channels, dilations):
            blocks.append(
                TCNBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=kernel_size,
                    dilation=dil,
                    dropout=dropout,
                    activation=activation,
                )
            )
            ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Linear(output_dim, actor_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        mu = self.head(x)
        return mu


if __name__ == '__main__':
    batch = 4
    seq_len = 16
    feat_dim = 4
    actor_dim = 14

    model = TCN(
        in_channels=feat_dim,
        seq_len=seq_len,
        channels=[64, 128, 64],
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.1,
    )

    x = torch.randn(batch, seq_len, feat_dim)
    mu = model(x)

    print(f"输入: {x.shape}")
    print(f"μ 输出: {mu.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"TCN 参数量: {total:,}")
