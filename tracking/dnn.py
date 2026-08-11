# -*- coding: utf-8 -*-
"""
DNN 分支，用于输出对数标准差 log_σ。

将时序特征展平后通过全连接网络，不依赖时序结构。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DNN(nn.Module):
    """MLP 分支，从轨迹序列映射到对数标准差 log_σ。"""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list = None,
        output_dim: int = 2,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'GELU':
                layers.append(nn.GELU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        log_sigma = self.net(x)
        return log_sigma


if __name__ == '__main__':
    batch = 4
    seq_len = 16
    feat_dim = 4
    output_dim = 2

    model = DNN(
        input_dim=seq_len * feat_dim,
        hidden_dims=[128, 64],
        output_dim=output_dim,
    )

    x = torch.randn(batch, seq_len, feat_dim)
    log_sigma = model(x)

    print(f"输入: {x.shape}")
    print(f"log_σ 输出: {log_sigma.shape}")

    sigma = torch.exp(log_sigma)
    print(f"σ: min={sigma.min():.4f}, max={sigma.max():.4f}")

    total = sum(p.numel() for p in model.parameters())
    print(f"DNN 参数量: {total:,}")
