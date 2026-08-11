# -*- coding: utf-8 -*-
"""
TCN-DNN 双分支策略网络。

TCN 分支输出动作均值 μ，DNN 分支输出对数标准差 log_σ。
部署时直接取 μ 作为确定性动作。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import TCN
from .dnn import DNN


class TCN_DNN_Policy(nn.Module):
    """TCN-DNN 双分支策略网络。"""

    def __init__(
        self,
        seq_len: int = 16,
        feat_dim: int = 4,
        actor_dim: int = 2,

        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        tcn_dilations: list = None,
        tcn_dropout: float = 0.1,
        tcn_activation: str = 'ReLU',
        tcn_output_dim: int = 64,

        dnn_hidden_dims: list = None,
        dnn_activation: str = 'ReLU',

        log_sigma_min: float = -5.0,
        log_sigma_max: float = 2.0,

        tau_min: float = -50.0,
        tau_max: float = 50.0,
    ) -> None:
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [64, 128, 64]
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4]
        if dnn_hidden_dims is None:
            dnn_hidden_dims = [128, 64]

        self.actor_dim = actor_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.tcn = TCN(
            in_channels=feat_dim,
            seq_len=seq_len,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=tcn_dropout,
            activation=tcn_activation,
            output_dim=tcn_output_dim,
            actor_dim=actor_dim,
        )

        self.dnn = DNN(
            input_dim=seq_len * feat_dim,
            hidden_dims=dnn_hidden_dims,
            output_dim=actor_dim,
            activation=dnn_activation,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, seq_len, feat_dim]

        Returns
        -------
        mu      : [B, actor_dim]  动作均值。
        log_sig : [B, actor_dim]  对数标准差（已裁剪）。
        """
        mu = self.tcn(x)
        log_sig = self.dnn(x)
        log_sig = torch.clamp(log_sig, self.log_sigma_min, self.log_sigma_max)
        return mu, log_sig

    def get_action(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        获取部署动作。

        deterministic=True 时直接使用 μ，否则从高斯分布采样。
        """
        mu, log_sig = self.forward(x)

        if deterministic:
            a = mu
        else:
            sigma = torch.exp(log_sig)
            eps = torch.randn_like(mu)
            a = mu + eps * sigma

        a[:, -2] = torch.clamp(a[:, -2], self.tau_min, self.tau_max)
        a[:, -1] = torch.clamp(a[:, -1], self.tau_min, self.tau_max)

        return a

    @property
    def num_params(self) -> dict[str, int]:
        return {
            'tcn': sum(p.numel() for p in self.tcn.parameters()),
            'dnn': sum(p.numel() for p in self.dnn.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }


if __name__ == '__main__':
    batch = 4
    seq_len = 32
    feat_dim = 4
    actor_dim = 2

    policy = TCN_DNN_Policy(seq_len=seq_len, feat_dim=feat_dim, actor_dim=actor_dim)
    x = torch.randn(batch, seq_len, feat_dim)

    mu, log_sig = policy.forward(x)
    print(f"输入 X_t  : {x.shape}")
    print(f"μ 输出    : {mu.shape}")
    print(f"log_σ 输出 : {log_sig.shape}")

    a_det = policy.get_action(x, deterministic=True)
    print(f"\n部署动作 (确定性): {a_det.shape}")
    print(f"  tau1 range: [{a_det[:, 4].min():.2f}, {a_det[:, 4].max():.2f}]")
    print(f"  tau2 range: [{a_det[:, 5].min():.2f}, {a_det[:, 5].max():.2f}]")

    a_exp = policy.get_action(x, deterministic=False)
    print(f"探索动作 (采样): {a_exp.shape}")

    print(f"\n参数量: {policy.num_params}")
