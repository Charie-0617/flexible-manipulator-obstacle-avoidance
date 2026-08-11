# -*- coding: utf-8 -*-
"""
柔性双连杆机械臂正运动学。

将广义坐标 (θ1, θ2, q1, q2, A, B) 映射到末端执行器位姿和速度，
考虑柔性杆尖端变形修正。
"""

from __future__ import annotations

import torch

from .dynamics.params import DynamicsParams
from .dynamics.precompute import PrecomputedTerms, precompute_all


class ForwardKinematics:
    """柔性双连杆机械臂正运动学。"""

    def __init__(
        self,
        params: DynamicsParams | None = None,
        const: PrecomputedTerms | None = None,
    ) -> None:
        self.params = params if params is not None else DynamicsParams()
        self.const = const if const is not None else precompute_all(self.params)

        self._Phi_A_tip = self.const.Phi_A[:, -1]
        self._Phi_B_tip = self.const.Phi_B[:, -1]
        self._L1 = self.params.L1
        self._L2 = self.params.L2

    def tip_deformation(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算柔性杆尖端的轴向变形 u_x 和横向变形 u_y。"""
        u_x = A @ self._Phi_A_tip
        u_y = B @ self._Phi_B_tip
        return u_x, u_y

    def forward(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据广义坐标计算末端执行器 (x, y) 位置。"""
        u_x, u_y = self.tip_deformation(A, B)

        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)
        c12 = torch.cos(theta1 + theta2)
        s12 = torch.sin(theta1 + theta2)

        x = self._L1 * c1 + (self._L2 + u_x) * c12 - u_y * s12
        y = self._L1 * s1 + (self._L2 + u_x) * s12 + u_y * c12

        return x, y

    def forward_with_velocity(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        dtheta1: torch.Tensor,
        dtheta2: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据广义坐标和速度计算末端执行器位姿和速度。"""
        u_x, u_y = self.tip_deformation(A, B)
        du_x, du_y = self.tip_deformation(dA, dB)

        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)
        c12 = torch.cos(theta1 + theta2)
        s12 = torch.sin(theta1 + theta2)

        L2_eff = self._L2 + u_x
        x = self._L1 * c1 + L2_eff * c12 - u_y * s12
        y = self._L1 * s1 + L2_eff * s12 + u_y * c12

        vx = (
            (-self._L1 * s1 - L2_eff * s12 - u_y * c12) * dtheta1
            + (-L2_eff * s12 - u_y * c12) * dtheta2
            + (self._Phi_A_tip * c12) @ dA
            + (-self._Phi_B_tip * s12) @ dB
        )

        vy = (
            (self._L1 * c1 + L2_eff * c12 - u_y * s12) * dtheta1
            + (L2_eff * c12 - u_y * s12) * dtheta2
            + (self._Phi_A_tip * s12) @ dA
            + (self._Phi_B_tip * c12) @ dB
        )

        return x, y, vx, vy


if __name__ == "__main__":
    params = DynamicsParams()
    fk = ForwardKinematics(params)

    dev = params.torch_device
    dt = params.torch_dtype

    theta1 = torch.tensor(0.0, device=dev, dtype=dt)
    theta2 = torch.tensor(0.0, device=dev, dtype=dt)
    A = torch.zeros(params.n_A, device=dev, dtype=dt)
    B = torch.zeros(params.n_B, device=dev, dtype=dt)

    x, y = fk.forward(theta1, theta2, A, B)
    print(f"零状态末端位置: x={x.item():.6f}, y={y.item():.6f}")
    print(f"期望: x={params.L1+params.L2:.6f}, y=0.0")

    import math
    theta1 = torch.tensor(math.pi / 4, device=dev, dtype=dt)
    theta2 = torch.tensor(math.pi / 4, device=dev, dtype=dt)

    x2, y2 = fk.forward(theta1, theta2, A, B)
    r = torch.sqrt(x2**2 + y2**2)
    print(f"\nθ1=π/4, θ2=π/4: x={x2.item():.6f}, y={y2.item():.6f}, r={r.item():.6f}")

    dtheta1 = torch.tensor(0.1, device=dev, dtype=dt)
    dtheta2 = torch.tensor(0.2, device=dev, dtype=dt)
    dA = torch.zeros(params.n_A, device=dev, dtype=dt)
    dB = torch.zeros(params.n_B, device=dev, dtype=dt)

    x3, y3, vx, vy = fk.forward_with_velocity(
        theta1, theta2, A, B, dtheta1, dtheta2, dA, dB
    )
    print(f"\n位置: x={x3.item():.6f}, y={y3.item():.6f}")
    print(f"速度: vx={vx.item():.6f}, vy={vy.item():.6f}")
