"""
dynamics/model.py

柔性机械臂动力学模型统一入口。

作用
----
1. 统一管理参数与预计算结果
2. 提供质量阵 M(xi) 计算接口
3. 提供广义力 Q(xi, dxi, tau) 计算接口
4. 提供广义加速度 ddxi 计算接口
5. 提供一阶状态方程右端 state_derivative 接口

当前广义坐标顺序约定为：
    xi  = [theta1, theta2, q1, q2, A..., B...]
    dxi = [dtheta1, dtheta2, dq1, dq2, dA..., dB...]

总状态向量约定为：
    x = [xi, dxi]
"""


from __future__ import annotations

import torch

from .params import DynamicsParams
from .precompute import PrecomputedTerms, precompute_all
from .mass_matrix import compute_M, get_dof as get_generalized_dof
from .generalized_force import compute_Q


class FlexibleArmModel:
    """柔性机械臂动力学模型类封装。"""

    def __init__(
        self,
        params: DynamicsParams | None = None,
        const: PrecomputedTerms | None = None,
    ) -> None:
        """
        初始化动力学模型。

        Parameters
        ----------
        params : DynamicsParams | None
            动力学参数对象；若为 None，则使用默认参数初始化。
        const : PrecomputedTerms | None
            预计算结果；若为 None，则根据 params 自动执行一次预计算。
        """
        self.params = params if params is not None else DynamicsParams()
        self.const = const if const is not None else precompute_all(self.params)

# 维度与状态拆分
    @property
    def dof(self) -> int:
        """系统广义自由度数。"""
        return get_generalized_dof(self.params)

    @property
    def state_dim(self) -> int:
        """系统一阶状态总维数。"""
        return 2 * self.dof

    def split_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将总状态向量拆分为 xi 与 dxi。

        Parameters
        ----------
        x : np.ndarray
            总状态向量，形状应为 (2*dof,)。

        Returns
        -------
        xi : np.ndarray
            广义坐标向量，形状为 (dof,)。
        dxi : np.ndarray
            广义速度向量，形状为 (dof,)。
        """
        x = torch.as_tensor(
            x,
            device=self.params.torch_device,
            dtype=self.params.torch_dtype,
        ).reshape(-1)

        if x.numel() != self.state_dim:
            raise ValueError(f"x 的长度应为 {self.state_dim}，当前为 {x.numel()}。")

        xi = x[:self.dof]
        dxi = x[self.dof:]
        return xi, dxi


    def build_state(self, xi: torch.Tensor, dxi: torch.Tensor) -> torch.Tensor:
        """
        将 xi 与 dxi 拼接为总状态向量 x。

        Parameters
        ----------
        xi : np.ndarray
            广义坐标向量。
        dxi : np.ndarray
            广义速度向量。

        Returns
        -------
        x : np.ndarray
            总状态向量，形状为 (2*dof,)。
        """
        xi = torch.as_tensor(
            xi,
            device=self.params.torch_device,
            dtype=self.params.torch_dtype,
        ).reshape(-1)
        dxi = torch.as_tensor(
            dxi,
            device=self.params.torch_device,
            dtype=self.params.torch_dtype,
        ).reshape(-1)

        if xi.numel() != self.dof:
            raise ValueError(f"xi 的长度应为 {self.dof}，当前为 {xi.numel()}。")
        if dxi.numel() != self.dof:
            raise ValueError(f"dxi 的长度应为 {self.dof}，当前为 {dxi.numel()}。")

        return torch.cat([xi, dxi], dim=0)


# 核心动力学接口
    def mass_matrix(self, xi: torch.Tensor) -> torch.Tensor:
        """计算质量阵 M(xi)。"""
        return compute_M(xi, self.params, self.const)

    def generalized_force(
        self,
        xi: torch.Tensor,
        dxi: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算广义力向量 Q(xi, dxi, tau)。"""
        if tau is None:
            tau = torch.zeros(2, device=self.params.torch_device, dtype=self.params.torch_dtype)

        return compute_Q(xi, dxi, tau, self.params, self.const)


    def acceleration(
        self,
        xi: torch.Tensor,
        dxi: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        计算广义加速度 ddxi。

        数学形式
        ----------
        当前模型采用：
            M(xi) * ddxi = Q(xi, dxi, tau)

        其中 Q 已包含输入力矩 tau 对应的广义力项。
        """

        if tau is None:
            tau = torch.zeros(2, device=self.params.torch_device, dtype=self.params.torch_dtype)

        M = self.mass_matrix(xi)
        Q = self.generalized_force(xi, dxi, tau)

        ddxi = torch.linalg.solve(M, Q.unsqueeze(-1)).squeeze(-1)
        return ddxi


    def state_derivative(
        self,
        t: float,
        x: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算一阶状态方程右端 dx/dt"""

        _ = t  # 当前模型未显式使用时间变量，保留接口统一性

        xi, dxi = self.split_state(x)
        ddxi = self.acceleration(xi, dxi, tau)

        dx = torch.cat([dxi, ddxi], dim=0)
        return dx

# 刷新预计算
    def refresh_precompute(self) -> None:
        """
        根据当前参数重新执行预计算。

        适用于修改 params 后刷新 const。
        """
        self.const = precompute_all(self.params)


# 类接口包装
    def residual(
        self,
        xi: torch.Tensor,
        dxi: torch.Tensor,
        ddxi: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        计算Loss动力学残差向量 r_dyn。

        数学形式
        ----------
        当前模型满足：
            M(xi) * ddxi = Q(xi, dxi, tau)

        其中 Q 已包含输入力矩 tau 对应的广义力项，因此残差定义为：
            r_dyn = M(xi) * ddxi - Q(xi, dxi, tau)

        Parameters
        ----------
        xi : np.ndarray
            广义坐标向量，形状为 (dof,)。
        dxi : np.ndarray
            广义速度向量，形状为 (dof,)。
        ddxi : np.ndarray
            广义加速度向量，形状为 (dof,)。
        tau : np.ndarray | None
            输入力矩向量，形状为 (2,)；若为 None，则默认零输入。

        Returns
        -------
        result : dict
            返回字典，包含：
            - "M"     : 质量阵
            - "Q"     : 广义力向量（已包含 tau）
            - "r_dyn" : 动力学残差向量
        """
        xi = torch.as_tensor(xi, device=self.params.torch_device, dtype=self.params.torch_dtype).reshape(-1)
        dxi = torch.as_tensor(dxi, device=self.params.torch_device, dtype=self.params.torch_dtype).reshape(-1)
        ddxi = torch.as_tensor(ddxi, device=self.params.torch_device, dtype=self.params.torch_dtype).reshape(-1)

        if xi.numel() != self.dof:
            raise ValueError(f"xi 的长度应为 {self.dof}，当前为 {xi.numel()}。")
        if dxi.numel() != self.dof:
            raise ValueError(f"dxi 的长度应为 {self.dof}，当前为 {dxi.numel()}。")
        if ddxi.numel() != self.dof:
            raise ValueError(f"ddxi 的长度应为 {self.dof}，当前为 {ddxi.numel()}。")

        if tau is None:
            tau = torch.zeros(2, device=self.params.torch_device, dtype=self.params.torch_dtype)
        else:
            tau = torch.as_tensor(tau, device=self.params.torch_device, dtype=self.params.torch_dtype).reshape(-1)

        if tau.numel() != 2:
            raise ValueError(f"tau 的长度应为 2，当前为 {tau.numel()}。")

        M = self.mass_matrix(xi)
        Q = self.generalized_force(xi, dxi, tau)
        r_dyn = M @ ddxi - Q

        return {
            "M": M,
            "Q": Q,
            "r_dyn": r_dyn,
        }


    def make_rhs(self, tau=None):
        """
        适合 torch 张量状态推进；若用于 solve_ivp，需额外做 numpy 适配。

        Parameters
        ----------
        tau : None, np.ndarray, or callable
            输入力矩：
            1. 若为 None，则默认零输入；
            2. 若为长度为 2 的数组，则视为常值输入；
            3. 若为可调用对象，则支持两种形式：
               - tau(t)
               - tau(t, x)

        Returns
        -------
        rhs : callable
            可直接传给 solve_ivp 的函数，形式为 rhs(t, x)。
        """

        if tau is None:
            def rhs(t, x):
                tau_zero = torch.zeros(2, device=self.params.torch_device, dtype=self.params.torch_dtype)
                return self.state_derivative(t, x, tau_zero)
            return rhs

        if callable(tau):
            def rhs(t, x):
                try:
                    tau_val = tau(t, x)
                except TypeError:
                    tau_val = tau(t)
                tau_val = torch.as_tensor(
                    tau_val,
                    device=self.params.torch_device,
                    dtype=self.params.torch_dtype,
                ).reshape(-1)
                return self.state_derivative(t, x, tau_val)
            return rhs

        tau_const = torch.as_tensor(
            tau,
            device=self.params.torch_device,
            dtype=self.params.torch_dtype,
        ).reshape(-1)

        if tau_const.numel() != 2:
            raise ValueError(f"常值 tau 的长度应为 2，当前为 {tau_const.numel()}。")

        def rhs(t, x):
            return self.state_derivative(t, x, tau_const)

        return rhs


if __name__ == "__main__":
    model = FlexibleArmModel()

    xi = torch.zeros(model.dof, device=model.params.torch_device, dtype=model.params.torch_dtype)
    dxi = torch.zeros(model.dof, device=model.params.torch_device, dtype=model.params.torch_dtype)
    tau = torch.zeros(2, device=model.params.torch_device, dtype=model.params.torch_dtype)

    x = model.build_state(xi, dxi)
    dx = model.state_derivative(0.0, x, tau)

    print("=== model test ===")
    print(f"dof       : {model.dof}")
    print(f"state_dim : {model.state_dim}")
    print(f"x shape   : {tuple(x.shape)}")
    print(f"dx shape  : {tuple(dx.shape)}")
    print(f"dx finite : {torch.all(torch.isfinite(dx)).item()}")
    print("dx =\n", dx.detach().cpu())