# -*- coding: utf-8 -*-
"""
动力学驱动 DRL 在线优化求解器。

逐时间步优化 TCN-DNN 策略网络：柔性机械臂动力学模型提供物理约束，
跟踪损失惩罚末端位置误差，在线迭代求解关节力矩。

核心流程：
1. 策略网络接收期望轨迹窗口，输出全状态和力矩。
2. 提取广义坐标、广义速度和驱动力矩。
3. 动力学模型计算质量矩阵和广义力，求解广义加速度与动力残差。
4. 正运动学映射到末端位置，计算跟踪误差。
5. 异方差组合损失在线监督：L = r1²/s1² + r2²/s2²。
6. 收敛则输出力矩，否则 Adam 更新后回到步骤 2。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .dynamics.model import FlexibleArmModel
from .dynamics.params import DynamicsParams
from .dynamics.precompute import precompute_all
from .forward_kinematics import ForwardKinematics
from .tcn_dnn_policy import TCN_DNN_Policy


class DRLDynamicSolver:
    """动力学驱动在线优化求解器，用于轨迹跟踪控制。"""

    def __init__(
        self,
        dt: float = 0.01,
        seq_len: int = 16,
        feat_dim: int = 4,
        actor_dim: int = 14,
        tcn_channels: list = None,
        tcn_kernel_size: int = 2,
        tcn_dilations: list = None,
        tcn_dropout: float = 0.0,
        tcn_output_dim: int = 64,
        dnn_hidden_dims: list = None,
        lr_tcn: float = 0.01,
        lr_dnn: float = 0.01,
        epsilon: float = 1e-6,
        max_iters_per_step: int = 500,
        w1: float = 1.0,
        w2: float = 1.0,
        s1: float = 50.0,
        s2: float = 0.1,
        tau_min: float = -50.0,
        tau_max: float = 50.0,
        device: str = "cuda",
    ) -> None:
        self.dt = dt
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.actor_dim = actor_dim
        self.epsilon = epsilon
        self.max_iters_per_step = max_iters_per_step
        self.w1 = w1
        self.w2 = w2
        self.s1 = s1
        self.s2 = s2
        self.tau_min = tau_min
        self.tau_max = tau_max

        if tcn_channels is None:
            tcn_channels = [64, 128, 128, 64]
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4, 8]
        if dnn_hidden_dims is None:
            dnn_hidden_dims = [64]

        # 动力学模型与正运动学。
        self.dyn_params = DynamicsParams()
        self.dyn_params.device = device
        self.dyn_params.__post_init__()
        self.dyn_const = precompute_all(self.dyn_params)
        self.dyn_model = FlexibleArmModel(self.dyn_params, self.dyn_const)
        self.fk = ForwardKinematics(self.dyn_params, self.dyn_const)

        self.dof_full = self.dyn_model.dof
        self.n_A = self.dyn_params.n_A
        self.n_B = self.dyn_params.n_B

        # TCN-DNN 策略网络。
        self.policy = TCN_DNN_Policy(
            seq_len=seq_len,
            feat_dim=feat_dim,
            actor_dim=actor_dim,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilations=tcn_dilations,
            tcn_dropout=tcn_dropout,
            tcn_output_dim=tcn_output_dim,
            dnn_hidden_dims=dnn_hidden_dims,
            log_sigma_min=-5.0,
            log_sigma_max=2.0,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        self.policy.to(
            device=self.dyn_params.torch_device,
            dtype=self.dyn_params.torch_dtype,
        )
        self.policy.train()

        self._init_state_dict = {
            k: v.clone().detach()
            for k, v in self.policy.state_dict().items()
        }

        self.lr_tcn = lr_tcn
        self.lr_dnn = lr_dnn

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.tcn.parameters(), "lr": lr_tcn},
            {"params": self.policy.dnn.parameters(), "lr": lr_dnn},
        ])

        self.total_opt_steps = 0
        self.converged_steps = 0

    def reinit_policy(self) -> None:
        """每时间步将策略网络恢复到初始随机权重并重置优化器。"""
        self.policy.load_state_dict(self._init_state_dict)
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.tcn.parameters(), "lr": self.lr_tcn},
            {"params": self.policy.dnn.parameters(), "lr": self.lr_dnn},
        ])

    def _expand_action_to_state(self, mu: torch.Tensor) -> tuple:
        """
        将 14 维策略输出展开为完整动力学状态。

        14 维布局:
            [θ1, θ2, q1, q2, A1, B1,  θ̇1, θ̇2, q̇1, q̇2, Ȧ1, Ḃ1,  τ1, τ2]

        Returns
        -------
        xi  : [dof_full]  广义坐标。
        dxi : [dof_full]  广义速度。
        tau : [2]         驱动力矩。
        """
        mu = mu.reshape(-1)

        theta1 = mu[0]
        theta2 = mu[1]
        q1 = mu[2]
        q2 = mu[3]
        A1 = mu[4:5]
        B1 = mu[5:6]

        dtheta1 = mu[6]
        dtheta2 = mu[7]
        dq1 = mu[8]
        dq2 = mu[9]
        dA1 = mu[10:11]
        dB1 = mu[11:12]

        tau1 = mu[12]
        tau2 = mu[13]
        tau = torch.stack([tau1, tau2])

        xi = torch.cat([
            torch.stack([theta1, theta2, q1, q2]),
            A1,  B1,
        ])

        dxi = torch.cat([
            torch.stack([dtheta1, dtheta2, dq1, dq2]),
            dA1, dB1,
        ])

        return xi, dxi, tau

    def compute_loss(
        self, mu: torch.Tensor, x_des: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        mu     : [14]  策略网络输出均值。
        x_des  : [2]   期望末端位置。

        Returns
        -------
        loss : 标量损失。
        info : 各分项详情字典。
        """
        xi, dxi, tau = self._expand_action_to_state(mu)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        M = self.dyn_model.mass_matrix(xi)
        Q = self.dyn_model.generalized_force(xi, dxi, tau)
        ddxi = torch.linalg.solve(M, Q.unsqueeze(-1)).squeeze(-1)
        E_dyn = M @ ddxi - Q
        r1 = torch.sum(E_dyn ** 2)

        theta1, theta2 = xi[0], xi[1]
        A = xi[4:4 + self.n_A]
        B = xi[4 + self.n_A:4 + self.n_A + self.n_B]
        x_act, y_act = self.fk.forward(theta1, theta2, A, B)

        x_d, y_d = x_des[0], x_des[1]
        r2 = (x_act - x_d) ** 2 + (y_act - y_d) ** 2

        loss = self.w1 * r1 / (self.s1 ** 2) + self.w2 * r2 / (self.s2 ** 2)

        info = {
            "r1_dyn": r1.detach(),
            "r2_track": r2.detach(),
            "tau": tau.detach(),
            "x_act": torch.stack([x_act, y_act]).detach(),
            "x_des": x_des.detach(),
        }

        return loss, info

    def optimize_step(
        self, x_window: torch.Tensor, x_des: torch.Tensor, verbose: bool = False,
    ) -> dict:
        """
        对单个时间步执行在线优化，返回收敛后的力矩。

        Parameters
        ----------
        x_window : [seq_len, feat_dim]  期望轨迹的局部窗口。
        x_des    : [2]                  当前时间步的期望末端位置。

        Returns
        -------
        result : dict
            tau, success, iters, loss, mu, x_act.
        """
        for it in range(self.max_iters_per_step):
            self.optimizer.zero_grad()

            x_batch = x_window.unsqueeze(0)
            mu, log_sig = self.policy.forward(x_batch)
            mu = mu.squeeze(0)

            loss, info = self.compute_loss(mu, x_des)

            if loss.item() < self.epsilon:
                self.converged_steps += 1
                self.total_opt_steps += (it + 1)
                return {
                    "tau": info["tau"].clone(),
                    "success": True,
                    "iters": it + 1,
                    "loss": loss.item(),
                    "mu": mu.detach().clone(),
                    "x_act": info["x_act"].clone(),
                }

            loss.backward()
            self.optimizer.step()

            if verbose and (it % 100 == 0):
                print(f"    iter {it:4d}: loss={loss.item():.2e}  "
                      f"r1={info['r1_dyn'].item():.2e}  r2={info['r2_track'].item():.2e}")

        self.total_opt_steps += self.max_iters_per_step
        with torch.no_grad():
            x_batch = x_window.unsqueeze(0)
            mu, _ = self.policy.forward(x_batch)
            mu = mu.squeeze(0)
            _, info = self.compute_loss(mu, x_des)

        return {
            "tau": info["tau"].clone(),
            "success": False,
            "iters": self.max_iters_per_step,
            "loss": info["r1_dyn"].item() + info["r2_track"].item(),
            "mu": mu.detach().clone(),
            "x_act": info["x_act"].clone(),
        }

    def solve_trajectory(
        self, trajectory: np.ndarray, verbose: bool = True,
    ) -> dict:
        """
        对整个期望轨迹逐时间步求解控制力矩。

        Parameters
        ----------
        trajectory : [N, 4]  期望末端轨迹 (x, y, vx, vy)。

        Returns
        -------
        results : dict
            tau_history, x_act_history, loss_history, iters_history, success_rate。
        """
        N = trajectory.shape[0]
        half = self.seq_len // 2

        tau_history = np.zeros((N, 2))
        x_act_history = np.zeros((N, 2))
        loss_history = np.zeros(N)
        iters_history = np.zeros(N, dtype=int)
        success_count = 0

        dev = self.dyn_params.torch_device
        dt = self.dyn_params.torch_dtype

        for i in range(N):
            self.reinit_policy()

            window = np.zeros((self.seq_len, 4))
            for k in range(self.seq_len):
                src_idx = i + k - half
                src_idx = max(0, min(N - 1, src_idx))
                window[k] = trajectory[src_idx]

            x_window = torch.tensor(window, device=dev, dtype=dt)
            x_des = torch.tensor(trajectory[i, :2], device=dev, dtype=dt)

            result = self.optimize_step(x_window, x_des, verbose=False)

            tau_history[i] = result["tau"].cpu().numpy()
            x_act_history[i] = result["x_act"].cpu().numpy()
            loss_history[i] = result["loss"]
            iters_history[i] = result["iters"]

            if result["success"]:
                success_count += 1

            if verbose and (i % 50 == 0 or i == N - 1):
                status = "OK" if result["success"] else "FAIL"
                print(f"  step {i:4d}/{N}: loss={result['loss']:.2e}  "
                      f"iters={result['iters']:4d}  "
                      f"tau=[{tau_history[i,0]:+.2f}, {tau_history[i,1]:+.2f}]  "
                      f"{status}")

        success_rate = success_count / N

        if verbose:
            avg_iters = np.mean(iters_history)
            print(f"\n=== 全轨迹求解完成 ===")
            print(f"  收敛率: {success_rate*100:.1f}% ({success_count}/{N})")
            print(f"  平均迭代次数: {avg_iters:.1f}")
            print(f"  总优化步数: {self.total_opt_steps}")

        return {
            "tau_history": tau_history,
            "x_act_history": x_act_history,
            "loss_history": loss_history,
            "iters_history": iters_history,
            "success_rate": success_rate,
        }


if __name__ == "__main__":
    print("=== DRL Dynamic Solver 测试 ===\n")

    solver = DRLDynamicSolver(
        dt=0.01, seq_len=16, feat_dim=4, actor_dim=14,
        epsilon=1e-6, max_iters_per_step=200, device="cuda",
    )

    print(f"动力学 DOF: {solver.dof_full}")
    print(f"策略网络参数量: {solver.policy.num_params}")
    print(f"设备: {solver.dyn_params.torch_device}")

    N_test = 10
    traj_test = np.zeros((N_test, 4))
    for i in range(N_test):
        t = i / (N_test - 1)
        traj_test[i, 0] = 0.5 + 0.5 * t
        traj_test[i, 1] = 0.0 + 0.5 * t

    print(f"\n测试轨迹: {N_test} 步, "
          f"从 ({traj_test[0,0]:.3f}, {traj_test[0,1]:.3f}) → ({traj_test[-1,0]:.3f}, {traj_test[-1,1]:.3f})")
    print("开始求解...\n")

    results = solver.solve_trajectory(traj_test, verbose=True)

    print(f"\n力矩序列:")
    for i in range(min(N_test, 10)):
        print(f"  t{i:2d}: τ=[{results['tau_history'][i,0]:+8.4f}, {results['tau_history'][i,1]:+8.4f}]  "
              f"x_act=[{results['x_act_history'][i,0]:.4f}, {results['x_act_history'][i,1]:.4f}]")
