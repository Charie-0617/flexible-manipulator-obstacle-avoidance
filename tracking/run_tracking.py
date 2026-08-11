# -*- coding: utf-8 -*-
"""
动力学驱动 DRL 轨迹跟踪 Demo。

加载期望末端轨迹，通过 DRL 求解器逐时间步在线优化，
输出关节驱动力矩和实际末端轨迹，可选可视化跟踪结果。

用法:
    python -m tracking.run_tracking
    python -m tracking.run_tracking --device cuda --max-steps 50
    python -m tracking.run_tracking --csv path/to/trajectory.csv

默认自动生成一条简单 demo 轨迹。通过 --csv 可指定外部轨迹文件。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from .drl_solver import DRLDynamicSolver


def generate_demo_trajectory(steps: int = 50, dt: float = 0.01) -> np.ndarray:
    """
    自动生成一条简单 demo 轨迹，沿直线从 (0.5, 0) 到 (4.0, 3.0) 匀速运动。

    输出: [steps, 4]，各列为 (x, y, vx, vy)。
    """
    start = np.array([0.5, 0.0])
    end = np.array([4.0, 3.0])
    traj = np.zeros((steps, 4))
    for i in range(steps):
        t = i / max(steps - 1, 1)
        traj[i, 0] = start[0] + (end[0] - start[0]) * t
        traj[i, 1] = start[1] + (end[1] - start[1]) * t
    traj[:, 2] = (traj[-1, 0] - traj[0, 0]) / (steps * dt)  # vx
    traj[:, 3] = (traj[-1, 1] - traj[0, 1]) / (steps * dt)  # vy
    return traj


def load_trajectory(csv_path: str, dt: float = 0.01) -> np.ndarray:
    """
    从 CSV 加载期望末端轨迹并重采样到目标频率。

    CSV 格式: time_s, x, y, vx, vy，原始间隔约 50ms。
    输出: [N, 4]，按 dt 重采样后的 (x, y, vx, vy)。
    """
    import csv

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            rows.append([float(v) for v in line])

    data = np.array(rows)
    t_raw = data[:, 0]
    x_raw = data[:, 1]
    y_raw = data[:, 2]
    vx_raw = data[:, 3]
    vy_raw = data[:, 4]

    t_max = t_raw[-1]
    t_new = np.arange(0, t_max, dt)
    N_new = len(t_new)

    x_new = np.interp(t_new, t_raw, x_raw)
    y_new = np.interp(t_new, t_raw, y_raw)
    vx_new = np.interp(t_new, t_raw, vx_raw)
    vy_new = np.interp(t_new, t_raw, vy_raw)

    traj = np.stack([x_new, y_new, vx_new, vy_new], axis=1)
    return traj


def main():
    parser = argparse.ArgumentParser(description="运行 DRL 轨迹跟踪 Demo。")
    parser.add_argument("--csv", type=str, default=None,
                        help="期望末端轨迹 CSV 文件路径（不指定则自动生成 demo 轨迹）")
    parser.add_argument("--dt", type=float, default=0.01, help="控制周期 (s)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-steps", type=int, default=0,
                        help="最大时间步数 (0 表示使用全部轨迹)")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="收敛阈值")
    parser.add_argument("--lr", type=float, default=0.01, help="优化学习率")
    parser.add_argument("--max-iters", type=int, default=500, help="每时间步最大迭代次数")
    parser.add_argument("--seq-len", type=int, default=16, help="TCN 输入窗口长度")
    parser.add_argument("--output", type=str, default="tracking_results.npz", help="结果输出文件")
    parser.add_argument("--no-plot", action="store_true", help="不显示可视化图表")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    csv_path = args.csv
    if csv_path is not None:
        if not os.path.exists(csv_path):
            print(f"错误: 找不到轨迹文件 {csv_path}")
            sys.exit(1)
        print(f"加载轨迹: {csv_path}")
        traj = load_trajectory(csv_path, dt=args.dt)
    else:
        steps = args.max_steps if args.max_steps > 0 else 50
        traj = generate_demo_trajectory(steps=steps, dt=args.dt)
        print(f"自动生成 demo 轨迹: {steps} 步")
    N_total = len(traj)
    print(f"  总步数: {N_total} (dt={args.dt}s, 总时长={N_total * args.dt:.1f}s)")

    if args.max_steps > 0 and args.max_steps < N_total:
        traj = traj[:args.max_steps]
        print(f"  截取前 {args.max_steps} 步")

    N = len(traj)
    print(f"  求解步数: {N}")
    print(f"  轨迹范围: x=[{traj[:,0].min():.3f}, {traj[:,0].max():.3f}], "
          f"y=[{traj[:,1].min():.3f}, {traj[:,1].max():.3f}]")

    print(f"\n初始化 DRL 跟踪求解器:")
    print(f"  设备: {args.device}")
    print(f"  TCN: kernel_size=2, dropout=0, levels=4, lr={args.lr}")
    print(f"  DNN: 32→64→14, lr={args.lr}")
    print(f"  ε = {args.epsilon}")

    solver = DRLDynamicSolver(
        dt=args.dt,
        seq_len=args.seq_len,
        feat_dim=4,
        actor_dim=14,
        tcn_kernel_size=2,
        tcn_dropout=0.0,
        dnn_hidden_dims=[64],
        lr_tcn=args.lr,
        lr_dnn=args.lr,
        epsilon=args.epsilon,
        max_iters_per_step=args.max_iters,
        device=args.device,
    )

    print(f"\n开始逐时间步在线优化...")
    t_start = time.time()
    results = solver.solve_trajectory(traj, verbose=True)
    t_elapsed = time.time() - t_start

    tau_history = results["tau_history"]
    x_act_history = results["x_act_history"]
    loss_history = results["loss_history"]
    iters_history = results["iters_history"]

    np.savez(args.output,
             tau_history=tau_history,
             x_act_history=x_act_history,
             loss_history=loss_history,
             iters_history=iters_history,
             x_des=traj[:, :2],
             dt=args.dt,
             success_rate=results["success_rate"],
    )
    print(f"\n结果已保存至: {args.output}")

    print(f"\n=== 求解统计 ===")
    print(f"  总耗时: {t_elapsed:.1f} s")
    print(f"  每步平均: {t_elapsed / N * 1000:.2f} ms")
    print(f"  收敛率: {results['success_rate'] * 100:.1f}%")
    print(f"  平均迭代: {np.mean(iters_history):.1f} 次/步")
    print(f"  最终损失: {loss_history[-1]:.2e}")

    dx = x_act_history[:, 0] - traj[:, 0]
    dy = x_act_history[:, 1] - traj[:, 1]
    rmse_x = np.sqrt(np.mean(dx ** 2))
    rmse_y = np.sqrt(np.mean(dy ** 2))
    print(f"  RMSE X: {rmse_x:.4f} m")
    print(f"  RMSE Y: {rmse_y:.4f} m")

    print(f"  tau1: min={tau_history[:,0].min():+.2f}, max={tau_history[:,0].max():+.2f}")
    print(f"  tau2: min={tau_history[:,1].min():+.2f}, max={tau_history[:,1].max():+.2f}")

    if not args.no_plot:
        plot_results(traj, x_act_history, tau_history, loss_history, iters_history, args.dt)


def plot_results(traj_des, x_act, tau, loss, iters, dt):
    """绘制四幅对比图：XY 轨迹、位置误差、力矩、损失与迭代次数。"""
    N = len(traj_des)
    t = np.arange(N) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dynamics-Driven DRL Control Results", fontsize=14)

    # XY 轨迹。
    ax = axes[0, 0]
    ax.plot(traj_des[:, 0], traj_des[:, 1], "b-", linewidth=1.5, label="Desired")
    ax.plot(x_act[:, 0], x_act[:, 1], "r--", linewidth=1.0, label="Actual (DRL)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("End-Effector Trajectory")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # 位置误差。
    ax = axes[0, 1]
    dx = x_act[:, 0] - traj_des[:, 0]
    dy = x_act[:, 1] - traj_des[:, 1]
    ax.plot(t, dx * 1000, "r-", linewidth=0.8, label="ΔX")
    ax.plot(t, dy * 1000, "b-", linewidth=0.8, label="ΔY")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (mm)")
    ax.set_title("Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 驱动力矩。
    ax = axes[1, 0]
    ax.plot(t, tau[:, 0], "r-", linewidth=0.8, label="τ₁ (Joint 1)")
    ax.plot(t, tau[:, 1], "b-", linewidth=0.8, label="τ₂ (Joint 2)")
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=-50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Driving Torques")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 损失与迭代次数。
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(t, loss, "b-", linewidth=0.8, label="Loss")
    ax2.plot(t, iters, "r-", linewidth=0.5, alpha=0.5, label="Iterations")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loss", color="b")
    ax2.set_ylabel("Iterations", color="r")
    ax.set_title("Loss & Iteration Count")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
