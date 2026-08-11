import torch
import numpy as np
import joblib
from collections import deque
from pathlib import Path
from .sta import SpatioTemporalAttentionGRU


class LocalObstacleAvoidance:
    """
    局部避障推理模块。

    使用 STA-GRU 模型从历史状态序列预测未来轨迹和碰撞风险，
    每次滚动执行若干步，固定取最近 3 个障碍物。
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        history_steps: int = 20,
        future_steps: int = 10,
        execute_steps: int = 3,
        num_obstacles: int = 3,
        device: str = "cpu",
        smooth_lambda: float = 0.7
    ):
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.execute_steps = execute_steps
        self.num_obstacles = num_obstacles
        self.device = device
        self.smooth_lambda = smooth_lambda

        self.prev_proj_pos = None

        self.model = SpatioTemporalAttentionGRU(
            future_steps=future_steps
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self.model.to(device)
        self.model.eval()

        scalers = joblib.load(scaler_path)
        self.rp_scaler = scalers["robot_pos"]
        self.rv_scaler = scalers["robot_vel"]
        self.op_scaler = scalers["obs_pos"]
        self.ov_scaler = scalers["obs_vel"]
        self.gr_scaler = scalers["goal_rel"]

        self.robot_pos_buf = deque(maxlen=history_steps)
        self.robot_vel_buf = deque(maxlen=history_steps)
        self.obs_pos_buf = deque(maxlen=history_steps)
        self.obs_vel_buf = deque(maxlen=history_steps)

    def update_history(self, robot_state, obs_state):
        """每帧追加机器人状态和障碍物观测到历史缓冲。"""
        self.robot_pos_buf.append(
            np.asarray(robot_state["position"], dtype=np.float32)
        )
        self.robot_vel_buf.append(
            np.asarray(robot_state["velocity"], dtype=np.float32)
        )
        self.obs_pos_buf.append(
            np.asarray(obs_state["position"], dtype=np.float32)
        )
        self.obs_vel_buf.append(
            np.asarray(obs_state["velocity"], dtype=np.float32)
        )

    def is_ready(self):
        return len(self.robot_pos_buf) == self.history_steps

    def reset_smoothing(self):
        self.prev_proj_pos = None

    def local_plan(self, goal, update_smoothing=True):
        """
        返回 STA-GRU 预测的局部规划路径和当前风险值。

        仅在 LOCAL 模式下启用跨帧 EMA 平滑。
        """
        if not self.is_ready():
            return [], 0.0

        # 堆叠历史序列。
        robot_pos = np.stack(self.robot_pos_buf)
        robot_vel = np.stack(self.robot_vel_buf)
        obs_pos_all = np.stack(self.obs_pos_buf)
        obs_vel_all = np.stack(self.obs_vel_buf)

        T, M, _ = obs_pos_all.shape

        last_robot_pos = robot_pos[-1]
        last_obs_pos = obs_pos_all[-1]
        dists = np.linalg.norm(last_obs_pos - last_robot_pos, axis=1)

        idx = np.argsort(dists)[: self.num_obstacles]

        obs_pos = obs_pos_all[:, idx, :]
        obs_vel = obs_vel_all[:, idx, :]

        if obs_pos.shape[1] < self.num_obstacles:
            pad_n = self.num_obstacles - obs_pos.shape[1]
            obs_pos = np.pad(obs_pos, ((0, 0), (0, pad_n), (0, 0)))
            obs_vel = np.pad(obs_vel, ((0, 0), (0, pad_n), (0, 0)))

        goal_rel = np.asarray(goal, dtype=np.float32)[None, :] - robot_pos

        robot_pos = robot_pos[None]
        robot_vel = robot_vel[None]
        obs_pos = obs_pos[None]
        obs_vel = obs_vel[None]
        goal_rel = goal_rel[None]

        # 使用训练时保存的 scaler 做归一化。
        robot_pos = self._apply_scaler(robot_pos, self.rp_scaler)
        robot_vel = self._apply_scaler(robot_vel, self.rv_scaler)
        obs_pos = self._apply_scaler(obs_pos, self.op_scaler)
        obs_vel = self._apply_scaler(obs_vel, self.ov_scaler)
        goal_rel = self._apply_scaler(goal_rel, self.gr_scaler)

        robot_pos = torch.from_numpy(robot_pos).to(self.device)
        robot_vel = torch.from_numpy(robot_vel).to(self.device)
        obs_pos = torch.from_numpy(obs_pos).to(self.device)
        obs_vel = torch.from_numpy(obs_vel).to(self.device)
        goal_rel = torch.from_numpy(goal_rel).to(self.device)

        with torch.no_grad():
            future, risk = self.model(
                robot_pos, robot_vel, obs_pos, obs_vel, goal_rel
            )

        future = future.squeeze(0).cpu().numpy()
        risk_value = float(risk.squeeze().cpu().numpy())

        # 将网络输出的位移预测转换为世界坐标。
        last_pos_world = np.array(self.robot_pos_buf[-1])
        future_world = future + last_pos_world[None, :]

        # 跨帧 EMA 平滑，减少预测抖动。
        if update_smoothing and self.prev_proj_pos is not None:
            future_world = (self.smooth_lambda * future_world +
                            (1 - self.smooth_lambda) * self.prev_proj_pos)
        if update_smoothing:
            self.prev_proj_pos = future_world.copy()

        path = future_world[: self.execute_steps].tolist()
        return path, risk_value

    @staticmethod
    def _apply_scaler(x, scaler):
        """对 batch × time × ... 的张量批量施加 sklearn scaler 变换。"""
        B, T = x.shape[:2]
        x_flat = x.reshape(B * T, -1)
        x_norm = scaler.transform(x_flat)
        return x_norm.reshape(x.shape)
