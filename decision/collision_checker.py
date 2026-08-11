"""
基于几何的碰撞检测模块。

检查机器人在直线朝目标匀速运动时，在预测时域内是否会与障碍物碰撞，
作为全局避障的安全兜底。
"""
import numpy as np


class CollisionChecker:
    """
    前向仿真全局直线路径，判断碰撞风险。

    机器人和障碍物均采用匀速直线运动假设。
    """

    def __init__(self, robot_radius=0.3, safety_margin=0.15,
                 lookahead_steps=15, dt=0.05):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.lookahead_steps = lookahead_steps
        self.dt = dt
        self.max_speed = 1.0

    def _predict_global_path(self, robot_pos, goal, steps):
        """生成朝目标匀速直线运动的预测路径。"""
        direction = goal - robot_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.tile(robot_pos, (steps, 1))
        direction = direction / dist
        speed = min(self.max_speed, dist / (steps * self.dt))
        t = np.arange(1, steps + 1) * self.dt
        return robot_pos + t[:, None] * (speed * direction)

    def _predict_obstacle_paths(self, obstacles_state, steps):
        """根据当前速度线性外推障碍物位置。"""
        paths = []
        for obs in obstacles_state:
            pos = np.array(obs["position"], dtype=float)
            vel = np.array(obs["velocity"], dtype=float)
            t = np.arange(1, steps + 1) * self.dt
            pred = pos + t[:, None] * vel
            paths.append({"positions": pred, "radius": obs["radius"]})
        return paths

    def check_global_path(self, robot_state, obstacles_state, goal):
        """
        检查直线全局路径是否安全。

        返回值
        -------
        will_collide : bool
            是否预测到碰撞。
        min_clearance : float
            预测时域内的最小间隙。
        """
        robot_pos = np.array(robot_state["position"], dtype=float)

        global_path = self._predict_global_path(robot_pos, goal, self.lookahead_steps)
        obs_paths = self._predict_obstacle_paths(obstacles_state, self.lookahead_steps)
        if not obs_paths:
            return False, float("inf")

        min_clearance = float("inf")
        for obs in obs_paths:
            for k in range(self.lookahead_steps):
                dist = np.linalg.norm(global_path[k] - obs["positions"][k])
                clearance = dist - self.robot_radius - obs["radius"] - self.safety_margin
                min_clearance = min(min_clearance, clearance)
                if clearance < 0:
                    return True, clearance

        return False, min_clearance
