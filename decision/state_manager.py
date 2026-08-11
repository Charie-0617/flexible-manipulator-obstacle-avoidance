import numpy as np


class StateManager:
    """
    基于 TTC 的风险状态评估模块。

    根据碰撞时间判断是否需要激活局部避障，带滞回避免频繁切换。
    """

    def __init__(self, robot_radius=0.3, ttc_threshold=0.8, hysteresis=0.2):
        self.robot_radius = robot_radius
        self.ttc_threshold = ttc_threshold
        self.hysteresis = hysteresis

        self.is_local_avoidance_active = False
        self.last_min_ttc = np.inf

    def compute_ttc(self, robot_state, obs_state):
        """
        计算机器人与单个障碍物之间的碰撞时间。
        当前相对速度接近 0 或背离时返回 inf。
        """
        p_r = np.array(robot_state["position"], dtype=float)
        v_r = np.array(robot_state["velocity"], dtype=float)

        p_o = np.array(obs_state["position"], dtype=float)
        v_o = np.array(obs_state["velocity"], dtype=float)

        r_o = obs_state.get("radius", 0.3)
        R = self.robot_radius + r_o

        p = p_o - p_r
        v = v_o - v_r

        v_norm2 = np.dot(v, v)
        if v_norm2 < 1e-6:
            return np.inf

        t_star = -np.dot(p, v) / v_norm2
        if t_star <= 0:
            return np.inf

        d_min = np.linalg.norm(p + v * t_star)
        return t_star if d_min <= R else np.inf

    def min_ttc_over_obstacles(self, robot_state, obstacles_state):
        min_ttc = np.inf
        for obs in obstacles_state:
            min_ttc = min(min_ttc, self.compute_ttc(robot_state, obs))
        return min_ttc

    def update(self, robot_state, obstacles_state):
        """更新最小 TTC 并判断是否激活局部避障。"""
        min_ttc = self.min_ttc_over_obstacles(robot_state, obstacles_state)
        self.last_min_ttc = min_ttc

        if not self.is_local_avoidance_active:
            if min_ttc < self.ttc_threshold:
                self.is_local_avoidance_active = True
        else:
            if min_ttc > self.ttc_threshold + self.hysteresis:
                self.is_local_avoidance_active = False

        return {
            "use_local": self.is_local_avoidance_active,
            "min_ttc": min_ttc
        }
