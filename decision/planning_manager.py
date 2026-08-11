import numpy as np
from .collision_checker import CollisionChecker


class PlanningManager:
    """
    统一规划调度层。

    根据风险评分决定采用全局引导或 STA-GRU 局部避障。
    进入 LOCAL：risk > risk_enter 或碰撞检测到危险。
    退出 LOCAL：risk < risk_exit 后切回 GLOBAL。
    """

    def __init__(
        self,
        global_planner,
        local_planner,
        state_manager,
        execute_steps=3,
        collision_checker=None,
        risk_enter=0.9,
        risk_exit=0.5,
        smooth=True,
    ):
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.state_manager = state_manager
        self.execute_steps = execute_steps
        self.collision_checker = collision_checker or CollisionChecker()

        self.risk_enter = risk_enter
        self.risk_exit = risk_exit
        self.smooth = smooth

        self.current_mode = "GLOBAL"
        self.current_path = None
        self.current_risk = 0.0

    def update(self, robot_state, obstacles_state):
        """
        根据当前风险值和碰撞检测结果选择全局或局部模式。
        """
        goal = self.global_planner.get_reference()["goal"]

        if self.local_planner.is_ready():
            _, risk_value = self.local_planner.local_plan(goal, update_smoothing=False)
            self.current_risk = risk_value
        else:
            self.current_risk = 0.0

        ready = self.local_planner.is_ready()

        will_collide, clearance = self.collision_checker.check_global_path(
            robot_state, obstacles_state, goal
        )

        if ready:
            # 风险驱动进入局部避障，碰撞检测作为几何兜底。
            if self.current_risk > self.risk_enter:
                self.current_mode = "LOCAL"
            elif will_collide:
                self.current_mode = "LOCAL"
            else:
                self.current_mode = "GLOBAL"

            # 风险下降到安全线以下，退出局部避障。
            if self.current_mode == "LOCAL" and self.current_risk < self.risk_exit:
                self.current_mode = "GLOBAL"

        if self.current_mode == "LOCAL":
            local_path, _ = self.local_planner.local_plan(
                goal, update_smoothing=self.smooth)
            if local_path and len(local_path) > 0:
                local_arr = np.asarray(local_path[: self.execute_steps], dtype=float)
                if self._path_is_safe(local_arr, obstacles_state):
                    self.current_path = local_arr
                else:
                    self.current_path = self._emergency_evade(
                        robot_state, obstacles_state)
            else:
                self.current_path = self._run_global_planner(robot_state)
        else:
            self.current_path = self._run_global_planner(robot_state)

        return {
            "mode": self.current_mode,
            "path": self.current_path
        }

    def _path_is_safe(self, path, obstacles_state):
        """检查候选路径是否与当前障碍物无碰撞。"""
        if path is None or len(path) == 0:
            return False
        for obs in obstacles_state:
            obs_pos = np.array(obs["position"])
            obs_r = obs["radius"]
            for wp in path:
                d = np.linalg.norm(np.array(wp) - obs_pos)
                if d < obs_r + 0.3 + 0.1:
                    return False
        return True

    def _emergency_evade(self, robot_state, obstacles_state):
        """沿最近障碍物的垂直方向生成紧急避障路径点。"""
        robot_pos = np.array(robot_state["position"], dtype=float)

        min_dist = float("inf")
        closest_obs = None
        for obs in obstacles_state:
            d = np.linalg.norm(np.array(obs["position"]) - robot_pos)
            if d < min_dist:
                min_dist = d
                closest_obs = obs

        if closest_obs is None:
            return self._run_global_planner(robot_state)

        obs_pos = np.array(closest_obs["position"])
        to_obs = obs_pos - robot_pos
        dist = np.linalg.norm(to_obs)
        if dist < 1e-6:
            to_obs = np.array([1.0, 0.0])
        else:
            to_obs = to_obs / dist

        perp = np.array([-to_obs[1], to_obs[0]])

        goal = self.global_planner.goal
        if goal is not None:
            to_goal = goal - robot_pos
            if np.dot(perp, to_goal) < 0:
                perp = -perp

        step_size = 0.3
        waypoints = np.array([
            robot_pos + perp * step_size,
            robot_pos + perp * step_size * 2,
            robot_pos + perp * step_size * 3,
        ])
        return waypoints

    def _run_global_planner(self, robot_state):
        ref = self.global_planner.get_reference()

        if ref["reference_point"] is None:
            return np.asarray([robot_state["position"]], dtype=float)

        return np.asarray([ref["reference_point"]], dtype=float)
