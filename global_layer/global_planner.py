import numpy as np


class GlobalPathPlanner:
    """
    全局目标引导模块。

    根据机器人当前位置生成朝目标点的参考方向和参考点，
    供规划调度层和控制器使用。
    """

    def __init__(self, goal=None, reach_threshold=0.0):
        self.goal = goal
        self.reach_threshold = reach_threshold
        self.reference_direction = None
        self.reference_point = None

    def set_goal(self, goal):
        self.goal = np.array(goal, dtype=float)

    def reset(self, robot_state):
        self._update_reference(robot_state)

    def update(self, robot_state):
        self._update_reference(robot_state)

    def _update_reference(self, robot_state):
        if self.goal is None:
            self.reference_direction = None
            self.reference_point = None
            return

        pos = np.array(robot_state["position"], dtype=float)
        direction = self.goal - pos
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            self.reference_direction = np.zeros_like(direction)
        else:
            self.reference_direction = direction / norm

        self.reference_point = pos + self.reference_direction

    def get_reference(self):
        return {
            "direction": self.reference_direction,
            "reference_point": self.reference_point,
            "goal": self.goal
        }

    def is_goal_reached(self, robot_state):
        if self.goal is None:
            return False

        pos = np.array(robot_state["position"], dtype=float)
        return np.linalg.norm(self.goal - pos) < self.reach_threshold
