import numpy as np


class Controller:
    """
    二维末端执行器速度控制器。

    控制器优先跟踪局部避障路径；没有局部路径时，直接朝全局目标点运动。
    速度会根据目标距离自适应减小，角速度和加速度均进行限幅。
    """

    def __init__(
        self,
        global_path,
        local_path=None,
        max_speed=1.0,
        max_steering_angle=np.pi / 2,
        max_acc=1.0,
        max_yaw_rate=np.pi,
        dt=0.05,
        slow_down_radius=0.5
    ):
        self.global_path = global_path
        self.local_path = local_path

        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.max_acc = max_acc
        self.max_yaw_rate = max_yaw_rate
        self.dt = dt
        self.slow_down_radius = slow_down_radius

        self.last_yaw = 0.0
        self.last_speed = 0.0
        self.current_position = None
        self.current_velocity = None

    def update_state(self, position, velocity):
        self.current_position = position
        self.current_velocity = velocity

        speed = np.linalg.norm(velocity)
        yaw = np.arctan2(velocity[1], velocity[0]) if speed > 1e-3 else self.last_yaw

        self.last_speed = speed
        self.last_yaw = yaw

    def follow_path(self):
        # 局部路径优先；没有局部路径时跟踪全局目标。
        path = self.local_path if (self.local_path is not None and len(self.local_path) > 0) else None
        if path is None or len(path) == 0:
            path = [self.global_path[-1]]

        lookahead_idx = min(2, len(path) - 1)
        target_position = path[lookahead_idx]

        delta = target_position - self.current_position
        distance = np.linalg.norm(delta)
        desired_yaw = np.arctan2(delta[1], delta[0])

        yaw_error = self._wrap_to_pi(desired_yaw - self.last_yaw)
        yaw_rate = np.clip(yaw_error / self.dt, -self.max_yaw_rate, self.max_yaw_rate)

        # 靠近目标时减速，避免末端执行器在目标附近震荡。
        if distance > self.slow_down_radius:
            desired_speed = self.max_speed
        else:
            desired_speed = self.max_speed * (distance / self.slow_down_radius)

        speed_error = desired_speed - self.last_speed
        acc = np.clip(speed_error / self.dt, -self.max_acc, self.max_acc)
        speed = self.last_speed + acc * self.dt

        return speed, yaw_rate

    @staticmethod
    def _wrap_to_pi(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
