import numpy as np


class Robot:
    """末端执行器在二维任务空间中的质点模型。"""

    def __init__(self, start, goal):
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.vel = np.zeros(2)

    def step(self, velocity, dt):
        self.vel = np.array(velocity, dtype=float)
        self.pos += self.vel * dt


class Obstacle:
    """
    二维动态障碍物，支持三种运动模式：
    - uniform：匀速直线运动。
    - variable：变加速曲线运动。
    - sinusoidal：基准速度上叠加正弦横向偏移。
    """

    def __init__(self, position, velocity, radius, homing=False):
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.base_vel = self.vel.copy()
        self.radius = radius
        self.homing = homing

        # 随机选择运动模式。
        self.motion = np.random.choice(["uniform", "variable", "sinusoidal"])
        self._time = 0.0
        self._direction = np.arctan2(self.vel[1], self.vel[0])
        self._speed = np.linalg.norm(self.vel)

    def step(self, dt, bounds, target=None):
        self._time += dt

        # homing 模式下，基准速度方向始终指向目标。
        if self.homing and target is not None:
            to_target = target - self.pos
            dist = np.linalg.norm(to_target)
            if dist > 1e-6:
                self._speed = np.linalg.norm(self.vel)
                self.vel = self._speed * to_target / dist
                self.base_vel = self.vel.copy()
                self._direction = np.arctan2(self.vel[1], self.vel[0])

        if self.motion == "uniform":
            self.pos += self.vel * dt

        elif self.motion == "variable":
            acc = 0.3 * np.sin(0.8 * self._time) + 0.15 * np.cos(1.3 * self._time)
            omega = 0.5 * np.cos(0.6 * self._time)
            self._speed += acc * dt
            self._speed = np.clip(self._speed, 0.05, 0.5)
            self._direction += omega * dt
            self.vel = self._speed * np.array([
                np.cos(self._direction), np.sin(self._direction)
            ])
            self.pos += self.vel * dt

        elif self.motion == "sinusoidal":
            base_dir = self.base_vel / (np.linalg.norm(self.base_vel) + 1e-6)
            lateral = np.array([-base_dir[1], base_dir[0]])
            amplitude = 0.3
            freq = 1.5
            self.vel = self.base_vel + amplitude * np.sin(freq * self._time) * lateral
            self.pos += self.vel * dt

        # 边界反弹。
        for i in range(2):
            if self.pos[i] < bounds[i, 0] or self.pos[i] > bounds[i, 1]:
                self.vel[i] *= -1
                self.base_vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], bounds[i, 0], bounds[i, 1])


class ObstacleGenerator:
    """根据预设场景生成动态障碍物。"""

    def __init__(self, n_obs, bounds, profile="balanced"):
        self.n_obs = n_obs
        self.bounds = bounds
        self.profile = profile

    def reset(self, robot_pos, goal, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if self.profile == "strong":
            return self._generate_strong(robot_pos, goal)
        elif self.profile == "balanced":
            return self._generate_balanced(robot_pos, goal)
        elif self.profile == "realistic":
            return self._generate_realistic()
        elif self.profile == "extreme":
            return self._generate_extreme(robot_pos, goal)
        else:
            raise ValueError(f"Unknown obstacle profile: {self.profile}")

    def _generate_strong(self, robot_pos, goal):
        obstacles = []
        n_interactive = max(1, self.n_obs // 2)

        direction = goal - robot_pos
        direction /= (np.linalg.norm(direction) + 1e-6)
        lateral = np.array([-direction[1], direction[0]])

        for _ in range(n_interactive):
            t = np.random.uniform(0.3, 0.7)
            base_pos = robot_pos + t * (goal - robot_pos)
            offset = np.random.uniform(-1.0, 1.0)
            pos = base_pos + offset * lateral

            future_robot = robot_pos + direction
            vel_dir = future_robot - pos
            vel_dir /= (np.linalg.norm(vel_dir) + 1e-6)

            speed = np.random.uniform(0.4, 0.9)
            vel = speed * vel_dir
            radius = np.random.uniform(0.25, 0.4)

            obstacles.append(Obstacle(pos, vel, radius))

        obstacles += self._generate_random(self.n_obs - n_interactive)
        return obstacles

    def _generate_extreme(self, robot_pos, goal):
        """生成靠近名义路径的高风险障碍物。"""
        obstacles = []
        direction = goal - robot_pos
        dist = np.linalg.norm(direction)
        direction /= (dist + 1e-6)
        lateral = np.array([-direction[1], direction[0]])

        for _ in range(self.n_obs):
            t = np.random.uniform(0.15, 0.30)
            base_pos = robot_pos + t * (goal - robot_pos)
            offset = 0.0
            pos = base_pos + offset * lateral

            vel_dir = robot_pos - pos
            vel_dir /= (np.linalg.norm(vel_dir) + 1e-6)
            speed = np.random.uniform(0.6, 1.0)
            vel = speed * vel_dir

            radius = np.random.uniform(0.4, 0.7)
            obstacles.append(Obstacle(pos, vel, radius, homing=True))

        return obstacles

    def _generate_balanced(self, robot_pos, goal):
        obstacles = []
        n_interactive = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])

        direction = goal - robot_pos
        direction /= (np.linalg.norm(direction) + 1e-6)
        lateral = np.array([-direction[1], direction[0]])

        for _ in range(n_interactive):
            t = np.random.uniform(0.2, 0.8)
            base_pos = robot_pos + t * (goal - robot_pos)
            offset = np.random.uniform(-1.2, 1.2)
            pos = base_pos + offset * lateral

            vel_dir = robot_pos - pos
            vel_dir /= (np.linalg.norm(vel_dir) + 1e-6)

            speed = np.random.uniform(0.3, 0.7)
            vel = speed * vel_dir
            radius = np.random.uniform(0.2, 0.35)

            obstacles.append(Obstacle(pos, vel, radius))

        obstacles += self._generate_random(self.n_obs - n_interactive)
        return obstacles

    def _generate_realistic(self):
        return self._generate_random(self.n_obs)

    def _generate_random(self, n):
        obstacles = []
        for _ in range(n):
            x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
            y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
            pos = np.array([x, y])

            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.2, 0.6)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            radius = np.random.uniform(0.2, 0.35)
            obstacles.append(Obstacle(pos, vel, radius))
        return obstacles


class World:
    """统一保存机器人、障碍物和工作空间状态。"""

    def __init__(self, robot, obstacles, bounds):
        self.robot = robot
        self.obstacles = obstacles
        self.bounds = bounds

    def step(self, action, dt):
        self.robot.step(action, dt)
        for obs in self.obstacles:
            obs.step(dt, self.bounds, target=self.robot.pos)

    def get_robot_state(self):
        return {
            "position": self.robot.pos.copy(),
            "velocity": self.robot.vel.copy(),
            "goal": self.robot.goal.copy()
        }

    def get_obstacles_state(self):
        return [
            {
                "position": obs.pos.copy(),
                "velocity": obs.vel.copy(),
                "radius": obs.radius
            }
            for obs in self.obstacles
        ]

    def reset(self, robot, obstacles):
        self.robot = robot
        self.obstacles = obstacles
