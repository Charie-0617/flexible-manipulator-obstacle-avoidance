import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np


class SimulationVisualizerDynamic:
    """动态显示机器人、障碍物和规划参考路径。"""

    def __init__(self, robot_hist, obs_hist, obs_radii, local_paths, global_refs, bounds,
                 dt=0.05):
        """
        参数
        ----
        robot_hist : array-like, shape [T, 2]
            机器人历史位置。
        obs_hist : array-like, shape [T, N, 3]
            障碍物历史位置和半径。
        obs_radii : list[float]
            障碍物半径。
        local_paths : list
            每一帧对应的局部规划路径。
        global_refs : list
            每一帧对应的全局参考点。
        bounds : ndarray, shape [2, 2]
            工作空间边界。
        dt : float
            动画帧间隔。
        """
        self.robot_hist = robot_hist
        self.obs_hist = obs_hist
        self.obs_radii = obs_radii
        self.local_paths = local_paths
        self.global_refs = global_refs
        self.bounds = bounds
        self.dt = dt
        self.n_steps = len(robot_hist)
        self.n_obs = obs_hist.shape[1] if len(obs_hist) > 0 else 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(bounds[0, 0] - 1, bounds[0, 1] + 1)
        self.ax.set_ylim(bounds[1, 0] - 1, bounds[1, 1] + 1)
        self.ax.set_aspect('equal')
        self.ax.set_title("Dynamic Obstacle Avoidance Simulation")

        self.robot_patch = patches.Circle((0, 0), 0.2, color='blue', label='Robot')
        self.ax.add_patch(self.robot_patch)
        self.goal_patch = patches.Circle(tuple(robot_hist[-1]), 0.2, color='green', label='Goal')
        self.ax.add_patch(self.goal_patch)

        self.obs_patches = []
        for i in range(self.n_obs):
            label = 'Obstacle' if i == 0 else '_nolegend_'
            patch = patches.Circle((0, 0), self.obs_radii[i], color='red', alpha=0.6, label=label)
            self.obs_patches.append(patch)
            self.ax.add_patch(patch)

        self.robot_path_line, = self.ax.plot([], [], 'b--', linewidth=1, label='Robot Trajectory')
        self.local_path_line, = self.ax.plot([], [], 'orange', linestyle='--', linewidth=1, label='Local Plan')
        self.global_path_line, = self.ax.plot([], [], 'gray', linestyle='-', linewidth=1, label='Global Path')

        self.ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fontsize=9
        )

    def _update(self, frame):
        frame = min(frame, self.n_steps - 1)

        robot_pos = self.robot_hist[frame]
        self.robot_patch.center = tuple(robot_pos)

        traj = np.array(self.robot_hist[:frame + 1])
        self.robot_path_line.set_data(traj[:, 0], traj[:, 1])

        for i, patch in enumerate(self.obs_patches):
            obs_pos = self.obs_hist[frame, i]
            patch.center = tuple(obs_pos)

        local_future = self.local_paths[frame] if frame < len(self.local_paths) else []
        if local_future:
            local_future = np.array(local_future)
            self.local_path_line.set_data(local_future[:, 0], local_future[:, 1])
        else:
            self.local_path_line.set_data([], [])

        global_ref = self.global_refs[frame] if frame < len(self.global_refs) else None
        if global_ref is not None:
            self.global_path_line.set_data([robot_pos[0], global_ref[0]],
                                           [robot_pos[1], global_ref[1]])
        else:
            self.global_path_line.set_data([], [])

        return self.robot_patch, self.goal_patch, self.obs_patches, \
               self.robot_path_line, self.local_path_line, self.global_path_line

    def run(self):
        animation.FuncAnimation(
            self.fig,
            self._update,
            frames=self.n_steps,
            interval=self.dt * 1000,
            blit=False,
            repeat=False
        )
        plt.show()
