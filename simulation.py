import numpy as np


class Simulation:
    """闭环仿真流程：World → StateManager → PlanningManager → Controller → World。"""

    def __init__(
            self,
            world,
            state_manager,
            planning_manager,
            controller,
            execution_layer=None,
            dt=0.05,
            max_steps=500,
            verbose=True
    ):
        self.world = world
        self.state_manager = state_manager
        self.planning_manager = planning_manager
        self.controller = controller
        self.execution_layer = execution_layer

        self.dt = dt
        self.max_steps = max_steps
        self.verbose = verbose

        self.step_count = 0
        self.done = False
        self._prev_mode = "GLOBAL"

    def step(self):
        if self.done:
            return

        # 读取当前世界状态。
        robot_state = self.world.get_robot_state()
        obstacles_state = self.world.get_obstacles_state()

        # 更新风险状态，并选择当前参考路径。
        self.state_manager.update(robot_state, obstacles_state)
        planning_output = self.planning_manager.update(
            robot_state=robot_state,
            obstacles_state=obstacles_state
        )

        path = planning_output["path"]
        mode = planning_output["mode"]

        # 可选执行层检查，用于扩展机械臂可执行性验证。
        if self.execution_layer is not None:
            if path is not None and len(path) > 1:
                exec_result = self.execution_layer.execute(path)

                if not exec_result["success"]:
                    print(
                        f"[EXECUTION FAIL] step={self.step_count} "
                        f"reason={exec_result.get('reason', 'unknown')}"
                    )

        if mode == "LOCAL":
            self.controller.local_path = path
        else:
            self.controller.local_path = None
            # 从局部避障切回全局跟踪时，重新对齐朝向。
            if self._prev_mode == "LOCAL":
                goal = self.planning_manager.global_planner.goal
                if goal is not None:
                    delta = goal - robot_state["position"]
                    self.controller.last_yaw = np.arctan2(delta[1], delta[0])
        self._prev_mode = mode

        # 控制器根据当前参考路径生成速度指令。
        self.controller.update_state(
            position=robot_state["position"],
            velocity=robot_state["velocity"]
        )

        speed, yaw_rate = self.controller.follow_path()
        yaw = self.controller.last_yaw + yaw_rate * self.dt

        velocity_cmd = speed * np.array([
            np.cos(yaw),
            np.sin(yaw)
        ])

        self.world.step(velocity_cmd, self.dt)

        if self.planning_manager.global_planner.is_goal_reached(robot_state):
            self.done = True
            if self.verbose:
                print(f"[Simulation] Goal reached at step {self.step_count}")

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            if self.verbose:
                print("[Simulation] Max steps reached")

    def run(self):
        if self.verbose:
            print("[Simulation] Start")

        while not self.done:
            self.step()

        if self.verbose:
            print("[Simulation] Finished")
