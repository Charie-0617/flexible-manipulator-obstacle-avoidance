import argparse
from pathlib import Path

import numpy as np
import torch

from global_layer.global_planner import GlobalPathPlanner
from decision.state_manager import StateManager
from decision.planning_manager import PlanningManager
from decision.collision_checker import CollisionChecker
from local_layer.local_planner import LocalObstacleAvoidance

from environment import Robot, ObstacleGenerator, World
from simulation import Simulation
from controller import Controller
from visualization import SimulationVisualizerDynamic

"""
动态避障轨迹规划主入口。

系统在二维末端执行器（EE）空间中运行，结合全局目标引导、风险判断、
STA-GRU 局部避障和闭环仿真，输出实际执行的末端轨迹与障碍物轨迹。
"""


def main():
    parser = argparse.ArgumentParser(description="运行动态避障轨迹规划 demo。")
    parser.add_argument("--no-gui", action="store_true", help="不显示 Matplotlib 动画窗口。")
    args = parser.parse_args()

    # 机器人起点与目标点。
    robot_start = np.array([0.5, 0.0])
    robot_goal = np.array([6.5, 9.0])
    robot = Robot(robot_start, robot_goal)
    goal = robot_goal

    # 仿真环境与动态障碍物。
    bounds = np.array([[-1, 10], [-1, 10]])
    n_obs = 5
    obstacle_generator = ObstacleGenerator(
        n_obs=n_obs,
        bounds=bounds,
        profile="strong"
    )
    obstacles = obstacle_generator.reset(robot.pos, goal)
    world = World(robot=robot, obstacles=obstacles, bounds=bounds)

    # 全局目标引导与风险状态管理。
    global_planner = GlobalPathPlanner(goal=goal, reach_threshold=0.3)
    state_manager = StateManager(
        robot_radius=0.3,
        ttc_threshold=1.5
    )

    # 几何碰撞检测作为直线全局跟踪的安全兜底。
    collision_checker = CollisionChecker(
        robot_radius=0.3,
        safety_margin=0.3,
        lookahead_steps=25,
        dt=0.05
    )

    # STA-GRU 局部避障模型。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path('models/best_model.pth')
    scaler_path = Path('models/scalers.pkl')
    local_planner = LocalObstacleAvoidance(
        model_path=model_path,
        scaler_path=scaler_path,
        history_steps=20,
        future_steps=10,
        execute_steps=3,
        num_obstacles=3,
        device=device
    )

    # 统一调度全局规划、局部避障和安全检查。
    planning_manager = PlanningManager(
        state_manager=state_manager,
        global_planner=global_planner,
        local_planner=local_planner,
        execute_steps=3,
        collision_checker=collision_checker,
        risk_enter=0.8,
        risk_exit=0.45,
    )

    # 底层速度控制器。
    controller = Controller(
        global_path=[robot_start, robot_goal],
        local_path=None,
        max_speed=1.0,
        max_steering_angle=np.pi/4,
        max_acc=1.0,
        max_yaw_rate=np.pi,
        dt=0.05
    )

    simulation = Simulation(
        world=world,
        state_manager=state_manager,
        planning_manager=planning_manager,
        controller=controller,
        execution_layer=None,
        dt=0.05,
        max_steps=500,
        verbose=True
    )

    robot_hist = []
    obs_hist = []
    global_refs_hist = []
    obs_radii = None

    # 闭环仿真主循环。
    for step in range(simulation.max_steps):
        if simulation.done:
            break

        simulation.step()

        robot_state = simulation.world.get_robot_state()
        obstacles_state = simulation.world.get_obstacles_state()

        if obs_radii is None:
            obs_radii = [obs["radius"] for obs in obstacles_state]

        obs_state_for_sta = {
            "position": np.array(
                [obs["position"] for obs in obstacles_state],
                dtype=np.float32
            ),
            "velocity": np.array(
                [obs["velocity"] for obs in obstacles_state],
                dtype=np.float32
            )
        }
        local_planner.update_history(robot_state, obs_state_for_sta)

        robot_hist.append(robot_state["position"].copy())
        obs_hist.append([
            np.array([obs["position"][0], obs["position"][1], obs["radius"]])
            for obs in obstacles_state
        ])

        global_ref = global_planner.get_reference()["goal"]
        global_refs_hist.append(global_ref.copy())

        if step % 10 == 0:
            print(
                f"[Step {step}] "
                f"Mode={planning_manager.current_mode}, "
                f"Risk={planning_manager.current_risk:.3f}, "
                f"TTC={state_manager.last_min_ttc:.2f}"
            )

    robot_hist = np.array(robot_hist)
    obs_hist = np.array(obs_hist)

    print(f"[INFO] Simulation finished. Steps = {len(robot_hist)}")

    if not args.no_gui:
        visualizer = SimulationVisualizerDynamic(
            robot_hist=robot_hist,
            obs_hist=obs_hist,
            obs_radii=obs_radii,
            local_paths=[],
            global_refs=global_refs_hist,
            bounds=bounds,
            dt=0.05
        )
        visualizer.run()

    # 导出实际执行轨迹，供后续分析或跟踪模块使用。
    Path("planning_results").mkdir(exist_ok=True)

    np.save(
        "planning_results/ee_traj_from_system.npy",
        robot_hist
    )

    np.save(
        "planning_results/obs_traj_from_system.npy",
        obs_hist
    )

    print(
        f"[INFO] Exported EXECUTED EE trajectory "
        f"({robot_hist.shape[0]} steps) "
        f"to planning_results/ee_traj_from_system.npy"
    )
    print(
        f"[INFO] Exported OBSTACLE trajectory "
        f"({obs_hist.shape[0]} steps, {obs_hist.shape[1]} obstacles) "
        f"to planning_results/obs_traj_from_system.npy"
    )


if __name__ == "__main__":
    main()
