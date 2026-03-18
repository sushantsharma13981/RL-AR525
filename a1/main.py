"""
==========================================================================
  MAIN.PY - UR5 GRID NAVIGATION Final working simulation file 
==========================================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random

from utils import GridEnv, policy_iteration, value_iteration


# ==========================================================================
# HEATMAP FUNCTION
# ==========================================================================

def plot_value_heatmap(V, rows, cols, path=None, start=None, goal=None,
                       title="Value Function Heatmap"):

    V_grid = np.reshape(V, (rows, cols))

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(V_grid, annot=True, fmt=".2f", cmap="RdYlGn", cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.invert_yaxis()

    if path:
        coords = []
        for state in path:
            row = state // cols
            col = state % cols
            coords.append((col + 0.5, row + 0.5))
        if len(coords) > 1:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=(1.0, 1.0, 0.0, 0.5), linewidth=3, zorder=3)

    plt.tight_layout()
    plt.show()


# ==========================================================================
# GRID VISUALIZATION HELPERS
# ==========================================================================

def state_to_position(state, rows, cols, grid_size=0.10,
                      table_center=[0, -0.3, 0.65], z_offset=0.10):

    row = state // cols
    col = state % cols

    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset

    return [x, y, z]


def draw_grid_lines(rows, cols, grid_size=0.10,
                    table_center=[0, -0.3, 0.65]):

    z = table_center[2] + 0.001

    x_start = table_center[0] - (cols/2) * grid_size
    x_end = table_center[0] + (cols/2) * grid_size
    y_start = table_center[1] - (rows/2) * grid_size
    y_end = table_center[1] + (rows/2) * grid_size

    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], [0,0,0], 2)

    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], [0,0,0], 2)


def draw_square(pos, color, grid_size=0.10):

    half = (grid_size / 2) * 0.9
    z = pos[2] + 0.005

    p.addUserDebugLine([pos[0]-half,pos[1]-half,z],
                       [pos[0]+half,pos[1]-half,z], color, 4)
    p.addUserDebugLine([pos[0]+half,pos[1]-half,z],
                       [pos[0]+half,pos[1]+half,z], color, 4)
    p.addUserDebugLine([pos[0]+half,pos[1]+half,z],
                       [pos[0]-half,pos[1]+half,z], color, 4)
    p.addUserDebugLine([pos[0]-half,pos[1]+half,z],
                       [pos[0]-half,pos[1]-half,z], color, 4)


def compute_path_metrics(env, path, gamma):

    metrics = {
        "length": max(len(path) - 1, 0),
        "total_reward": 0.0,
        "discounted_return": 0.0,
        "goal_reached": bool(path) and path[-1] == env.goal,
        "unique_states": len(set(path)) if path else 0,
        "valid": True,
        "loop_detected": False
    }

    total_reward = 0.0
    discounted_return = 0.0

    for step_idx in range(len(path) - 1):
        state = path[step_idx]
        next_state = path[step_idx + 1]

        transition = None
        for action in range(env.nA):
            for prob, ns, reward, done in env.P[state][action]:
                if prob > 0 and ns == next_state:
                    transition = (reward, done)
                    break
            if transition is not None:
                break

        if transition is None:
            metrics["valid"] = False
            break

        reward, done_flag = transition
        total_reward += reward
        discounted_return += (gamma ** step_idx) * reward

        if done_flag:
            break

    metrics["total_reward"] = total_reward
    metrics["discounted_return"] = discounted_return
    metrics["loop_detected"] = (not metrics["goal_reached"]) and metrics["length"] > 0

    return metrics


def run_dp_suite(env, gamma, theta):

    results = {}

    t0 = time.time()
    policy_pi, V_pi, info_pi = policy_iteration(env, gamma=gamma, theta=theta,
                                                return_metadata=True)
    time_pi = time.time() - t0
    path_pi = env.get_optimal_path(policy_pi)
    metrics_pi = compute_path_metrics(env, path_pi, gamma)
    info_pi_formatted = {
        "iterations": info_pi["policy_iterations"],
        "policy_changes": info_pi["policy_changes"],
        "total_evaluation_sweeps": info_pi["total_evaluation_sweeps"],
        "evaluation_sweeps": info_pi["evaluation_sweeps"]
    }
    results["Policy Iteration"] = {
        "policy": policy_pi,
        "V": V_pi,
        "time": time_pi,
        "info": info_pi_formatted,
        "path": path_pi,
        "path_metrics": metrics_pi
    }

    t0 = time.time()
    policy_vi, V_vi, info_vi = value_iteration(env, gamma=gamma, theta=theta,
                                               return_metadata=True)
    time_vi = time.time() - t0
    path_vi = env.get_optimal_path(policy_vi)
    metrics_vi = compute_path_metrics(env, path_vi, gamma)
    info_vi_formatted = {
        "iterations": info_vi["iterations"],
        "final_delta": info_vi["final_delta"],
        "delta_history": info_vi["delta_history"]
    }
    results["Value Iteration"] = {
        "policy": policy_vi,
        "V": V_vi,
        "time": time_vi,
        "info": info_vi_formatted,
        "path": path_vi,
        "path_metrics": metrics_vi
    }

    return results


def print_analysis_summary(results, gamma, theta):

    print("\n===== ANALYSIS SUMMARY =====")
    print(f"gamma={gamma}, theta={theta}")

    for reward_label, data in results.items():
        print(f"\nReward Structure: {reward_label}")
        for algo_label, bundle in data.items():
            time_spent = bundle["time"]
            info = bundle["info"]
            metrics = bundle["path_metrics"]

            iterations = info.get("iterations", info.get("policy_iterations", "-"))
            iterations_display = str(iterations)
            eval_sweeps = info.get("total_evaluation_sweeps", "-")
            eval_display = str(eval_sweeps)
            policy_changes = info.get("policy_changes", "-")
            policy_changes_display = str(policy_changes)
            goal_reached = "Yes" if metrics["goal_reached"] else "No"

            print(
                f"  {algo_label}: time={time_spent:.6f}s, "
                f"iterations={iterations_display}, eval sweeps={eval_display}, "
                f"policy changes={policy_changes_display}, "
                f"path length={metrics['length']}, total reward={metrics['total_reward']:.2f}, "
                f"discounted return={metrics['discounted_return']:.2f}, goal reached={goal_reached}"
            )


def plot_analysis_comparison(results):

    scenarios = list(results.keys())
    algorithms = []
    for scenario_data in results.values():
        for algo in scenario_data.keys():
            if algo not in algorithms:
                algorithms.append(algo)

    if not scenarios or not algorithms:
        return

    algorithms.sort()
    colors = sns.color_palette("Set2", n_colors=len(algorithms))
    metrics_config = [
        ("Runtime (s)", "Time (s)", lambda entry: entry["time"]),
        ("Path Length", "Steps", lambda entry: entry["path_metrics"]["length"]),
        ("Discounted Return", "Value", lambda entry: entry["path_metrics"]["discounted_return"])
    ]

    rows = len(metrics_config)
    cols = len(scenarios)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for row_idx, (title_suffix, ylabel, extractor) in enumerate(metrics_config):
        for col_idx, scenario in enumerate(scenarios):
            scenario_data = results[scenario]
            values = []
            for algo in algorithms:
                entry = scenario_data.get(algo)
                values.append(extractor(entry) if entry else np.nan)

            x = np.arange(len(algorithms))
            ax = axes[row_idx, col_idx]
            ax.bar(x, values, color=colors)
            ax.set_title(f"{scenario} – {title_suffix}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            if row_idx == rows - 1:
                ax.set_xticklabels(algorithms, rotation=15)
            else:
                ax.set_xticklabels([])
            ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()


def get_ur5_joint_indices(ur5_id):

    joint_indices = []
    ee_link_index = None
    ee_tip_index = None

    for idx in range(p.getNumJoints(ur5_id)):
        info = p.getJointInfo(ur5_id, idx)
        joint_type = info[2]
        link_name = info[12].decode("utf-8")

        if joint_type == p.JOINT_REVOLUTE:
            joint_indices.append(idx)

        if link_name == "ee_link":
            ee_link_index = idx
        if link_name == "ee_tip":
            ee_tip_index = idx

    if ee_tip_index is not None:
        ee_link_index = ee_tip_index
    elif ee_link_index is None and joint_indices:
        ee_link_index = joint_indices[-1]

    return joint_indices, ee_link_index


def move_ur5_along_path(ur5_id, ee_link_index, joint_indices, path, env,
                        grid_size=0.10, table_center=[0, -0.3, 0.65],
                        z_offset=0.04, ee_orn_euler=(0.0, 3.14, 0.0),
                        steps_per_edge=30):

    if not path:
        return

    target_orn = p.getQuaternionFromEuler(ee_orn_euler)

    rest_template = [0.0, -1.2, 1.7, -1.9, -1.57, 0.0]
    rest_poses = rest_template[:len(joint_indices)]
    joint_ranges = [6.28] * len(joint_indices)
    joint_limits = [(-6.28, 6.28)] * len(joint_indices)
    lower_limits = [limit[0] for limit in joint_limits]
    upper_limits = [limit[1] for limit in joint_limits]

    fixed_z = table_center[2] + z_offset
    prev_ee_pos = None

    start_pos = state_to_position(path[0], env.rows, env.cols,
                                  grid_size, table_center, z_offset)
    start_target = [start_pos[0], start_pos[1], fixed_z]

    start_joints = p.calculateInverseKinematics(
        ur5_id,
        ee_link_index,
        start_target,
        target_orn,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=500,
        residualThreshold=1e-5
    )
    start_joints = [start_joints[i] for i in range(len(joint_indices))]

    for joint_idx, body_joint in enumerate(joint_indices):
        p.resetJointState(ur5_id, body_joint, start_joints[joint_idx], targetVelocity=0.0)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    link_state = p.getLinkState(ur5_id, ee_link_index)
    prev_ee_pos = link_state[0]

    for segment_idx in range(len(path) - 1):
        start_state = path[segment_idx]
        end_state = path[segment_idx + 1]

        p0 = state_to_position(start_state, env.rows, env.cols,
                               grid_size, table_center, z_offset)
        p1 = state_to_position(end_state, env.rows, env.cols,
                               grid_size, table_center, z_offset)

        for step in range(1, steps_per_edge + 1):
            alpha = step / steps_per_edge
            x = p0[0] + (p1[0] - p0[0]) * alpha
            y = p0[1] + (p1[1] - p0[1]) * alpha
            target_pos = [x, y, fixed_z]

            joint_positions = p.calculateInverseKinematics(
                ur5_id,
                ee_link_index,
                target_pos,
                target_orn,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=500,
                residualThreshold=1e-5
            )
            joint_positions = [joint_positions[i] for i in range(len(joint_indices))]

            for joint_idx, body_joint in enumerate(joint_indices):
                p.resetJointState(ur5_id, body_joint, joint_positions[joint_idx], targetVelocity=0.0)

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            link_state = p.getLinkState(ur5_id, ee_link_index)
            ee_pos = link_state[0]
            if prev_ee_pos is not None:
                p.addUserDebugLine(prev_ee_pos, ee_pos, [0, 1, 0], 2, 0)
            prev_ee_pos = ee_pos

        for _ in range(20):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)



# ==========================================================================
# MAIN EXECUTION
# ==========================================================================

if __name__ == "__main__":

    ROWS, COLS = 8, 8
    START, GOAL = 0, ROWS * COLS - 1
    GAMMA = 0.99
    THETA = 1e-6

    all_states = list(range(ROWS * COLS))
    all_states.remove(START)
    all_states.remove(GOAL)
    obstacle_states = random.sample(all_states, k=16)

    dense_env = GridEnv(rows=ROWS, cols=COLS,
                        start=START, goal=GOAL,
                        obstacles=obstacle_states,
                        reward_step=-1.0,
                        reward_goal=10.0,
                        reward_obstacle=-50.0)

    sparse_env = GridEnv(rows=ROWS, cols=COLS,
                         start=START, goal=GOAL,
                         obstacles=obstacle_states,
                         reward_step=0.0,
                         reward_goal=10.0,
                         reward_obstacle=-50.0)

    print("\nObstacles at:", obstacle_states)

    reward_scenarios = {
        "Dense (-1 per step)": dense_env,
        "Sparse (0 per step)": sparse_env
    }

    analysis_results = {}
    for label, env_instance in reward_scenarios.items():
        analysis_results[label] = run_dp_suite(env_instance, GAMMA, THETA)

    dense_label = "Dense (-1 per step)"
    dense_results = analysis_results[dense_label]
    env_for_sim = dense_env

    choice = input("\nVisualize:\n1. Policy Iteration\n2. Value Iteration\nEnter: ")

    if choice == "1":
        selected_key = "Policy Iteration"
    else:
        selected_key = "Value Iteration"

    selected_entry = dense_results[selected_key]
    policy = selected_entry["policy"]
    V = selected_entry["V"]
    path = selected_entry["path"]
    path_metrics = selected_entry["path_metrics"]
    name = selected_key

    print(f"\nSelected {name}")
    print("Optimal Path:", path)
    print(
        f"Path length={path_metrics['length']} steps, "
        f"total reward={path_metrics['total_reward']:.2f}, "
        f"discounted return={path_metrics['discounted_return']:.2f}, "
        f"goal reached={'Yes' if path_metrics['goal_reached'] else 'No'}"
    )

    # ----------------------------------------------------------------------
    # PYBULLET SIMULATION
    # ----------------------------------------------------------------------

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)

    table_center = [0, -0.3, 0.65]
    grid_size = 0.10

    p.resetDebugVisualizerCamera(1.5,45,-30,table_center)

    p.loadURDF("plane.urdf")

    base_path = "assest"

    p.loadURDF(os.path.join(base_path,"table/table.urdf"),
               [0,-0.3,0], globalScaling=2.0)

    p.loadURDF(os.path.join(base_path,"robot_stand.urdf"),
               [0,-0.8,0], useFixedBase=True)

    ur5_id = p.loadURDF(os.path.join(base_path,"ur5.urdf"),
                        [0,-0.8,0.65], useFixedBase=True)

    draw_grid_lines(ROWS, COLS, grid_size, table_center)

    draw_square(state_to_position(START,ROWS,COLS,grid_size,table_center,0),
                [1,1,0])

    draw_square(state_to_position(GOAL,ROWS,COLS,grid_size,table_center,0),
                [1,0,0])

    obs_path = os.path.join(base_path,
                            "cube_and_square/cube_small_cyan.urdf")

    for obs in obstacle_states:
        pos = state_to_position(obs, ROWS, COLS, grid_size, table_center, 0.025)
        p.loadURDF(obs_path, pos)

    # ----------------------------------------------------------------------
    # ROBOT MOTION  (STABLE IK)
    # ----------------------------------------------------------------------

    joint_indices, ee_link_index = get_ur5_joint_indices(ur5_id)
    downward_euler = (0.0, -1.5 * np.pi, np.pi)

    move_ur5_along_path(
        ur5_id,
        ee_link_index,
        joint_indices,
        path,
        env_for_sim,
        grid_size=grid_size,
        table_center=table_center,
        z_offset=0.04,
        ee_orn_euler=downward_euler
    )

    print("\nTask Complete!")

    p.disconnect()
    plot_value_heatmap(V, ROWS, COLS, path=path, start=START, goal=GOAL, title=name)

    print_analysis_summary(analysis_results, GAMMA, THETA)
    plot_analysis_comparison(analysis_results)

