"""
utils.py — Shared utilities for Assignment 3: Biped 1 m Platform Jump.

Contains
--------
  - SAC_CONFIG          Hyperparameters for Soft Actor-Critic (YOU will tune these)
  - Training constants  TOTAL_TIMESTEPS, EVAL_FREQ, EVAL_EPISODES, ROBOT_MASS_KG
  - RewardPlotCallback  Records episode rewards and saves a plot after training
  - BipedJumpEnv        Gymnasium environment — provided, do not modify
"""

# ===========================================================================
# Hyperparameters  (edit these for Task 3)
# ===========================================================================

# ============================================================
# TODO: Set the total number of training timesteps (e.g. 1_000_000).
# ============================================================
TOTAL_TIMESTEPS = None

# ============================================================
# TODO: Set how often (in steps) the evaluator runs during training (e.g. 10_000).
# ============================================================
EVAL_FREQ = None

# ============================================================
# TODO: Set the max steps per episode — must match BipedJumpEnv.max_steps (500).
# ============================================================
MAX_EPISODE_STEPS = None

# ---------------------------------------------------------------------------
# SAC  (Soft Actor-Critic) — the only algorithm used in this assignment
# ---------------------------------------------------------------------------
# ============================================================
# TODO: Fill in SAC_CONFIG with your chosen hyperparameters.
#       Required keys: policy, learning_rate, buffer_size, batch_size,
#                      tau, gamma, ent_coef, verbose.
# ============================================================
SAC_CONFIG = dict(
    policy        = "MlpPolicy",
    learning_rate = None,   
    buffer_size   = None,   
    batch_size    = None,   
    tau           = None,   
    gamma         = None,   
    ent_coef      = None,   
    verbose       = 1,
)

# ---------------------------------------------------------------------------
# Evaluation / metric settings  (do not change)
# ---------------------------------------------------------------------------
EVAL_EPISODES = 10
ROBOT_MASS_KG = 2.05   # used to compute Cost of Transport (CoT)


# ===========================================================================
# RewardPlotCallback
# ===========================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless training
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones",   [False])[0]

        self._current_episode_reward += reward
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True   # returning False would stop training

    def plot_rewards(self, save_path="reward_curve_sac.png"):
        if not self.episode_rewards:
            print("No episode rewards recorded yet.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")

        window = 20
        if len(self.episode_rewards) >= window:
            rolling = [
                sum(self.episode_rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.episode_rewards) + 1)
            ]
            plt.plot(rolling, color="red", linewidth=2, label=f"{window}-ep Rolling Avg")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training Reward Curve — Biped 1 m Jump")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Reward plot saved to {save_path}")


# ===========================================================================
# BipedJumpEnv  — provided environment, do not modify
# ===========================================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

_ASSEST_DIR = os.path.join(os.path.dirname(__file__), "assest")


class BipedJumpEnv(gym.Env):
    """
    Task: the biped robot spawns on top of a 1 m tall platform and must
    jump off, then land upright on the ground below.

    Phases
    ------
    1. On platform  
    2. In flight    
    3. Landing      

   
    """

    PLATFORM_H = 1.0          # top surface height (m)
    SPAWN_Z    = 1.0 + 0.81   # robot COM at spawn  (platform top + standing height)
    GROUND_Z   = 0.81         # robot COM when standing on flat ground

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        cid = p.connect(p.GUI if render else p.DIRECT)
        self.physics_client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep, physicsClientId=cid)

        self.max_steps         = 500
        self.step_counter      = 0
        self.land_stable_steps = 0

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=cid)

        # 1 m platform  (box 1.2 × 1.2 × 1.0 m, centre at z = 0.5)
        plat_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.5],
                                          physicsClientId=cid)
        plat_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.5],
                                       rgbaColor=[0.55, 0.27, 0.07, 1],
                                       physicsClientId=cid)
        self.platform_id = p.createMultiBody(0, plat_col, plat_vis,
                                              [0, 0, 0.5], physicsClientId=cid)

        # Robot
        urdf_path = os.path.join(_ASSEST_DIR, "biped_.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, self.SPAWN_Z],
                                    useFixedBase=False, physicsClientId=cid)
        p.changeDynamics(self.robot_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

        # Joint discovery
        self.joint_indices   = []
        self.joint_limits    = []
        self.left_foot_link  = 2
        self.right_foot_link = 5

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            ji = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits.append((ji[8], ji[9]))
            if b"left_foot"  in ji[12]: self.left_foot_link  = i
            if b"right_foot" in ji[12]: self.right_foot_link = i

        p.changeDynamics(self.robot_id, self.left_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)
        p.changeDynamics(self.robot_id, self.right_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)

        self.n_actuated = len(self.joint_indices)

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.n_actuated,), dtype=np.float32)
        obs_dim  = self.n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1
        obs_high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.prev_z     = self.SPAWN_Z
        self.has_landed = False
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # ============================================================
        # TODO: Reset the robot joints, base position, and counters, then return the initial observation.
        # ============================================================
        pass

    # ------------------------------------------------------------------
    def _get_obs(self):
        # ============================================================
        # TODO: Read joint states, base pose, velocities, and foot contacts, then return them as a single np.float32 array.
        # ============================================================
        pass

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, orn, lin_vel, landed_now):
        # ============================================================
        # TODO: Compute and return the reward from upright_penalty, z_progress, flight_bonus, and landing_reward.
        # ============================================================
        pass

    # ------------------------------------------------------------------
    def get_joint_indices(self):
        # ============================================================
        # TODO: Return a list of all non-fixed joint indices for the robot.
        # ============================================================
        pass

    def robot_initial_position(self):
        # ============================================================
        # TODO: Return the robot base position at the start of the episode.
        # ============================================================
        pass

    def robot_current_position(self):
        # ============================================================
        # TODO: Return the current robot base position.
        # ============================================================
        pass

    # ------------------------------------------------------------------
    def step(self, action):
        # ============================================================
        # TODO: Apply actions to joints, step the simulation, compute reward, check termination, and return (obs, reward, terminated, truncated, {}).
        # ============================================================
        pass

    # ------------------------------------------------------------------
    def close(self):
        p.disconnect(self.physics_client)
