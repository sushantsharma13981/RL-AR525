"""
main.py — Assignment 3: Biped RL (1 m Platform Jump with SAC)

Usage examples
--------------
# View the environment (biped + stair in GUI, no model needed):
    python main.py --mode view

# Train SAC (timesteps set in utils.py):
    python main.py --mode train

# Train SAC for a custom number of steps:
    python main.py --mode train --timesteps 500000

# Evaluate the best saved checkpoint (10 episodes, headless):
    python main.py --mode test

# Evaluate with GUI rendering:
    python main.py --mode test --render --episodes 5

# Evaluate a specific model file:
    python main.py --mode test --model_path "models/sac_best/best_model"
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils import (
    BipedJumpEnv, RewardPlotCallback,
    TOTAL_TIMESTEPS, EVAL_FREQ,
    SAC_CONFIG,
    EVAL_EPISODES, ROBOT_MASS_KG,
)

# ── Algorithm registry ────────────────────────────────────────────────────────
# TODO: Register the SAC algorithm with its config from utils.py.
ALGO_MAP = {
    # ============================================================
    # TODO: Add the SAC mapping
    # ============================================================
}


# ── Environment Preview ────────────────────────────────────────────────────────
def view():
    """Spawns the biped + stair in GUI mode. Press Ctrl+C to quit."""
    import pybullet_data

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)

    assest = os.path.join(os.path.dirname(__file__), "assest")
    p.loadURDF(os.path.join(assest, "biped_.urdf"), [0, 0, 0.81],
               useFixedBase=False, physicsClientId=cid)
    p.loadURDF(os.path.join(assest, "stair.urdf"),  [0, 2, 0],
               p.getQuaternionFromEuler([0, 0, -3.1416]),
               useFixedBase=True, physicsClientId=cid)

    print("[view] Biped + stair spawned. Press Ctrl+C to quit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        pass
    p.disconnect(cid)


# ── Training ──────────────────────────────────────────────────────────────────
def train(timesteps: int, render: bool = False):
    """
    Trains a SAC agent on the 1 m platform jump task and saves the model.

    Steps
    -----
    1. Create training and evaluation environments (wrapped in Monitor).
    2. Instantiate SAC with SAC_CONFIG from utils.py.
    3. Set up EvalCallback (saves best model) and RewardPlotCallback.
    4. Call model.learn() and handle KeyboardInterrupt for crash-saves.
    5. Save the final model and plot the reward curve.
    """
    # ============================================================
    # TODO: Create output directories for models and logs.
    #       Suggested layout:
    #           models/               ← saved model files
    #           logs/sac_goal/        ← TensorBoard logs
    #           logs/sac_eval/        ← EvalCallback logs
    # ============================================================

    # ============================================================
    # TODO: Instantiate the training environment wrapped with Monitor.
    #       Save monitor logs to ./logs/sac_monitor.csv.
    #       The eval environment should always be headless (render=False).
    # ============================================================

    # ============================================================
    # TODO: Instantiate the SAC model using SAC_CONFIG.
    #       Pass env and tensorboard_log="logs/sac_goal/".
    # ============================================================

    # ============================================================
    # TODO: Create a RewardPlotCallback and an EvalCallback.
    #       EvalCallback should:
    #           - save the best model to models/sac_best/
    #           - evaluate every EVAL_FREQ steps
    #           - use deterministic=True
    # ============================================================

    # ============================================================
    # TODO: Call model.learn() with total_timesteps and both callbacks.
    #       Wrap in try/except KeyboardInterrupt to save a crash checkpoint
    #       at models/sac_biped_crashsave.zip.
    # ============================================================

    # ============================================================
    # TODO: Save the final model to models/sac_biped_goal.zip.
    # TODO: Call reward_cb.plot_rewards() to save reward_curve_sac.png.
    # TODO: Close both environments.
    # ============================================================
    pass


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(model_path: str, episodes: int, render: bool):
    """
    Loads a trained SAC model and evaluates it for a given number of episodes.

    Metrics reported per episode
    ----------------------------
    - Steps taken
    - Total reward
    - Energy consumed  (sum of |torque × velocity| × dt)
    - Distance travelled (Euclidean, spawn → landing)

    Summary metrics printed at the end
    -----------------------------------
    - Average reward
    - Fall rate  (%)
    - Average distance (m)
    - Average energy (J)
    - Cost of Transport (CoT) = Energy / (mass × g × distance)
    """
    DT = 1.0 / 50.0   # simulation timestep (must match utils.py)

    # ============================================================
    # TODO: Create a BipedJumpEnv (with render flag) and load the model
    #       using SAC.load(model_path, env=env).
    # ============================================================

    # ============================================================
    # TODO: Get the list of actuated joint indices via env.get_joint_indices().
    #       Initialise accumulators:
    #           total_energy, total_distance, total_reward, fall_count = 0, 0, 0, 0
    # ============================================================

    # ============================================================
    # TODO: Loop over `episodes`. For each episode
    # ============================================================

    # ============================================================
    # TODO: After all episodes, compute and print the summary:
    #   avg_reward   = total_reward / n
    #   fall_rate    = 100 * fall_count / n
    #   avg_distance = total_distance / n
    #   avg_energy   = total_energy / n
    #   CoT          = total_energy / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)
    # ============================================================

    # ============================================================
    # TODO: Close the environment.
    # ============================================================
    pass


# ── CLI entry-point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 — Biped 1 m Platform Jump (SAC)"
    )
    parser.add_argument("--mode",       choices=["view", "train", "test"], required=True,
                        help="view: preview env  |  train: train SAC  |  test: evaluate")
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Override TOTAL_TIMESTEPS from utils.py")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model (.zip) for --mode test")
    parser.add_argument("--episodes",   type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes (default: {EVAL_EPISODES})")
    parser.add_argument("--render",     action="store_true",
                        help="Enable PyBullet GUI")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "view":
        view()

    else:
        # ============================================================
        # TODO: Route to the correct function based on args.mode:
        #
        #   "train" → resolve timesteps:
        #               ts = args.timesteps if args.timesteps else TOTAL_TIMESTEPS
        #             then call train(ts, args.render)
        #
        #   "test"  → if args.model_path is None, use the default path:
        #               "models/sac_best/best_model"
        #             then call test(args.model_path, args.episodes, args.render)
        # ============================================================
        pass


if __name__ == "__main__":
    main()
