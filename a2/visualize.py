"""
Visualization script for trained MC and Q-Learning policies.

Usage:
    python visualize.py --method mc      # Visualize Monte Carlo
    python visualize.py --method td      # Visualize Q-Learning
    python visualize.py --method both    # Visualize both (default)
"""

import argparse
import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from user_code import (
    run_monte_carlo, run_q_learning,
    discretize_state, extract_position, format_action,
    NUM_EPISODES, EPSILON, GAMMA, ALPHA, MAX_STEPS
)


def run_gui_until_closed(env, q_table, label="Policy"):
    """
    Run episodes in a loop until the PyBullet GUI window is closed manually.
    Detects window closure via pybullet connection error and exits cleanly.
    """
    import pybullet as p

    episode = 0
    while True:
        episode += 1
        state, _ = env.reset()
        state = discretize_state(extract_position(state))
        total_reward = 0.0
        print(f"\n[{label}] Episode {episode}...")

        for step in range(MAX_STEPS):
            # Check if GUI window was closed
            try:
                p.getConnectionInfo(env.getPyBulletClient())
            except Exception:
                print(f"\n[{label}] GUI window closed. Exiting.")
                return

            action = np.argmax(q_table[state])
            try:
                next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            except Exception:
                print(f"\n[{label}] GUI window closed. Exiting.")
                return

            next_state = discretize_state(extract_position(next_obs))
            pos = extract_position(next_obs)
            total_reward += reward

            if step % 30 == 0:
                print(f"  Step {step:3d} | pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) | reward={reward:.3f}")

            state = next_state
            if terminated or truncated:
                break

        print(f"[{label}] Episode {episode} total reward: {total_reward:.2f}")


def visualize(method="both"):
    # ---- Monte Carlo ----
    if method in ("mc", "both"):
        print("=" * 60)
        print("Training Monte Carlo (headless)...")
        print("=" * 60)
        env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
        q_table_mc, rewards_mc = run_monte_carlo(env, num_episodes=NUM_EPISODES,
                                                  epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA)
        env.close()

        print(f"\nTraining complete. Avg reward (last 50): {np.mean(rewards_mc[-50:]):.2f}")
        input("\nPress Enter to open PyBullet GUI for Monte Carlo visualization...")

        env_gui = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=True)
        print("Close the PyBullet window to stop.")
        run_gui_until_closed(env_gui, q_table_mc, label="MC")
        try:
            env_gui.close()
        except Exception:
            pass

    # ---- Q-Learning ----
    if method in ("td", "both"):
        print("\n" + "=" * 60)
        print("Training Q-Learning (headless)...")
        print("=" * 60)
        env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
        q_table_td, rewards_td = run_q_learning(env, num_episodes=NUM_EPISODES,
                                                  epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA)
        env.close()

        print(f"\nTraining complete. Avg reward (last 50): {np.mean(rewards_td[-50:]):.2f}")
        input("\nPress Enter to open PyBullet GUI for Q-Learning visualization...")

        env_gui = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=True)
        print("Close the PyBullet window to stop.")
        run_gui_until_closed(env_gui, q_table_td, label="Q-Learning")
        try:
            env_gui.close()
        except Exception:
            pass

    print("\nVisualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mc", "td", "both"], default="both",
                        help="Which algorithm to visualize")
    args = parser.parse_args()
    visualize(args.method)
