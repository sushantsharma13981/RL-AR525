"""
==========================================================================
                        UTILS.PY - FINAL IMPLEMENTATION
Dynamic Programming for Grid Navigation (UR5 Assignment)
==========================================================================
"""

import numpy as np


# ==========================================================================
#                           GRID ENVIRONMENT
# ==========================================================================

class GridEnv:

    def __init__(self, rows=5, cols=6, start=0, goal=29, obstacles=None,
                 reward_step=-1.0, reward_goal=10.0, reward_obstacle=-50.0):

        if obstacles is None:
            obstacles = []

        self.rows = rows
        self.cols = cols
        self.nS = rows * cols          # Number of states
        self.nA = 4                    # Actions: LEFT, DOWN, RIGHT, UP
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_obstacle = reward_obstacle

        self.action_names = {
            0: 'LEFT',
            1: 'DOWN',
            2: 'RIGHT',
            3: 'UP'
        }

        self.P = self._build_dynamics()

    # ----------------------------------------------------------------------

    def _state_to_pos(self, state):
        return state // self.cols, state % self.cols

    def _pos_to_state(self, row, col):
        return row * self.cols + col

    def _is_valid_pos(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ----------------------------------------------------------------------

    def _get_next_state(self, state, action):

        row, col = self._state_to_pos(state)

        if action == 0:      # LEFT
            col -= 1
        elif action == 1:    # DOWN
            row += 1
        elif action == 2:    # RIGHT
            col += 1
        elif action == 3:    # UP
            row -= 1

        if not self._is_valid_pos(row, col):
            return state

        return self._pos_to_state(row, col)

    # ----------------------------------------------------------------------

    def _build_dynamics(self):

        P = {}

        for state in range(self.nS):
            P[state] = {}

            # If obstacle → terminal state
            if state in self.obstacles:
                for action in range(self.nA):
                    P[state][action] = [(1.0, state, self.reward_obstacle, True)]
                continue

            for action in range(self.nA):

                next_state = self._get_next_state(state, action)

                # Goal
                if next_state == self.goal:
                    reward = self.reward_goal
                    done = True

                # Obstacle collision
                elif next_state in self.obstacles:
                    reward = self.reward_obstacle
                    done = True

                # Normal step
                else:
                    reward = self.reward_step
                    done = False

                P[state][action] = [(1.0, next_state, reward, done)]

        return P

    # ----------------------------------------------------------------------

    def get_optimal_path(self, policy):

        path = [self.start]
        curr = self.start
        visited = {curr}

        while curr != self.goal:

            action = np.argmax(policy[curr])
            next_state = self.P[curr][action][0][1]

            if next_state in visited:
                break

            path.append(next_state)
            visited.add(next_state)
            curr = next_state

        return path


# ==========================================================================
#                DYNAMIC PROGRAMMING ALGORITHMS
# ==========================================================================

# --------------------------------------------------------------------------
# PART 1 — POLICY EVALUATION
# --------------------------------------------------------------------------

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8,
                      return_iterations=False):

    V = np.zeros(env.nS)
    sweeps = 0

    while True:
        delta = 0

        for s in range(env.nS):

            v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            delta = max(delta, abs(v - V[s]))
            V[s] = v

        sweeps += 1
        if delta < theta:
            break

    if return_iterations:
        return V, sweeps

    return V


# --------------------------------------------------------------------------
# PART 2 — Q(s,a) FROM V(s)
# --------------------------------------------------------------------------

def q_from_v(env, V, s, gamma=0.99):

    q = np.zeros(env.nA)

    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (
                reward + gamma * V[next_state] * (not done)
            )

    return q


# --------------------------------------------------------------------------
# PART 3 — POLICY IMPROVEMENT
# --------------------------------------------------------------------------

def policy_improvement(env, V, gamma=0.99):

    policy = np.zeros((env.nS, env.nA))

    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        best_action = np.argmax(q)
        policy[s][best_action] = 1.0

    return policy


# --------------------------------------------------------------------------
# PART 4 — POLICY ITERATION
# --------------------------------------------------------------------------

def policy_iteration(env, gamma=0.99, theta=1e-8, return_metadata=False):

    policy = np.ones((env.nS, env.nA)) / env.nA
    iteration_count = 0
    evaluation_sweeps = []

    while True:

        iteration_count += 1
        V, sweeps = policy_evaluation(env, policy, gamma, theta,
                                      return_iterations=True)
        evaluation_sweeps.append(sweeps)
        new_policy = policy_improvement(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    if not return_metadata:
        return policy, V

    metadata = {
        "policy_iterations": iteration_count,
        "policy_changes": max(iteration_count - 1, 0),
        "evaluation_sweeps": evaluation_sweeps,
        "total_evaluation_sweeps": sum(evaluation_sweeps)
    }

    return policy, V, metadata


# --------------------------------------------------------------------------
# PART 5 — VALUE ITERATION
# --------------------------------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8, return_metadata=False):

    V = np.zeros(env.nS)
    iteration_count = 0
    delta_history = []

    while True:
        delta = 0

        for s in range(env.nS):

            q = q_from_v(env, V, s, gamma)
            best_value = np.max(q)

            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value

        iteration_count += 1
        delta_history.append(delta)
        if delta < theta:
            break

    policy = policy_improvement(env, V, gamma)

    if not return_metadata:
        return policy, V

    metadata = {
        "iterations": iteration_count,
        "delta_history": delta_history,
        "final_delta": delta_history[-1] if delta_history else 0.0
    }

    return policy, V, metadata

