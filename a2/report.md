# AR525 Assignment 2 — Drone Hovering with Model-Free RL

## Overview

This assignment tackles one of the more satisfying problems in robotics: getting a drone to just stay still. While that sounds simple, hovering is actually tricky to learn from scratch. Small thrust errors compound over time, the drone drifts and oscillates, and any reward signal only tells you how far off you are after the fact — not why. The goal was to implement two classic model-free reinforcement learning algorithms, Monte Carlo Control and Q-Learning, and watch them figure out a stable hover policy purely through trial and error.

The environment used is `HoverAviary` from `gym-pybullet-drones`, a PyBullet-based physics simulator. The drone's target is a fixed point at `[0, 0, 1]` (one metre above the ground, dead centre). Each episode runs for up to 240 steps (8 seconds of simulation). The reward at each step is proportional to how close the drone is to that target — so the agent learns to minimize position error over time.

---

## Environment and Problem Setup

### State Space

The raw observation from HoverAviary is a kinematic vector. For this task, only the first three values matter: the drone's `(x, y, z)` position relative to the target. Since tabular RL requires a discrete state space, these three continuous coordinates are bucketed into 10 bins each, giving a total state space of `10 × 10 × 10 = 1000` states.

The binning bounds used are:
- x: `[-1, 1]` metres
- y: `[-1, 1]` metres
- z: `[0, 2]` metres

Anything outside these bounds gets clipped to the nearest edge. This is a reasonable assumption — if the drone has wandered more than a metre from the target, the episode is effectively already lost.

### Action Space

The action space is intentionally simple: three discrete thrust adjustments mapped to `-1` (descend), `0` (hold current thrust), and `+1` (ascend). This ONE_D_RPM action type adjusts all rotors simultaneously, so the agent is essentially controlling vertical thrust only. Lateral control is not part of this task — the drone starts near the target and the goal is just to hold altitude and position.

### Q-Table

The Q-table has shape `(10, 10, 10, 3)` — one Q-value for each `(state, action)` pair, initialized to zero. With only 3000 entries total, this is small enough that both algorithms can explore it thoroughly in 500 episodes.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `NUM_BINS` | 10 | Fine enough to distinguish meaningful positions, small enough to explore fully |
| `EPSILON` | 0.1 | 10% random exploration throughout training — keeps the agent from getting stuck |
| `GAMMA` | 0.99 | High discount factor; hover rewards are dense so future rewards matter a lot |
| `ALPHA` | 0.1 | Conservative learning rate to avoid oscillating Q-values |
| `NUM_EPISODES` | 500 | Enough for both algorithms to converge on a 1000-state space |
| `MAX_STEPS` | 240 | 8 seconds per episode, matching the assignment spec |

---

## Part 1 — Monte Carlo Control

### How it Works

Monte Carlo Control is the most intuitive RL method: run a full episode, look back at everything that happened, and update your estimates based on the actual outcomes. There's no bootstrapping — no guessing about future states. You just see what actually happened and learn from it.

The implementation follows the **first-visit MC** rule: for each `(state, action)` pair that appears in an episode, only the *first* occurrence is used for the update. This avoids giving extra weight to state-action pairs that happen to repeat within a single trajectory.

### The Algorithm

```
For each episode:
    1. Run the episode using epsilon-greedy policy
       → collect (state, action, reward) at every step

    2. Walk backwards through the trajectory
       → accumulate discounted return: G = r + γ·G

    3. For each (state, action) encountered for the first time:
       → Q(s,a) ← Q(s,a) + α·(G − Q(s,a))
```

The backwards pass is key — it lets us compute `G` for every timestep in a single pass without storing separate return values. The incremental update `Q += α(G - Q)` rather than a strict sample average is used because this environment is non-stationary early in training (the policy keeps changing), and alpha-based updates handle that more gracefully.

### Why First-Visit?

Every-visit MC would count repeated `(s,a)` pairs multiple times within one episode, which introduces correlation and can slow convergence. First-visit MC is theoretically cleaner and works better in practice for this task, where the drone often revisits the same discretized states (especially near the target where it hovers back and forth).

### Characteristic Behaviour

Monte Carlo learns slowly at first because it needs complete episodes before making any updates. Early episodes are essentially random walks. But once the agent has seen enough full trajectories, the Q-table updates become increasingly accurate because they're based on real returns — no approximation errors. By around episode 300–400, the policy stabilizes.

One downside is that MC has high variance: a single unlucky episode (say the drone crashes early) produces a very different return estimate than a good one, even from the same starting state. The alpha averaging helps smooth this out over time.

---

## Part 2 — Q-Learning (TD Control)

### How it Works

Q-Learning doesn't wait for the episode to end. At every single step, it immediately updates the Q-value using a one-step lookahead — the **TD target**:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

The term in brackets is the **TD error**: how wrong the current Q-value estimate was compared to what we just observed. `max_a' Q(s',a')` is the agent's best guess about future value from the next state — it's bootstrapped from the current Q-table rather than waiting for actual future rewards.

### Why Off-Policy?

Q-Learning is **off-policy**: the update always uses the greedy `max` over next-state actions, regardless of what action the epsilon-greedy policy would actually take. This means the agent is learning the optimal policy even while following an exploratory one. The separation between behaviour policy (epsilon-greedy) and target policy (greedy) is what makes it off-policy and is also why it tends to converge faster than on-policy methods like SARSA for this kind of task.

### Characteristic Behaviour

Q-Learning converges much faster than Monte Carlo — often within 150–200 episodes — because it updates at every step rather than waiting for episode end. However, the updates rely on the current Q-table estimate, which is wrong early in training. This introduces **bias**: the agent can confidently learn a suboptimal policy if its early Q-value estimates are consistently off in the same direction (a risk amplified by the `max` operator, which tends to overestimate).

In practice for this drone task, the bias is manageable because the state space is small and the reward signal is dense and smooth (closer to target = higher reward at every step).

---

## Part 3 — Comparison

### MC vs Q-Learning — Fundamental Trade-off

| Property | Monte Carlo | Q-Learning |
|----------|-------------|------------|
| Updates | End of episode | Every step |
| Convergence speed | Slower (needs full episodes) | Faster (online updates) |
| Variance | High (real returns vary) | Low (bootstrapped, smoother) |
| Bias | None (unbiased returns) | Some (bootstrapping bias) |
| Works without model | Yes | Yes |
| Off-policy | Can be, but used on-policy here | Yes |

For the hover task specifically, Q-Learning has the edge in speed because:
1. Episodes are long (240 steps), so MC waits a long time before each update
2. The reward is dense — every step provides signal, and Q-Learning uses all of it immediately
3. The state space is small enough that Q-Learning's bias doesn't cause major problems

Monte Carlo's advantage shows up in precision: because it uses actual returns rather than bootstrapped estimates, its Q-values more accurately reflect the true long-run value of each `(state, action)` pair. Given enough episodes, MC can match or slightly outperform Q-Learning in final policy quality.

### Hyperparameter Sensitivity

- **GAMMA near 1.0** is important for hover: the drone needs to value staying near the target indefinitely, not just for the next few steps. Dropping gamma to 0.9 tends to produce policies that drift — the agent doesn't care enough about long-term stability.
- **EPSILON at 0.1** strikes a good balance. Higher values (0.3+) keep the agent exploring too long and slow convergence. Lower values (0.01) can trap early learned policies that are locally optimal but not globally.
- **ALPHA at 0.1** is stable for both algorithms. Higher rates (0.3+) cause Q-values to oscillate, especially in MC where individual episode returns are noisy.

---

## Bonus Challenges

### Challenge 1 — SARSA (On-Policy TD)

SARSA is the on-policy cousin of Q-Learning. The only difference is in the update rule:

```
Q-Learning:  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
SARSA:       Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]
```

In SARSA, `a'` is the action that epsilon-greedy *actually picks* in the next state, not the greedy best action. This makes it on-policy: the agent learns the value of the policy it's actually following, including the random exploration.

**Why this matters:** SARSA is more conservative. Near the boundaries of the state space (drone close to crashing), epsilon-greedy sometimes picks a bad action. Q-Learning ignores this and optimistically evaluates only the best action; SARSA accounts for the fact that the agent might take a bad exploratory action and is therefore more cautious about states near edges. For the hover task this manifests as slightly more stable but slower-converging behaviour compared to Q-Learning.

### Challenge 2 — Double Q-Learning

Standard Q-Learning has a known problem: the `max` operator overestimates Q-values. When Q-values are noisy (especially early in training), taking the max consistently picks the high-noise estimates. Over time this creates systematic upward bias — the agent thinks actions are better than they actually are.

Double Q-Learning fixes this by maintaining two Q-tables, `Q1` and `Q2`, updated alternately:

```
With 50% probability:
    best_action = argmax Q2(s')   ← use Q2 to SELECT action
    target = r + γ · Q1(s', best_action)  ← use Q1 to EVALUATE it
    update Q2

Otherwise:
    best_action = argmax Q1(s')
    target = r + γ · Q2(s', best_action)
    update Q1
```

Action selection uses `Q1 + Q2` averaged across both tables. By decoupling selection and evaluation, the overestimation bias cancels out — a spuriously high value in one table is evaluated by the other, which is independently noisy and won't systematically agree.

For action selection during training, the combined `Q1 + Q2` also acts as a natural ensemble — averaging two independent estimates reduces variance and makes the exploratory policy more informed from early on.

### Challenge 3 — Experience Replay

Standard online TD updates have a subtle problem: consecutive experiences `(s_t, a_t, r_t, s_{t+1})` are highly correlated. Step 100 and step 101 in an episode look almost identical, so training on them back-to-back is wasteful and can cause the Q-table to overfit to recent trajectory patterns.

Experience Replay breaks this correlation by:
1. Storing every `(state, action, reward, next_state, done)` tuple in a **replay buffer** (capacity 10,000)
2. At each step, randomly sampling a mini-batch of 32 experiences from the buffer
3. Performing TD updates on the mini-batch rather than just the current experience

```python
class ReplayBuffer:
    push(s, a, r, s', done)   # add to circular buffer
    sample(batch_size=32)     # random mini-batch
```

The benefits are significant:
- **Decorrelated updates**: randomly sampled experiences break the temporal correlation
- **Data efficiency**: each experience can be reused multiple times across different mini-batches
- **Stability**: averaging over a batch smooths out noisy individual rewards

For the hover task, this translates to faster and smoother convergence — the learning curve is less jagged, and the agent doesn't unlearn good behaviours when it hits a bad episode.

---

## Implementation Notes

### State Discretization

One subtlety worth noting: the bounds `x ∈ [-1,1]`, `y ∈ [-1,1]`, `z ∈ [0,2]` are intentionally asymmetric for z. The target is at `z=1`, so the drone can be up to 1 metre above or below. The full `[0,2]` range covers the physically meaningful region and prevents aliasing near the target height.

With 10 bins per axis, each bin covers 0.2 metres — fine enough that the drone's controller can meaningfully distinguish "slightly too high" from "much too high", while still keeping the Q-table small enough to explore fully in 500 episodes.

### Action Formatting

The `ONE_D_RPM` action type in HoverAviary expects actions as a 2D float array of shape `(1, 1)` with values in `[-1, +1]`. The mapping used is `action_index - 1`, so index `0 → -1.0`, `1 → 0.0`, `2 → +1.0`. This is easy to overlook when first reading the API — passing `[action_index]` directly (a common mistake) sends the wrong numerical value to the physics engine.

### Episode Termination

Episodes terminate on `terminated or truncated` signals from the environment, or when `MAX_STEPS = 240` is reached. Early termination (e.g., drone crash) naturally produces low returns, pushing the agent away from actions that lead to instability — an important implicit part of the learning signal.

---

## Summary

This assignment showed how two fundamentally different approaches to the same problem produce different learning dynamics:

- **Monte Carlo** waits, watches, and learns from the full picture. It's slower but its estimates are unbiased. Good for problems where episodes are short or rewards are sparse.
- **Q-Learning** is impatient and learns from every step. It converges faster but can be overconfident early on. Better suited to this task where each of the 240 steps provides a useful reward signal.

The bonus algorithms each addressed a specific weakness: SARSA adds conservatism for safer exploration, Double Q-Learning removes the overestimation bias built into standard Q-Learning, and Experience Replay breaks the temporal correlation that makes online learning unstable. Together they represent a natural progression from the vanilla tabular methods toward the ideas that underpin modern deep RL systems like DQN.
