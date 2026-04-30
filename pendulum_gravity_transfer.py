"""
Pendulum-v1 ; Gravity Transfer Experiment
==========================================
Train PPO on Earth gravity (g=9.81), then evaluate the SAME frozen policy
on Moon (g=1.62) and Jupiter (g=24.79) without retraining.

The pendulum ODE is:
    θ'' = (3g/2l)·sin(θ) + (3/ml²)·u

Changing g changes the ODE itself ; so we are testing whether a policy
learned for one differential equation generalises to another.

Hyperparameters sourced from RL Baselines3 Zoo:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
  → Pendulum-v1 block
"""

"""
Note:
This code only worked for me using python 3.12, any later versions broke.
Run with the following after installing python 3.12 if not working:
>>> py -3.12 pendulum_gravity_transfer.py
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Reproducibility
SEED = 42

# Gravity conditions
GRAVITIES = {
    "Moon":    1.62,
    "Earth":   9.81,   # trained here
    "Jupiter": 24.79,
}
TRAIN_GRAVITY = 9.81   # Earth

# Reward logger callback
class RewardLogger(BaseCallback):
    """Records mean episodic reward at each rollout end for plotting."""
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            self.rewards.append(mean_r)
        return True

# TRAIN on Earth gravity
print(f"\n{'='*55}")
print(f"  TRAINING on Earth gravity (g={TRAIN_GRAVITY})")
print(f"{'='*55}\n")

# Vectorised training env, Zoo uses n_envs=4
train_env = make_vec_env(
    "Pendulum-v1",
    n_envs=4,
    seed=SEED,
    env_kwargs={"g": TRAIN_GRAVITY},
)

# PPO with exact Zoo hyperparameters for Pendulum-v1
model = PPO(
    policy         = "MlpPolicy",
    env            = train_env,
    n_steps        = 1024,        # steps per env per update
    batch_size     = 256,
    n_epochs       = 10,
    gamma          = 0.9,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.0,
    learning_rate  = 1e-3,
    use_sde        = True,        # gSDE: smoother continuous-action exploration
    sde_sample_freq= 4,
    verbose        = 1,
    seed           = SEED,
)

reward_logger = RewardLogger()

# Zoo default: n_timesteps = 1e5
model.learn(total_timesteps=100_000, callback=reward_logger, progress_bar=True)
model.save("pendulum_earth_policy")
print("\nPolicy saved → pendulum_earth_policy.zip")

# EVALUATE frozen policy on all three gravities 
print(f"\n{'='*55}")
print("  EVALUATING frozen policy across gravity conditions")
print(f"{'='*55}\n")

results = {}
N_EVAL_EPISODES = 20

for label, g in GRAVITIES.items():
    eval_env = Monitor(gym.make("Pendulum-v1", g=g))
    mean_r, std_r = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )
    results[label] = (mean_r, std_r)
    print(f"  {label:8s} (g={g:5.2f})  →  mean reward = {mean_r:7.1f}  ±  {std_r:.1f}")
    eval_env.close()

# Learning curve (Earth training)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(reward_logger.rewards, color="royalblue", linewidth=1.5)
axes[0].axhline(-200, color="salmon", linestyle="--", label="Random policy baseline")
axes[0].set_xlabel("Rollout")
axes[0].set_ylabel("Mean Episode Reward")
axes[0].set_title("PPO Training Curve, Earth (g=9.81)")
axes[0].legend()

# Transfer bar chart
labels  = list(results.keys())
means   = [results[l][0] for l in labels]
stds    = [results[l][1] for l in labels]
colors  = ["#a0c4ff", "#4c72b0", "#d62728"]   # Moon, Earth, Jupiter

bars = axes[1].bar(labels, means, yerr=stds, color=colors,
                   capsize=6, edgecolor="white", linewidth=0.8)
axes[1].set_ylabel("Mean Episode Reward (20 episodes)")
axes[1].set_title("Policy Transfer: Earth Policy on Moon / Jupiter")
axes[1].axhline(-200, color="salmon", linestyle="--", label="Random baseline")
axes[1].legend()

# Annotate bars with values
for bar, mean in zip(bars, means):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 80,
        f"{mean:.0f}",
        ha="center", va="top", color="white", fontweight="bold"
    )

plt.suptitle("Pendulum-v1: Does an Earth-trained PPO policy generalise\n"
             "to different gravitational ODEs?", fontsize=12)
plt.tight_layout()
plt.savefig("gravity_transfer.png", dpi=150)
print("\nPlot saved → gravity_transfer.png")
plt.show()

# Print summary
print(f"\n{'='*55}")
print("  SUMMARY")
print(f"{'='*55}")
for label, (mean_r, std_r) in results.items():
    trained_here = " ← trained here" if GRAVITIES[label] == TRAIN_GRAVITY else ""
    print(f"  {label:8s} (g={GRAVITIES[label]:5.2f})  {mean_r:7.1f} ± {std_r:.1f}{trained_here}")
print()
print("  Interpretation:")
print("  • Moon:    weaker restoring force → easier swing-up, policy may over-actuate")
print("  • Earth:   trained here → best performance expected")
print("  • Jupiter: much stronger restoring force → policy likely struggles to swing up")

# ANIMATE
for label, g in GRAVITIES.items():
    print(f"\nRendering {label} (g={g})...")
    render_env = gym.make("Pendulum-v1", g=g, render_mode="human")
    obs, _ = render_env.reset(seed=SEED)
    
    total_reward = 0
    for _ in range(200):   # one episode = 200 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"  Episode reward: {total_reward:.1f}")
    render_env.close()
    input("  Press Enter to continue to next gravity...")