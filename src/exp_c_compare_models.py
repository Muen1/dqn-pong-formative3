from statistics import mean, pstdev

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack


MODELS = [
    "member_c_01_baseline_200k",
    "member_c_07_200k",
]

gym.register_envs(ale_py)

results = []

for index, name in enumerate(MODELS):
    print(f"\nEvaluating {name}...")

    env = make_atari_env(
        "PongNoFrameskip-v4",
        n_envs=1,
        seed=3000 + index,
    )
    env = VecFrameStack(env, n_stack=4)

    try:
        model = DQN.load(
            f"models/{name}.zip",
            env=env,
            device="cpu",
        )

        rewards, lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
        )
    finally:
        env.close()

    result = {
        "name": name,
        "mean_reward": mean(rewards),
        "reward_std": pstdev(rewards),
        "best_reward": max(rewards),
        "worst_reward": min(rewards),
        "mean_length": mean(lengths),
    }

    results.append(result)

    print("Rewards:", rewards)
    print("Mean reward:", round(result["mean_reward"], 3))
    print("Reward standard deviation:", round(result["reward_std"], 3))
    print("Best reward:", result["best_reward"])
    print("Worst reward:", result["worst_reward"])
    print("Mean episode length:", round(result["mean_length"], 3))

best = max(results, key=lambda item: item["mean_reward"])

print("\nBest 200k model:", best["name"])
print("Best mean reward:", round(best["mean_reward"], 3))
