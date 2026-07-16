from pathlib import Path
import csv
from statistics import mean, pstdev

import ale_py
import gymnasium as gym
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack


ENV_ID = "PongNoFrameskip-v4"
EVAL_EPISODES = 5
OUTPUT = Path("experiments/member_c_results.csv")

EXPERIMENTS = [
    ("member_c_01_baseline", 0.0001, 0.99, 32, 1.00, 0.05, 0.10),
    ("member_c_02_fast_decay", 0.0001, 0.99, 32, 1.00, 0.05, 0.02),
    ("member_c_03_slow_decay", 0.0001, 0.99, 32, 1.00, 0.05, 0.50),
    ("member_c_04_high_final_epsilon", 0.0001, 0.99, 32, 1.00, 0.20, 0.30),
    ("member_c_05_low_start_epsilon", 0.0001, 0.99, 32, 0.50, 0.05, 0.10),
    ("member_c_06_high_lr_low_gamma", 0.0005, 0.95, 32, 1.00, 0.05, 0.10),
    ("member_c_07_low_lr_high_gamma", 0.00005, 0.995, 64, 1.00, 0.05, 0.20),
    ("member_c_08_aggressive_edge", 0.001, 0.999, 32, 1.00, 0.01, 0.05),
    ("member_c_09_large_batch", 0.0001, 0.99, 128, 1.00, 0.05, 0.20),
    ("member_c_10_small_batch", 0.00025, 0.97, 16, 1.00, 0.10, 0.30),
]


def create_environment(seed):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    return VecFrameStack(env, n_stack=4)


def training_statistics(log_path):
    data = pd.read_csv(log_path)

    rewards = pd.to_numeric(data["reward"], errors="coerce").dropna()
    lengths = pd.to_numeric(data["length"], errors="coerce").dropna()

    window = min(20, len(rewards))

    return {
        "training_episodes": len(data),
        "final_training_reward": round(float(rewards.iloc[-window:].mean()), 3),
        "best_training_reward": round(float(rewards.max()), 3),
        "mean_training_length": round(float(lengths.mean()), 3),
    }


def main():
    gym.register_envs(ale_py)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        (
            run_name,
            learning_rate,
            gamma,
            batch_size,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
        ) = experiment

        model_path = Path("models") / f"{run_name}.zip"
        log_path = Path("logs") / f"{run_name}_episodes.csv"

        if not model_path.exists():
            print(f"Missing model: {model_path}")
            continue

        if not log_path.exists():
            print(f"Missing log: {log_path}")
            continue

        print(f"\nEvaluating {run_name}...")

        env = create_environment(seed=1000 + index)

        try:
            model = DQN.load(model_path, env=env, device="cpu")

            rewards, lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=EVAL_EPISODES,
                deterministic=True,
                return_episode_rewards=True,
                warn=False,
            )
        finally:
            env.close()

        train_stats = training_statistics(log_path)

        row = {
            "experiment": f"C{index:02d}",
            "run_name": run_name,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "training_timesteps": 50000,
            **train_stats,
            "evaluation_episodes": len(rewards),
            "evaluation_mean_reward": round(mean(rewards), 3),
            "evaluation_reward_std": round(pstdev(rewards), 3),
            "evaluation_best_reward": round(max(rewards), 3),
            "evaluation_worst_reward": round(min(rewards), 3),
            "evaluation_mean_length": round(mean(lengths), 3),
        }

        results.append(row)

        print(
            f"Mean reward: {row['evaluation_mean_reward']} | "
            f"Standard deviation: {row['evaluation_reward_std']}"
        )

    if not results:
        raise RuntimeError("No completed experiments were found.")

    results.sort(
        key=lambda item: item["evaluation_mean_reward"],
        reverse=True,
    )

    fieldnames = list(results[0].keys())

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT}")
    print(f"Best model: {results[0]['run_name']}")
    print(f"Best mean reward: {results[0]['evaluation_mean_reward']}")


if __name__ == "__main__":
    main()
