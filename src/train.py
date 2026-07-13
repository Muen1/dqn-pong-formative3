# src/train.py
import argparse
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import csv
import os

gym.register_envs(ale_py)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="PongNoFrameskip-v4")
    p.add_argument("--policy", type=str, choices=["MlpPolicy", "CnnPolicy"], default="CnnPolicy")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_frac", type=float, default=0.1)  # fraction of training
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--run_name", type=str, default="run1")
    p.add_argument("--save_path", type=str, default="models/dqn_model.zip")
    return p.parse_args()

class RewardLogger(BaseCallback):
    """Logs episode reward + length to a CSV for the writeup/table."""
    def __init__(self, run_name):
        super().__init__()
        self.run_name = run_name
        os.makedirs("logs", exist_ok=True)
        self.path = f"logs/{run_name}_episodes.csv"
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "reward", "length"])
        self.ep_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_count += 1
                with open(self.path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self.ep_count, info["episode"]["r"], info["episode"]["l"]]
                    )
        return True

def main():
    args = parse_args()

    env = make_atari_env(args.env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        args.policy,
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_decay_frac,
        buffer_size=100_000 if args.policy == "CnnPolicy" else 50_000,
        learning_starts=10_000,
        tensorboard_log="./tb_logs/",
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=RewardLogger(args.run_name),
        tb_log_name=args.run_name,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path.replace(".zip", "") if args.run_name == "final" else f"models/{args.run_name}.zip")
    print(f"Saved model for run: {args.run_name}")

if __name__ == "__main__":
    main()
