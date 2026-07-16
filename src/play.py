# src/play.py
import argparse
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

gym.register_envs(ale_py)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="PongNoFrameskip-v4")
    p.add_argument("--model_path", type=str, default="models/dqn_model.zip")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--record", action="store_true", help="Save video instead of rendering live")
    p.add_argument("--video_folder", type=str, default="gameplay_videos")
    return p.parse_args()

def main():
    args = parse_args()

    if args.record:
        env = make_atari_env(args.env_id, n_envs=1, seed=0, env_kwargs={"render_mode": "rgb_array"})
        env = VecFrameStack(env, n_stack=4)
        os.makedirs(args.video_folder, exist_ok=True)
        env = VecVideoRecorder(
            env,
            args.video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=100_000,
            name_prefix="pong_agent",
        )
    else:
        env = make_atari_env(args.env_id, n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
        env = VecFrameStack(env, n_stack=4)

    model = DQN.load(args.model_path, env=env)

    obs = env.reset()
    episode_rewards = [0.0]
    episodes_done = 0

    while episodes_done < args.n_episodes:
        action, _states = model.predict(obs, deterministic=True)  # GreedyQPolicy equivalent
        obs, reward, done, info = env.step(action)
        episode_rewards[-1] += reward[0]

        if done[0]:
            episodes_done += 1
            print(f"Episode {episodes_done}: reward={episode_rewards[-1]}")
            episode_rewards.append(0.0)

    env.close()
    avg = sum(episode_rewards[:-1]) / args.n_episodes
    print(f"\nAvg reward over {args.n_episodes} episodes: {avg:.2f}")

if __name__ == "__main__":
    main()