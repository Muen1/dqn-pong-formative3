"""
Summarize expB_1 .. expB_10 episode logs into one comparison table.

Usage (run from your repo root, e.g. dqn-pong-formative3):
    python summarize_experiments.py

Reads: logs/expB_1_episodes.csv ... logs/expB_10_episodes.csv
   (columns: episode, reward, length)
Writes: experiments/batch_eps_experiments.csv
Prints: a readable table sorted by best final performance
"""

import csv
import os

RUNS = [
    {"name": "expB_1",  "batch_size": 32,  "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.1,  "note": "baseline"},
    {"name": "expB_2",  "batch_size": 16,  "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.1,  "note": "small batch"},
    {"name": "expB_3",  "batch_size": 64,  "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.1,  "note": "large batch"},
    {"name": "expB_4",  "batch_size": 128, "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.1,  "note": "xl batch"},
    {"name": "expB_5",  "batch_size": 32,  "eps_start": 1.0, "eps_end": 0.01, "eps_decay_frac": 0.1,  "note": "low eps floor"},
    {"name": "expB_6",  "batch_size": 32,  "eps_start": 1.0, "eps_end": 0.2,  "eps_decay_frac": 0.1,  "note": "high eps floor"},
    {"name": "expB_7",  "batch_size": 32,  "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.02, "note": "fast decay"},
    {"name": "expB_8",  "batch_size": 32,  "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.3,  "note": "slow decay"},
    {"name": "expB_9",  "batch_size": 32,  "eps_start": 0.5, "eps_end": 0.05, "eps_decay_frac": 0.1,  "note": "low eps start"},
    {"name": "expB_10", "batch_size": 64,  "eps_start": 1.0, "eps_end": 0.02, "eps_decay_frac": 0.05, "note": "combo: large batch + aggressive decay"},
]

LOG_DIR = "logs"
OUT_DIR = "experiments"
OUT_PATH = os.path.join(OUT_DIR, "batch_eps_experiments.csv")
N_LAST = 20  # how many final episodes to average over


def read_episode_rewards(run_name):
    path = os.path.join(LOG_DIR, f"{run_name}_episodes.csv")
    if not os.path.exists(path):
        return None
    rewards = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rewards.append(float(row["reward"]))
            except (KeyError, ValueError):
                pass
    return rewards


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for run in RUNS:
        rewards = read_episode_rewards(run["name"])
        if rewards is None:
            print(f"WARNING: missing log for {run['name']}, skipping")
            continue

        total_episodes = len(rewards)
        first10_mean = mean(rewards[:10])
        last_n = rewards[-N_LAST:] if len(rewards) >= N_LAST else rewards
        last_n_mean = mean(last_n)
        best_single_ep = max(rewards) if rewards else float("nan")
        improvement = last_n_mean - first10_mean

        results.append({
            **run,
            "total_episodes": total_episodes,
            "first10_mean_reward": round(first10_mean, 2),
            f"last{N_LAST}_mean_reward": round(last_n_mean, 2),
            "best_single_episode": best_single_ep,
            "improvement": round(improvement, 2),
        })

    # sort best (highest last-N mean reward) first
    results.sort(key=lambda r: r[f"last{N_LAST}_mean_reward"], reverse=True)

    # write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(OUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved summary to {OUT_PATH}\n")

    # print readable table
    header = f"{'run':<10}{'batch':<7}{'eps_start':<10}{'eps_end':<9}{'decay':<8}{'episodes':<10}{'first10':<10}{'last'+str(N_LAST):<10}{'best_ep':<9}{'improve':<9}note"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<10}{r['batch_size']:<7}{r['eps_start']:<10}{r['eps_end']:<9}"
            f"{r['eps_decay_frac']:<8}{r['total_episodes']:<10}{r['first10_mean_reward']:<10}"
            f"{r[f'last{N_LAST}_mean_reward']:<10}{r['best_single_episode']:<9}{r['improvement']:<9}{r['note']}"
        )

    if results:
        best = results[0]
        print(f"\nBest performer: {best['name']} ({best['note']}) "
              f"-> last{N_LAST} mean reward = {best[f'last{N_LAST}_mean_reward']}")


if __name__ == "__main__":
    main()
