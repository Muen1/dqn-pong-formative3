# Group Contributions — DQN Pong (Formative 3)

## Team
- Cynthia Mutie
- Patricia Mugabo
- Henriette Utatsineza

## Shared Setup (all members, done together)
- Selected Pong as the group's Atari environment
- Set up the shared GitHub repository structure (`src/`, `notebooks/`, `experiments/`, `logs/`, `models/`)
- Agreed on `requirements.txt` and resolved environment/dependency setup (gymnasium, stable-baselines3, ale-py)
- Agreed on a common CSV log format so all 30 individual experiments could be merged into one final table

## Cynthia Mutie — Training Pipeline & Policy Comparison
**Owns:** `src/train.py`

- Built and maintained the shared `train.py` training script used by all three members for every experiment, including:
  - CLI arguments for policy type, learning rate, gamma, batch size, and epsilon schedule
  - Environment wrapping (frame stacking, Atari preprocessing)
  - Model saving to `.zip` format
  - A custom `RewardLogger` callback that logs per-episode reward and episode length to CSV
  - Checkpointing every 10,000 steps so long training runs can resume after a disconnect
- Debugged and resolved several dependency version conflicts (gymnasium/ale-py/shimmy/stable-baselines3) so the group's shared environment installs cleanly
- Ran the MLPPolicy vs CnnPolicy comparison and documented the architectural tradeoff for Pong
- Ran 10 individual hyperparameter experiments varying learning rate and gamma (`notebooks/gamma_experiments.ipynb`)
- Documented findings in `experiments/lr_gamma_experiments.csv`
- Prepared to speak on: training architecture, MLP vs CNN policy decision, reward/episode-length trends

## Patricia Mugabo — Evaluation Pipeline & Gameplay Demonstration
**Owns:** `src/play.py`

- Built and maintained `play.py`, which loads the trained `dqn_model.zip` and runs the agent using a greedy Q-policy for evaluation
- Implemented environment rendering (`env.render()`) and multi-episode evaluation runs
- Recorded the gameplay video demonstrating the trained agent, included in the README
- Ran 10 individual hyperparameter experiments varying batch size and epsilon (start/end/decay) (`notebooks/[member_b_notebook].ipynb`)
- Documented findings in `experiments/batchsize_epsilon_experiments.csv`
- Prepared to speak on: agent evaluation results, exploration-exploitation tradeoffs, final model behavior

## Henriette Utatsineza — Documentation, Synthesis & Submission Logistics
**Owns:** `README.md`, final submission package

- Merged all three members' experiment tables (30 rows total) into `experiments/final_combined_table.csv`, with consolidated observations
- Wrote and maintained the group README, including the full hyperparameter table, embedded gameplay video, and discussion of tuning results
- Completed the required Google Sheet, exported as PDF, and added it to the repository
- Booked the Week 6 slot with the Coach
- Ran 10 individual hyperparameter experiments covering combined/edge-case configurations (`notebooks/[member_c_notebook].ipynb`)
- Documented findings in `experiments/combined_experiments.csv`
- Ran the final combined-hyperparameter training run producing `models/dqn_model.zip`
- Prepared to speak on: overall trends across all 30 experiments, final configuration and rationale

## Shared Preparation (all members)
- Each member independently prepared their own 2-minute presentation segment
- All members reviewed the full 30-row combined table together and prepared to answer Q&A on any part of it — not just their own section — since the Coach's questions are not limited to individual contributions
- Group dry-ran the gameplay clip and confirmed camera/presentation logistics ahead of the scheduled slot

## Summary
Work was split evenly across three roles (training pipeline, evaluation pipeline, documentation/synthesis), each contributing an equal share of the required 10 individual hyperparameter experiments (30 total), with visible, attributable commits from each member throughout the repository history.