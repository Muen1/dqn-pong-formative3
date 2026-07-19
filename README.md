# DQN Pong - Deep Q-Learning Agent (Formative 3 Group 9)

A Deep Q-Network (DQN) agent trained with Stable Baselines3 and Gymnasium to play Atari Pong, including a full hyperparameter tuning study across three team members (30 experiments total).

## Team
- Cynthia Mutie — Training pipeline, MLP vs CNN comparison, lr/gamma sweep (Member A)
- Henriette Utatsineza — Evaluation pipeline, gameplay video, batch size/epsilon sweep (Member B)
- Patricia Mugabo — Documentation, submission logistics, combined-config sweep, final model (Member C)

See [Team Task Sheet](https://github.com/Muen1/dqn-pong-formative3/blob/main/submission/Team%20Task%20Sheet_%5BDeep%20Q%20Learning_Formative%203_C1%23_Group%209%23%5D.pdf) for a full breakdown of individual contributions.

## Environment
**Game:** Pong (`PongNoFrameskip-v4`, Gymnasium/ALE)

## Project Structure

```
dqn-pong-formative3/
├── src/
│   ├── train.py          # Training script (DQN, CLI-configurable hyperparameters)
│   └── play.py            # Loads trained model and runs greedy-policy evaluation
├── notebooks/              # Individual Colab notebooks per member
├── experiments/            # Per-member and combined hyperparameter tables
├── logs/                   # Per-episode reward/length logs (CSV)
├── models/
├──submission
   ├──Team Task Sheet      # Team contributions
   ├── report              # Report
├──.gitignore 
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate    # or venv/bin/activate on Mac/Linux
pip install -r requirements.txt
AutoROM --accept-license
```

## Usage

**Train:**
```bash
python src/train.py --policy CnnPolicy --lr 1e-4 --gamma 0.99 --run_name my_run --timesteps 500000
```

**Play / Evaluate:**
```bash
python src/play.py --model_path models/dqn_model.zip
```

## Policy Architecture: MLP vs CNN

We compared `MlpPolicy` and `CnnPolicy` under identical settings. `CnnPolicy` is the architecturally appropriate choice for Pong, since it processes stacked raw pixel frames through convolutional layers and can learn spatial patterns (ball position, paddle position) directly — while `MlpPolicy` flattens frames into a single vector and loses that spatial structure entirely. At the 150,000-timestep budget used for this comparison, neither policy showed strong learning yet, but `CnnPolicy` remains the correct architectural choice for image-based Atari environments and was used for all further experiments and the final model.

## Hyperparameter Tuning — Combined Results (30 Experiments)

Each member ran 10 independent experiments, holding some hyperparameters fixed to isolate the effect of others:
- **Cynthia Mutie:** varied `lr` and `gamma`, held `batch_size`/`epsilon` at defaults
- **Henriette Utatsineza:** varied `batch_size` and `epsilon` schedule, held `lr`/`gamma` at defaults
- **Patricia Mugabo:** varied combined/edge-case configurations across all six hyperparameters

| Run | lr | gamma | batch_size | eps_start | eps_end | eps_decay_frac | Noted Behavior |
|---|---|---|---|---|---|---|---|
| expA_1 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | No learning observed; reward stayed near -20.9 throughout, consistent with lr being too high for stable early Q-updates given limited timesteps |
| expA_2 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed flat around -20.9; no meaningful improvement within 150k timesteps |
| expA_3 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.9; default-like config, still within the random-play range at this timestep budget |
| expA_4 | 5e-05 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed around -20.5, marginally less negative than higher-lr runs but not a meaningful difference |
| expA_5 | 1e-05 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.8; lr likely too low to drive any updates within 150k steps |
| expA_6 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.9; higher gamma showed no measurable effect at this timestep budget |
| expA_7 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.8; lower gamma did not noticeably change short-term behavior |
| expA_8 | 0.0001 | 0.90 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.9; no clear effect from further lowering gamma |
| expA_9 | 0.0005 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.9; combining higher lr with lower gamma still showed no learning signal |
| expA_10 | 0.001 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.9; same divergence-free but non-learning pattern as expA_1 |
| expB_1 (baseline) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed flat near -20.9 over 500k timesteps; establishes the group's baseline |
| expB_2 (small batch) | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 0.1 | No improvement over baseline; smaller batch produced noisier gradient estimates that slowed learning |
| expB_3 (large batch) | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | Reached mean reward of -15.0, a substantial improvement over baseline |
| expB_4 (extra-large batch) | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 0.1 | **Best result of all 30 experiments** — reached -12.1 mean reward, showing real learning progress |
| expB_5 (low epsilon floor) | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Modest improvement to around -15.7; near-fully-greedy behavior helped exploit learned policy |
| expB_6 (high epsilon floor) | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 0.1 | Stayed flat near -20.8; excessive forced exploration prevented exploitation |
| expB_7 (fast epsilon decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | Modest improvement to around -17.2; committing to exploitation early helped somewhat |
| expB_8 (slow epsilon decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.3 | Modest improvement, comparable to fast decay; Pong tolerant of a range of decay schedules |
| expB_9 (low epsilon start) | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 0.1 | Stayed flat near -20.8; starting with less exploration hurt early learning |
| expB_10 (large batch + fast decay) | 0.0001 | 0.99 | 64 | 1.0 | 0.02 | 0.05 | Reached -15.6, confirming batch size as the dominant factor even combined with decay changes |
| C01 (baseline) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -21.0 (std 0.0); flat baseline at 50k timesteps |
| C02 (fast decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | Reward stayed near -21.0 (std 0.0); no measurable effect at this short budget |
| C03 (slow decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.5 | Reward stayed near -21.0 (std 0.0); no measurable effect at this short budget |
| C04 (high final epsilon) | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 0.3 | Reward -20.4, slightly better than pure baseline but not significant |
| C05 (low start epsilon) | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 0.1 | Reward stayed near -20.8; reduced early exploration did not help at this budget |
| C06 (high lr, low gamma) | 0.0005 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.8; combination showed no clear benefit |
| C07 (low lr, high gamma) | 5e-05 | 0.995 | 64 | 1.0 | 0.05 | 0.2 | Best of Member C's runs (trained 200k timesteps vs 50k for the others): best training reward -16, and evaluation episodes lasted nearly twice as long as any other C model (~5700 steps vs ~3100) — the agent visibly learned to return the ball and sustain rallies. Low lr + far-sighted gamma suits Pong's delayed rewards, though the result is partly explained by the longer training budget  |
| C08 (aggressive edge case) | 0.001 | 0.999 | 32 | 1.0 | 0.01 | 0.05 | Reward stayed near -21.0 (std 0.0); aggressive combined settings gave no benefit at this budget |
| C09 (large batch) | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 0.2 | Reward stayed near -21.0 (std 0.0); unlike Member B's larger-batch runs, this did not show improvement, likely due to the much shorter 50k-timestep budget |
| C10 (small batch) | 0.00025 | 0.97 | 16 | 1.0 | 0.10 | 0.3 | Reward stayed near -20.6; no clear improvement |

## Key Findings

1. **Batch size was the single most impactful hyperparameter** across all 30 experiments. Increasing batch size from 32 to 128 (Henriette, 500k timesteps) improved mean reward from ~-20.9 to **-12.1**, the best result the group achieved.
2. **Training duration matters as much as hyperparameter choice.** The two clearest learning signals in the group — Henriette's 500k-timestep batch experiments and Patricia's C07 (200k timesteps) — were also the longest-trained runs, while 50k–150k-timestep runs stayed flat near -21 regardless of configuration. Pong requires substantial training time before hyperparameter differences become visible.
3. **Learning rate and gamma had minimal standalone effect** within the timestep budgets tested; extreme values (very high or very low lr) showed no divergence but also no learning.
4. **Epsilon schedule had a moderate effect**, primarily through the exploration floor: too-high a floor (0.2) or too little initial exploration (start=0.5) consistently hurt performance, while a low floor (0.01) modestly helped.
5. **Best overall configuration:** `batch_size=128, lr=0.0001, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.1`, trained for 500,000 timesteps (Henriette Utatsineza, expB_4).


## Gameplay Demonstration

```markdown
![Pong agent gameplay](link-to-video-or-gif)
```
or
```markdown
[Watch the gameplay video](link-to-video-file-or-youtube)
```
