# DQN Pong - Deep Q-Learning Agent (Formative 3 Group 9)

A Deep Q-Network (DQN) agent trained with Stable Baselines3 and Gymnasium to play Atari Pong, including a full hyperparameter tuning study across three team members (30 experiments total).

## Team
- Cynthia Mutie — Training pipeline, MLP vs CNN comparison, lr/gamma sweep (Member A)
- Henriette Utatsineza — Evaluation pipeline, gameplay video, batch size/epsilon sweep (Member B)
- Patricia Mugabo — Documentation, submission logistics, combined-config sweep, final model (Member C)

See [Team Task Sheet](https://github.com/Muen1/dqn-pong-formative3/blob/main/submission/Team%20Task%20Sheet_%5BDeep%20Q%20Learning_Formative%203_C1%23_Group%209%23%5D.pdf) for a full breakdown of individual contributions.

## Environment
**Game:** Pong (`PongNoFrameskip-v4`, Gymnasium/ALE). The trained agent controls one paddle; the opposing paddle is Pong's built-in, non-learning opponent AI baked into the Atari ROM — this is not a two-player or self-play setup.

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
| expB_1 (baseline) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed flat near -20.8 over 500k timesteps; establishes the group's baseline |
| expB_2 (small batch) | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 0.1 | No improvement over baseline; smaller batch produced noisier gradient estimates that slowed learning |
| expB_3 (large batch) | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | **Best result of all 30 experiments** — reached -7.75 mean reward (last 20 episodes), with individual episodes peaking at +8, showing the agent regularly winning rallies against the built-in opponent |
| expB_4 (extra-large batch) | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.9, flat like baseline. An earlier training run at this configuration showed a promising trend (~-12.1) before compute-quota interruptions across platforms forced a rerun; the rerun did not reproduce the earlier result, illustrating the run-to-run variance inherent to DQN (see Limitations) |
| expB_5 (low epsilon floor) | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Modest improvement to around -18.8; near-fully-greedy behavior helped exploit learned policy slightly |
| expB_6 (high epsilon floor) | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 0.1 | Stayed flat near -20.8; excessive forced exploration prevented exploitation |
| expB_7 (fast epsilon decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | Strong improvement to -12.45, with individual episodes reaching -5.0; committing to exploitation early helped substantially |
| expB_8 (slow epsilon decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.3 | Solid improvement to -16.0, comparable direction to fast decay; Pong tolerant of a range of decay schedules |
| expB_9 (low epsilon start) | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 0.1 | Stayed flat near -20.85; starting with less exploration hurt early learning |
| expB_10 (large batch + fast decay) | 0.0001 | 0.99 | 64 | 1.0 | 0.02 | 0.05 | Reached -14.4, confirming batch size and faster decay both help, though not as strongly as batch=64 alone (expB_3) |
| C01 (baseline) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Baseline config, trained 200k timesteps (script default; other C runs except C07 used 50k). Even with the longer budget the greedy policy stayed near -21.0 — evidence that Pong needs far more training before configs separate |
| C02 (fast decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | Reward stayed near -21.0 (std 0.0); no measurable effect at this short budget |
| C03 (slow decay) | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.5 | Reward stayed near -21.0 (std 0.0); no measurable effect at this short budget |
| C04 (high final epsilon) | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 0.3 | Reward -20.4, slightly better than pure baseline but not significant |
| C05 (low start epsilon) | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 0.1 | Reward stayed near -20.8; reduced early exploration did not help at this budget |
| C06 (high lr, low gamma) | 0.0005 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | Reward stayed near -20.8; combination showed no clear benefit |
| C07 (low lr, high gamma) | 5e-05 | 0.995 | 64 | 1.0 | 0.05 | 0.2 | Best of Member C's runs (trained 200k timesteps vs 50k for the others): best training reward -16, and evaluation episodes lasted nearly twice as long as any other C model (~5700 steps vs ~3100) — the agent visibly learned to return the ball and sustain rallies. Low lr + far-sighted gamma suits Pong's delayed rewards, though the result is partly explained by the longer training budget |
| C08 (aggressive edge case) | 0.001 | 0.999 | 32 | 1.0 | 0.01 | 0.05 | Reward stayed near -21.0 (std 0.0); aggressive combined settings gave no benefit at this budget |
| C09 (large batch) | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 0.2 | Reward stayed near -21.0 (std 0.0); unlike Member B's larger-batch runs, this did not show improvement, likely due to the much shorter 50k-timestep budget |
| C10 (small batch) | 0.00025 | 0.97 | 16 | 1.0 | 0.10 | 0.3 | Reward stayed near -20.6; no clear improvement |

## Key Findings

1. **Batch size was the single most impactful hyperparameter** across all 30 experiments. Increasing batch size from 32 to 64 (Henriette, 500k timesteps) improved mean reward from ~-20.8 to **-7.75**, the best result the group achieved, with the agent regularly winning individual rallies.
2. **Training duration matters as much as hyperparameter choice.** The clearest learning signals in the group — Henriette's 500k-timestep batch/epsilon experiments and Patricia's C07 (200k timesteps) — were also the longest-trained runs, while 50k–150k-timestep runs stayed flat near -21 regardless of configuration. Pong requires substantial training time before hyperparameter differences become visible.
3. **Learning rate and gamma had minimal standalone effect** within the timestep budgets tested; extreme values (very high or very low lr) showed no divergence but also no learning.
4. **Epsilon schedule had a real effect on top of batch size.** A fast decay to a low floor (expB_7, expB_8) gave the second- and third-best results in the whole group, while too-high an exploration floor (0.2) or too little initial exploration (start=0.5) consistently hurt performance.
5. **Best overall configuration:** `batch_size=64, lr=0.0001, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.1`, trained for 500,000 timesteps (Henriette Utatsineza, expB_3) — final mean reward **-7.75** over the last 20 evaluation episodes.

## Limitations

DQN training showed noticeable run-to-run variance even under identical hyperparameters, which is expected given the algorithm's sensitivity to random weight initialization, exploration noise, and replay buffer sampling order. This was compounded by training being split across multiple cloud platforms (Google Colab and Kaggle) due to free-tier GPU quota limits, introducing minor hardware/driver differences between runs. As a result, one configuration (expB_4, batch_size=128) that showed a promising trend in an earlier session did not reproduce when rerun after a quota interruption. Given more compute, each configuration would ideally be run with multiple random seeds and averaged to give a more reliable comparison.

## Gameplay Demonstration

The agent shown below plays as the right paddle against Pong's built-in (non-learning) opponent AI, using the trained `expB_3` model (batch_size=64) with a fully greedy policy.

```markdown
[Watch the gameplay video](gameplay_videos/pong_agent-step-0-to-step-100000.mp4)
or 
```
[Watch the gameplay video on youtube](https://youtu.be/joO35d-Vyhs)
```