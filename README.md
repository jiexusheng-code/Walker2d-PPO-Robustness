# Walker2d PPO Robustness

A reproducible comparison of standard PPO and PPO trained with modest MuJoCo dynamics randomization on Gymnasium's `Walker2d-v5` task. The project targets CPU-only Linux cloud instances; no GPU is required.

The experiment tests one question: **does training across small variations in contact friction and actuator strength improve performance when the dynamics differ from the nominal simulator?** It does not assume the randomized policy will win. Report every seed and every evaluation condition, including neutral or negative results.

## Method

- **Baseline:** PPO trains on the nominal Walker2d-v5 dynamics.
- **Domain-randomized:** same PPO code and hyperparameters; at each episode reset, every MuJoCo geom-friction vector (including the floor and feet) is scaled uniformly from `[0.8, 1.2]`, and all actuator gears from `[0.9, 1.1]`. Scaling all contact geoms together preserves the nominal model at scale `1.0` while ensuring the effective contact friction changes.
- **Evaluation:** deterministic policy on the nominal environment and two fixed held-out dynamics settings: `(contact-friction scale=0.7, actuator scale=0.7)` and `(contact-friction scale=1.3, actuator scale=1.3)`. Both are outside the training randomization ranges.
- The baseline and randomized variant use identical seeds, step budgets, evaluation episode seeds, and PPO settings. The primary metric is, for each training seed, the average of the `heldout_low` and `heldout_high` mean episode returns. Report both held-out conditions separately, the nominal return as a guard, and paired per-seed differences (domain-randomized minus baseline) with their mean and spread. Results are preserved per episode in CSV and per condition in JSON.

This tests robustness to two simulator parameters only. It is not evidence of sim-to-real transfer or performance on a physical robot.

## Requirements and setup

Use Python 3.11 or 3.12 and `uv` on Linux. The lockfile was generated and checked with uv 0.10.10; use that version or a compatible newer one. If `uv` is missing, install it with the official installer, then start from this project directory:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
```

The dependencies include Gymnasium MuJoCo, Stable-Baselines3, and TensorBoard. PyTorch is resolved from the CPU-only wheel index to avoid installing CUDA packages on a CPU run. Rendering is not used during training/evaluation.

## First cloud run: smoke test

Before starting the full experiment, run the short end-to-end test:

```bash
bash scripts/cloud_smoke.sh
```

It installs the locked environment, runs tests that verify effective MuJoCo contact friction and seeded randomization, trains both variants for 4,096 steps, saves models, evaluates them in all three conditions, and writes artifacts to `runs/smoke/`. This checks the runtime path; it does **not** establish that either policy has learned to walk.

Next, measure cloud throughput before choosing a large budget:

```bash
bash scripts/benchmark.sh
```

This runs 50,000 steps for each variant and records elapsed time and steps/second in each run's `config.json`. Use those measurements to decide whether to keep or reduce the default budget before launching the full matrix. Also inspect TensorBoard's `rollout/ep_rew_mean`; 50k is a throughput check, not proof that PPO has learned to walk. If returns show no learning signal, run a longer single-seed pilot and review the configuration before paying for all six runs.

## Full experiment

Default matrix: two variants × seeds 0, 1, 2 × 1,000,000 requested environment steps per run, followed by 100 deterministic episodes per test condition. It is sequential and does not resume an interrupted PPO run. To survive SSH disconnects, run it inside `tmux`:

```bash
mkdir -p runs
tmux new -s walker2d
# In tmux:
bash scripts/run_matrix.sh 2>&1 | tee runs/matrix.log
# Detach with Ctrl-b, then d; reattach with: tmux attach -t walker2d
```

To run it in the foreground instead, use:

```bash
bash scripts/run_matrix.sh
```

Override the budget or episode count if needed:

```bash
TIMESTEPS=1500000 SEEDS="0 1 2" EPISODES=100 bash scripts/run_matrix.sh
```

The script runs jobs sequentially to keep CPU and memory use predictable. To launch an individual run manually:

```bash
uv run python train.py --variant baseline --seed 0 --timesteps 1000000
uv run python evaluate.py \
  --model runs/baseline/seed_0/final_model.zip \
  --episodes 100 --seed 40000 \
  --output-dir runs/baseline/seed_0/evaluation
```

Use `--variant domain-rand` for the randomized training condition. The script defaults its outputs to `runs/<variant>/seed_<seed>/`; the matrix script uses `runs/main/...` to separate the formal runs. Do not run both into the same output directory.

## Outputs

Each training directory contains:

- `config.json`: requested and actual timesteps, PPO settings, package versions, timing, and randomization bounds.
- `final_model.zip` and periodic checkpoints.
- `tensorboard/`: TensorBoard event files.
- `evaluation/episodes.csv` and `evaluation/summary.json`: per-episode and per-condition returns, forward velocity, displacement, fall rate, and time-limit rate.

View training curves with:

```bash
uv run tensorboard --logdir runs/main --host 127.0.0.1 --port 6006
```

On a remote instance, reach it through an SSH tunnel; do not expose port 6006 publicly.

Keep the cloud run directory intact and download/archive it before deleting the instance. `runs/` is git-ignored.

## Interpretation

Compare `heldout_low` and `heldout_high` per seed first, then average those two condition means within each seed for the primary score. Calculate each seed's paired difference as domain-randomized minus baseline, and summarize the three paired differences with their mean and sample standard deviation; show each held-out condition and nominal returns separately. A gain on held-out dynamics accompanied by a nominal regression is a tradeoff, not an unconditional win. Three seeds are a small sample and should not be presented as definitive statistical proof.
