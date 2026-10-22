# Walker2d PPO Robustness

A reproducible comparison of standard PPO, PPO with established implementation details, and a Truly PPO port on Gymnasium's `Walker2d-v5` task. The project targets CPU-only Linux cloud instances; no GPU is required.

The experiment tests whether Walker2d-supported PPO implementation details improve over the current SB3 baseline, and whether adding Truly PPO provides a further gain. It does not assume either treatment will win. Report every seed and every evaluation condition, including neutral or negative results.

## Method

- **Baseline:** Stable-Baselines3 PPO on nominal Walker2d-v5 dynamics with the project's original hyperparameters.
- **PPO implementation details:** the same PPO configuration with observation/reward normalization (`VecNormalize`) and a linear learning-rate schedule. Walker2d studies motivate these choices, but their measured gains are not guaranteed to transfer to v5.
- **Truly PPO:** the normalized, scheduled PPO setup plus Truly PPO's KL-triggered rollback objective (`delta=0.03`, `alpha=5` for non-Humanoid MuJoCo). The paper reports improved Walker2d results, but does not specify the Walker2d version; this project tests the port on v5.
- All arms use identical training seeds, step budgets, evaluation episode seeds, and evaluation conditions. The primary comparison is nominal mean episode return per seed; held-out low/high dynamics are secondary robustness measures. Report paired per-seed differences and spread rather than pooling episodes across seeds.

This is an empirical comparison, not a promise that either treatment will win. Held-out conditions change friction and actuator strength together, so they measure joint dynamics robustness rather than isolating either parameter.

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

It installs the locked environment, runs unit tests, trains all three variants for 4,096 steps, saves models, evaluates them in all three conditions, and writes artifacts to `runs/smoke/`. This checks the runtime path; it does **not** establish that any policy has learned to walk.

Next, measure cloud throughput before choosing a large budget:

```bash
bash scripts/benchmark.sh
```

This runs each of the three PPO variants for 50,000 steps and records elapsed time and steps/second in each run's `config.json`. Use those measurements to decide whether to keep or reduce the default budget before launching the full matrix. Also inspect TensorBoard's `rollout/ep_rew_mean`; 50k is a throughput check, not proof that PPO has learned to walk. If returns show no learning signal, run a longer single-seed pilot and review the configuration before paying for all 9 runs.

## Full experiment

Default matrix: three variants × seeds 0, 1, 2 × 1,500,000 requested environment steps per run, followed by 100 deterministic episodes per evaluation condition. That is 9 training runs; at the measured CPU throughput (~750–960 steps/s on a single thread) each run takes roughly 26–33 minutes, so the full matrix is about 4.5 hours. It is sequential and skips runs that already have both `final_model.zip` and `evaluation/summary.json`. Resuming is opt-in: without `RESUME=1` an interrupted run restarts from scratch. To survive SSH disconnects, run it inside `tmux`:

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
TIMESTEPS=3000000 SEEDS="0 1 2" EPISODES=100 bash scripts/run_matrix.sh
```

The script runs jobs sequentially to keep CPU and memory use predictable. To launch an individual run manually:

```bash
uv run python train.py --variant baseline --seed 0 --timesteps 1500000
uv run python evaluate.py \
  --model runs/baseline/seed_0/final_model.zip \
  --variant baseline \
  --episodes 100 --seed 40000 \
  --output-dir runs/baseline/seed_0/evaluation
```

The supported variants are `baseline`, `ppo-normalized`, and `truly-ppo`. The script defaults its outputs to `runs/<variant>/seed_<seed>/`; the matrix script uses `runs/main/...` to separate the formal runs. Do not run variants into the same output directory.

## Resuming an interrupted run

If a run is stopped (SSH drop, crash, `Ctrl-C`), it can continue from its latest checkpoint instead of starting over. Resume is off by default; enable it explicitly:

```bash
RESUME=1 bash scripts/run_matrix.sh
```

With `RESUME=1`, `run_matrix.sh` passes `--resume` to every training job, which loads `final_model.zip` when it exists, otherwise the checkpoint with the largest step count under `checkpoints/`, together with the matching normalization statistics (checkpoint statistics are stored as `checkpoints/ppo_walker2d_vecnormalize_<steps>_steps.pkl`). Without the flag, a partially trained run restarts from zero: its `final_model.zip`, `vecnormalize.pkl`, and checkpoints are deleted first, so a later `RESUME=1` cannot pick up artifacts from the discarded run. Earlier TensorBoard event files and `evaluation/` outputs are left in place until overwritten. `--timesteps` always means the absolute budget for the run: a resumed run trains only the remaining steps, and the linear learning-rate schedule keeps annealing against the original budget rather than restarting. To resume manually:

```bash
uv run python train.py --variant truly-ppo --seed 1 --timesteps 1500000 \
  --run-dir runs/main/truly-ppo/seed_1 --resume
uv run python train.py --variant truly-ppo --seed 1 --timesteps 1500000 \
  --run-dir runs/main/truly-ppo/seed_1 \
  --resume runs/main/truly-ppo/seed_1/checkpoints/ppo_walker2d_1000000_steps.zip
```

`--resume` without a value auto-detects the starting point in `--run-dir`; with a value it takes a run directory or a model `.zip`. A normalized run refuses to resume if its `vecnormalize.pkl` is missing, because the observation statistics cannot be reconstructed.

Limits: the rollout buffer is not checkpointed, so the rollout in flight when the process died (at most `n_steps - 1 = 2047` steps) is discarded, and the environment RNG stream is not restored. A resumed run is a close continuation, not a bit-exact one. TensorBoard appends to the run's existing event directory, and `config.json` records `resumed_from`, `resume_start_timesteps`, and `resume_history`.

## Outputs

Each training directory contains:

- `config.json`: requested and actual timesteps, PPO settings, normalization settings, package versions, timing, and resume history.
- `final_model.zip` and periodic checkpoints; normalized variants also save `vecnormalize.pkl` for correctly normalized evaluation.
- `tensorboard/`: TensorBoard event files.
- `evaluation/episodes.csv` and `evaluation/summary.json`: per-episode and per-condition returns, forward velocity, displacement, fall rate, and time-limit rate.

For normalized variants, evaluation loads `vecnormalize.pkl` from the model's directory by default. When evaluating a checkpoint, pass `--vecnormalize checkpoints/ppo_walker2d_vecnormalize_<steps>_steps.pkl` explicitly, because checkpoint statistics are saved under `checkpoints/` rather than next to the checkpoint file.

View training curves with:

```bash
uv run tensorboard --logdir runs/main --host 127.0.0.1 --port 6006
```

On a remote instance, reach it through an SSH tunnel; do not expose port 6006 publicly.

`rollout/ep_rew_mean` is logged from the inner `Monitor`, which sits inside `VecNormalize`, so it records raw episode returns for every arm and is useful as a cross-arm training diagnostic. The `ppo-normalized` and `truly-ppo` arms still optimize normalized rewards (`norm_reward=True`), so their training dynamics differ, and TensorBoard should not replace the controlled comparison. The formal outcome is the deterministic raw-return evaluation in `evaluation/episodes.csv` and `evaluation/summary.json`.

Keep the cloud run directory intact and download/archive it before deleting the instance. `runs/` is git-ignored.

## Interpretation

Use nominal mean return as the primary outcome for each seed. Report the paired per-seed differences for `ppo-normalized` minus `baseline`, `truly-ppo` minus `ppo-normalized`, and `truly-ppo` minus `baseline`; summarize means and sample standard deviations across the three seeds. Report both held-out conditions separately as secondary robustness measures. Three seeds is a small sample: it cannot support significance claims, and a single divergent seed dominates the mean and standard deviation. Prefer reporting all three per-seed values and the paired differences rather than only the aggregate, and treat any ranking as preliminary unless the paired differences are consistent across all three seeds. Adding seeds 3 and 4 later with `SEEDS="3 4"` is cheap if the budget allows.
