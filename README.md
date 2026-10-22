# Walker2d PPO Robustness

A reproducible comparison of standard PPO, PPO with established implementation details, and a Truly PPO port on Gymnasium's `Walker2d-v5` task. The project targets CPU-only Linux cloud instances; no GPU is required.

The current recommended route is the standard PPO reference pilot below. The original three-arm experiment, which tests whether Walker2d-supported PPO implementation details improve over the SB3 baseline and whether adding Truly PPO provides a further gain, is kept as legacy context. Neither experiment assumes a treatment will win. Report every seed and every evaluation condition, including neutral or negative results.

## Recommended first cloud run: standard PPO reference pilot

For the first end-to-end cloud run, use the standard PPO reference pilot instead of the full matrix:

```bash
git pull --ff-only
bash scripts/pilot_reference.sh
```

The script syncs the locked environment, trains `ppo-reference` for 1,000,000 steps with seed 0, evaluates every 50,000 steps (10 episodes with periodic seed 50,000), and then runs the final deterministic evaluation over nominal, heldout_low, and heldout_high: 100 episodes per condition with hold-out per-episode seeds 40,000 + episode. It writes to a new `runs/reference_pilot/<UTC timestamp>_seed_<seed>/` directory and refuses to write under `runs/main/`. Override defaults only when needed:

```bash
TIMESTEPS=2000000 SEED=1 bash scripts/pilot_reference.sh
RUN_DIR=runs/reference_pilot/my_run bash scripts/pilot_reference.sh
RESUME=1 RUN_DIR=runs/reference_pilot/my_run bash scripts/pilot_reference.sh
```

A non-empty `RUN_DIR` is refused unless `RESUME=1`; conversely, `RESUME=1` requires an explicit `RUN_DIR`, so a resume can never silently create a new run. The pilot does not run the smoke test, the full matrix, or the training-heavy test suite automatically.

The fixed recipe follows the Walker2d entry in DLR-RM/rl-baselines3-zoo's [`ppo.yml`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml): constant learning rate 5.05041e-5, `n_steps=512`, `batch_size=32`, `n_epochs=20`, `clip_range=0.1`, `ent_coef=0.000585045`, `vf_coef=0.871923`, `max_grad_norm=1.0`, `gamma=0.99`, `gae_lambda=0.95`, and 64x64 policy/value networks. The source recipe is for Walker2d-v4; porting it unchanged to Walker2d-v5 has **not** been validated. Treat the pilot as an empirical reference, not an expected score.

This round implements the reference profile for **standard PPO only**. A Truly PPO variant on the same reference base is not implemented; that would be a separate experiment requiring its own confirmation. Existing baseline, ppo-normalized, and truly-ppo results remain available. `scripts/run_matrix.sh` now guards budget and resume decisions, but its default variants and training parameters are unchanged. The pilot writes only under `runs/reference_pilot/` and refuses `runs/main/`.

The periodic evaluator calls `env.reset(seed=50000)` once and then continues the environment RNG stream, so its episode sets are not identical to the final evaluation's per-episode seeds 40,000 + i. Use periodic evaluations to choose checkpoints and `best_model`; report the final hold-out evaluation as the primary result. Do not substitute `best_model` for `final_model` in the main result.

A healthy episode reaching 1,000 steps in at least 90% of evaluations is a project target, not a guarantee. Run seed 0 first and review it before spending on additional seeds.

Download or archive the whole run directory before deleting the instance: `console.log`, the Monitor CSV, metrics JSONL, periodic evaluation npz files, checkpoints, `best/`, and `evaluation/`. TensorBoard event files are diagnostic only; view them through an SSH tunnel and never expose port 6006 publicly.

## Method (legacy three-arm comparison)

> The method and full-matrix sections below describe the original three-arm comparison. They remain for reproducing earlier results but are not the current recommended route. The reference pilot uses different PPO hyperparameters, so do not treat a new reference run and the old arms as sharing the same base.

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

## Optional cloud smoke test

If you explicitly want a short end-to-end environment check, run:

```bash
bash scripts/cloud_smoke.sh
```

It installs the locked environment, runs the full test suite (including training-based tests), trains all three older variants for 4,096 steps, saves models, evaluates them in all three conditions, and writes artifacts to `runs/smoke/`. This checks the runtime path; it does **not** establish that any policy has learned to walk. The reference pilot does not run this automatically.

If you plan to change the budget, measure cloud throughput first:

```bash
bash scripts/benchmark.sh
```

This runs each of the three PPO variants for 50,000 steps and records elapsed time and steps/second in each run's `config.json`. Use those measurements to decide whether to keep or reduce the default budget before launching the full matrix. Also inspect TensorBoard's `rollout/ep_rew_mean`; 50k is a throughput check, not proof that PPO has learned to walk. If returns show no learning signal, run a longer single-seed pilot and review the configuration before paying for all 9 runs.

## Full experiment (legacy three-arm matrix)

The original matrix remains available for reproducing the earlier three-arm results. It is not the reference pilot.

Default matrix: three variants × seeds 0, 1, 2 × 1,500,000 requested environment steps per run, followed by 100 deterministic episodes per evaluation condition. That is 9 training runs; at the measured CPU throughput (~750–960 steps/s on a single thread) each run takes roughly 26–33 minutes, so the full matrix is about 4.5 hours. It is sequential. It skips a run only when its recorded completed steps meet the requested budget, its config matches the variant and seed, and its evaluation matches the final model and all three conditions. If an existing run has fewer recorded steps than the requested budget, the script stops with a friendly error unless `RESUME=1`; it never restarts from scratch over existing artifacts. With `RESUME=1`, a larger budget continues training and then re-evaluates. To survive SSH disconnects, run it inside `tmux`:

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

The supported variants are `baseline`, `ppo-normalized`, `truly-ppo`, and `ppo-reference`. `ppo-reference` is the fixed standard-PPO reference used by the pilot; it uses normalized observations and rewards with a constant learning rate. The script defaults its outputs to `runs/<variant>/seed_<seed>/`; the matrix script uses `runs/main/...` to separate the formal runs. Do not run variants into the same output directory.

## Resuming an interrupted run

If a run is stopped (SSH drop, crash, `Ctrl-C`), it can continue from its latest checkpoint instead of starting over. Resume is off by default; enable it explicitly:

```bash
RESUME=1 bash scripts/run_matrix.sh
```

With `RESUME=1`, `run_matrix.sh` passes `--resume` to every training job, which loads `final_model.zip` when it exists, otherwise the checkpoint with the largest step count under `checkpoints/`, together with the matching normalization statistics (checkpoint statistics are stored as `checkpoints/ppo_walker2d_vecnormalize_<steps>_steps.pkl`). Without the flag, `run_matrix.sh` refuses to overwrite an existing run whose recorded steps are below the requested budget. Calling `train.py` directly without `--resume` still restarts the legacy variants from zero: their `final_model.zip`, `vecnormalize.pkl`, and checkpoints are deleted first, so a later `RESUME=1` cannot pick up artifacts from the discarded run. The `ppo-reference` variant is stricter: it refuses to overwrite existing reference artifacts unless `--resume` is passed. Earlier TensorBoard event files and `evaluation/` outputs are left in place until overwritten. `--timesteps` always means the absolute budget for the run: a resumed run trains only the remaining steps, and the linear learning-rate schedule keeps annealing against the original budget rather than restarting. To resume manually:

```bash
uv run python train.py --variant truly-ppo --seed 1 --timesteps 1500000 \
  --run-dir runs/main/truly-ppo/seed_1 --resume
uv run python train.py --variant truly-ppo --seed 1 --timesteps 1500000 \
  --run-dir runs/main/truly-ppo/seed_1 \
  --resume runs/main/truly-ppo/seed_1/checkpoints/ppo_walker2d_1000000_steps.zip
```

`--resume` without a value auto-detects the starting point in `--run-dir`; with a value it takes a run directory or a model `.zip`. A normalized run refuses to resume if its `vecnormalize.pkl` is missing, because the observation statistics cannot be reconstructed.

Limits: the rollout buffer is not checkpointed, so the rollout in flight when the process died is discarded (at most `n_steps - 1`: 2,047 steps for the legacy variants with `n_steps=2048`, 511 for `ppo-reference` with `n_steps=512`), and the environment RNG stream is not restored. A resumed run is a close continuation, not a bit-exact one. TensorBoard appends to the run's existing event directory, and `config.json` records `resumed_from`, `resume_start_timesteps`, and `resume_history`.

## Outputs

Each training directory contains:

- `config.json`: requested and actual timesteps, PPO settings, normalization settings, package versions, timing, status, and resume history.
- `final_model.zip` and periodic checkpoints; normalized variants also save `vecnormalize.pkl` for correctly normalized evaluation.
- `tensorboard/`: TensorBoard event files.
- A Monitor CSV and `metrics.jsonl` with training diagnostics (see below).
- Periodic evaluation npz files, and `best/best_model.zip` plus `best/vecnormalize.pkl` when periodic evaluation selects a best model.
- `evaluation/episodes.csv` and `evaluation/summary.json`: per-episode and per-condition returns, forward velocity, displacement, fall rate, time-limit rate, healthy time-limit rate, and full-length rate. The summary also records deterministic evaluation metadata, model and VecNormalize SHA-256 hashes, the evaluation timestamp/version, and the training seed when it can be read from `config.json`.

For normalized variants, evaluation loads `vecnormalize.pkl` from the model's directory by default. When evaluating a checkpoint, pass `--vecnormalize checkpoints/ppo_walker2d_vecnormalize_<steps>_steps.pkl` explicitly, because checkpoint statistics are saved under `checkpoints/` rather than next to the checkpoint file. The pilot also evaluates `best/best_model.zip` when it exists, using `best/vecnormalize.pkl`; that evaluation is diagnostic and does not replace the final-model result.

View training curves with:

```bash
uv run tensorboard --logdir runs/main --host 127.0.0.1 --port 6006
```

On a remote instance, reach it through an SSH tunnel; do not expose port 6006 publicly.

`rollout/ep_rew_mean` is logged from the inner `Monitor`, which sits inside `VecNormalize`, so it records raw episode returns for every arm and is useful as a cross-arm training diagnostic. The `ppo-normalized`, `ppo-reference`, and `truly-ppo` arms still optimize normalized rewards (`norm_reward=True`), so their training dynamics differ, and TensorBoard should not replace the controlled comparison. The formal outcome is the deterministic raw-return evaluation in `evaluation/episodes.csv` and `evaluation/summary.json`.

## Training diagnostics

The run directory keeps the machine-readable training diagnostics intended for reviewing learning dynamics: `metrics.jsonl` (PPO update summaries and rollout episode summaries), the Monitor CSV (raw episode returns and lengths), periodic evaluation npz files, and `console.log`. The exact JSON keys are still evolving; inspect the first records rather than assuming a fixed schema.

Read the shape of these curves rather than single values:

- **PPO update health:** approximate KL, clip fraction, entropy, policy standard deviation, policy/value losses, and explained variance. A sudden KL or clip spike together with collapsing policy std or unstable losses is a reason to inspect the run. It is not proof of a specific root cause.
- **Exploration:** entropy and policy std should not collapse very early. A gradual decline is normal; an almost immediate collapse is worth investigating.
- **Rollout episodes:** raw episode return, length, and the healthy-termination proxy show whether behavior improves between updates, independently of normalized rewards.
- **Action and observation normalization:** action-clipping fraction, normalized-observation clipping fraction, and observation RMS scale flag saturation or a normalization mismatch. Persistent clipping or an RMS scale far from the expected range suggests checking the saved VecNormalize statistics; it does not by itself identify the cause.

These diagnostics are correlational. Do not encode fixed thresholds as causal rules; compare them with the final deterministic evaluation and the raw Monitor returns. Periodic evaluations use seed 50,000 for checkpoint/best selection, while the final evaluation uses per-episode seeds 40,000 + i, so the two episode sets are not identical.

Keep the cloud run directory intact and download/archive it before deleting the instance. `runs/` is git-ignored.

## Interpretation (legacy three-arm comparison)

Use nominal mean return as the primary outcome for each seed. Report the paired per-seed differences for `ppo-normalized` minus `baseline`, `truly-ppo` minus `ppo-normalized`, and `truly-ppo` minus `baseline`; summarize means and sample standard deviations across the three seeds. Report both held-out conditions separately as secondary robustness measures. Three seeds is a small sample: it cannot support significance claims, and a single divergent seed dominates the mean and standard deviation. Prefer reporting all three per-seed values and the paired differences rather than only the aggregate, and treat any ranking as preliminary unless the paired differences are consistent across all three seeds. Adding seeds 3 and 4 later with `SEEDS="3 4"` is cheap if the budget allows.
