# Walker2d PPO Robustness

A reproducible two-arm comparison on Gymnasium's `Walker2d-v5`:

- **`ppo-reference`**: Stable-Baselines3 PPO with the standard PPO-Clip actor objective.
- **`truly-ppo`**: the same shared reference recipe with only the actor objective replaced by Truly PPO's KL-triggered rollback.

Both arms use the same environment, observation/reward normalization, network, learning rate and schedule, rollout settings, value/entropy coefficients, seed, evaluation protocol, and step budget. The comparison is empirical; it does not assume either arm will win. Report every seed and every condition, including neutral or negative results.

## Shared recipe and the only planned difference

| Setting | Value |
| --- | --- |
| Environment | `Walker2d-v5` |
| Normalization | `VecNormalize` observations and rewards, clip 10, frozen during evaluation |
| Network | `[64, 64]` tanh policy and value networks |
| Learning rate | constant `5.05041e-5` |
| Rollout / update | `n_steps=512`, `batch_size=32`, `n_epochs=20` |
| Discount / GAE | `gamma=0.99`, `gae_lambda=0.95` |
| PPO clip | `clip_range=0.1` (used only by the standard PPO actor) |
| Entropy / value | `ent_coef=0.000585045`, `vf_coef=0.871923` |
| Gradient clipping | `max_grad_norm=1.0` |
| Environments | `n_envs=1`, seed controlled per run |

The PPO hyperparameters follow the Walker2d entry in DLR-RM/rl-baselines3-zoo's [`ppo.yml`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml). That entry targets `Walker2d-v4`; this project ports it unchanged to `Walker2d-v5`, which has not been independently validated as an optimal recipe.

The only planned actor difference is:

- `ppo-reference`: PPO-Clip actor objective from Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347, 2017), as implemented by Stable-Baselines3.
- `truly-ppo`: Truly PPO's KL-triggered rollback with `alpha=5`, `delta=0.03` (Wang, He, Wen, and Tan, *Truly Proximal Policy Optimization*, arXiv:1903.07940, 2019; extended 2020 version). When `kl(old||new) >= delta` and `ratio * advantage > advantage` (the authors' strict form, equivalent to `(ratio - 1) * advantage > 0`), the actor term becomes `ratio * advantage - alpha * kl(old||new)`. This is a KL penalty **inside the actor objective**. It is not a parameter, optimizer-state, or checkpoint restore. The paper's constant offset in the non-triggered branch is omitted because it has zero gradient. This is an SB3 port, not a bit-exact reproduction of the paper's training setup.

The rollback is a paper port, not a guarantee of higher return, sample efficiency, or robustness to perturbed dynamics. Held-out conditions change friction and actuator strength together, so they measure joint dynamics robustness rather than isolating either parameter. The trigger condition follows the authors' reference implementation: [wangyuhuix/TrulyPPO](https://github.com/wangyuhuix/TrulyPPO).

## Cloud run: both arms

Requirements: Python 3.11 or 3.12, `uv`, and a CPU-only Linux instance. This project is developed with uv 0.10.10; use that version or a compatible newer one.

```bash
git pull --ff-only
uv sync --locked
bash scripts/run_matrix.sh
```

Defaults: both arms, seed 0, 1,000,000 steps per arm, periodic evaluation every 50,000 steps (10 episodes, periodic RNG seed 50,000), then the final deterministic evaluation over `nominal`, `heldout_low`, and `heldout_high` with 100 episodes per condition and per-episode seeds `40000 + i`. Runs are sequential. The matrix writes to the new root `runs/two_arm/<variant>/seed_<seed>/`; the retired three-group history under `runs/main/` is left untouched. The startup line and each run's `config.json` prominently record the variant and `shared_profile=ppo-reference`.

Review seed 0 before paying for more seeds, then expand explicitly:

```bash
SEEDS="0 1 2" bash scripts/run_matrix.sh
```

To survive an SSH disconnect, run it inside `tmux`:

```bash
tmux new -s walker2d
mkdir -p runs
set -o pipefail
bash scripts/run_matrix.sh 2>&1 | tee runs/matrix.log
# Detach with Ctrl-b, then d; reattach with: tmux attach -t walker2d
```

Other entry points:

- `bash scripts/pilot_reference.sh` trains **only** `ppo-reference` and writes a timestamped directory under `runs/reference_pilot/`. It is a single-arm pilot; use `run_matrix.sh` for the two-arm comparison.
- `bash scripts/cloud_smoke.sh` runs the unit tests, trains both arms for 4,096 steps, and evaluates them briefly under a timestamped `runs/smoke/` directory.
- `bash scripts/benchmark.sh` trains both arms for 50,000 steps and writes timestamped throughput measurements under `runs/benchmark/`.

All scripts use `uv sync --locked` and do not modify dependencies or the lockfile. They run sequentially; there is no scheduler or concurrency layer.

## Results and evaluation protocol

Each run directory contains:

- `config.json`: variant, shared profile name, actor objective, all shared PPO settings, normalization, evaluation settings, source hashes, package versions, status, and resume history.
- `final_model.zip` and `vecnormalize.pkl`: the primary result and its matching frozen statistics.
- `checkpoints/`, `best/best_model.zip`, `best/vecnormalize.pkl`, `tensorboard/`, `monitor.csv`, `metrics.jsonl`, and `evaluations.npz`: diagnostics and resume material.
- `evaluation/episodes.csv` and `evaluation/summary.json`: final deterministic per-episode and per-condition results.

The **final model** is the primary result. The final evaluator loads matching VecNormalize statistics, keeps them frozen (`training=False`, `norm_reward=False`), evaluates raw environment return, and uses the same base seed protocol (`base_seed + episode`) for both arms. `summary.json` records the model and statistics SHA-256 hashes, variant, condition set, episode count, base seed, and determinism flag.

`best/best_model.zip` is **diagnostic only**. It is selected by periodic evaluation with periodic RNG seed 50,000, which is a different episode stream from the final evaluation. Never substitute it for the final-model result.

The matrix skips a completed run only when its final evaluation matches the exact final model SHA-256, VecNormalize SHA-256, variant, all three conditions, the requested episode count, and the final base seed. A stale, foreign, or one-episode summary is re-evaluated rather than treated as current. Compare a fixed step budget across arms and report paired per-seed differences and spread; do not pool episodes across seeds.

## Resume and overwrite safety

Both arms own their run directory. A fresh start refuses to write over existing run artifacts. Enable continuation explicitly:

```bash
RESUME=1 bash scripts/run_matrix.sh
```

or manually:

```bash
uv run python train.py --variant truly-ppo --seed 0 --timesteps 1000000 \
  --run-dir runs/two_arm/truly-ppo/seed_0 --resume
```

`--resume` without a value picks `final_model.zip`, otherwise the latest checkpoint, together with the matching statistics. It never silently starts a new run when there is nothing to resume. Resume validates the configuration before training: variant, seed, shared profile (`ppo-reference`), all shared PPO settings, normalization, and the actor objective (including `alpha`/`delta` for Truly PPO). Legacy `baseline`, `ppo-normalized`, and old 2048-step `truly-ppo` bases are rejected with a clear error instead of being resumed into the new profile. `--timesteps` remains the absolute budget; a resumed run trains only the remaining steps.

Limits: the rollout buffer and environment RNG stream are not checkpointed, so a resumed run is a close continuation, not a bit-exact one. With `n_steps=512`, at most the rollout in flight is discarded.

## Legacy historical models

The old three-arm training plan (`baseline`, `ppo-normalized`, `truly-ppo` on the old recipe) is retired. `train.py` accepts only `ppo-reference` and `truly-ppo`. Existing historical runs and models are left untouched. A historical unnormalized baseline model can still be evaluated offline:

```bash
uv run python evaluate.py \
  --model runs/pilot/baseline/seed_0/final_model.zip \
  --variant baseline \
  --episodes 100 --seed 40000 \
  --output-dir runs/pilot/baseline/seed_0/evaluation
```

This is evaluation-only compatibility, not a third training group. Do not mix those results with the new shared-profile two-arm comparison.

## Training diagnostics

Keep the whole run directory. `metrics.jsonl`, `monitor.csv`, `evaluations.npz`, and TensorBoard events are diagnostic, not the primary outcome. Useful signals:

- PPO update health: approximate KL, clip fraction, entropy, policy standard deviation, losses, explained variance.
- Exploration: entropy and policy std should not collapse immediately.
- Rollout episodes: raw episode return, length, and healthy-timeout rate.
- Normalization: action/observation clipping fractions and observation RMS scale.

These are correlational. A spike or collapse is a reason to inspect a run, not proof of a cause. View TensorBoard through an SSH tunnel (`uv run tensorboard --logdir runs/two_arm --host 127.0.0.1 --port 6006`); do not expose port 6006 publicly.

Before deleting a cloud instance, archive the entire run directory: `console.log` (when present), `config.json`, models, statistics, checkpoints, `best/`, TensorBoard events, Monitor/metrics files, and both `evaluation/` and `best/evaluation/`. `runs/` is git-ignored.

## Interpretation

Use nominal mean return from `final_model` as the primary outcome per seed. Report the paired per-seed difference `truly-ppo - ppo-reference` and both held-out conditions separately. Three seeds is a small sample: it cannot support significance claims, and one divergent seed dominates the mean and standard deviation. Prefer per-seed values and paired differences over a pooled aggregate. Any ranking is preliminary, and the Truly PPO rollback is not expected to guarantee a gain.
