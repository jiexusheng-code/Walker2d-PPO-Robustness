# Walker2d PPO Robustness

A reproducible shared-profile comparison of actor-objective variants on Gymnasium's `Walker2d-v5`:

- **`ppo-reference`**: Stable-Baselines3 PPO with the standard PPO-Clip actor objective.
- **`truly-ppo`**: the same shared reference recipe with only the actor objective replaced by Truly PPO's KL-triggered rollback.
- **`rpo`**: the same shared reference recipe with only the update-time policy mean perturbed by RPO noise. It is not part of the default matrix; select it explicitly.

All variants use the same environment, observation/reward normalization, network, learning rate and schedule, rollout settings, value/entropy coefficients, seed, evaluation protocol, and step budget. The comparison is empirical; it does not assume any variant will win. Report every seed and every condition, including neutral or negative results.

## Shared recipe and the only planned difference

| Setting | Value |
| --- | --- |
| Environment | `Walker2d-v5` |
| Normalization | `VecNormalize` observations and rewards, clip 10, frozen during evaluation |
| Network | `[64, 64]` tanh policy and value networks |
| Learning rate | constant `5.05041e-5` |
| Rollout / update | `n_steps=512`, `batch_size=32`, `n_epochs=20` |
| Discount / GAE | `gamma=0.99`, `gae_lambda=0.95` |
| PPO clip | `clip_range=0.1` (used by the standard PPO and RPO actors) |
| Entropy / value | `ent_coef=0.000585045`, `vf_coef=0.871923` |
| Gradient clipping | `max_grad_norm=1.0` |
| Environments | `n_envs=1`, seed controlled per run |

The PPO hyperparameters follow the Walker2d entry in DLR-RM/rl-baselines3-zoo's [`ppo.yml`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml). That entry targets `Walker2d-v4`; this project ports it unchanged to `Walker2d-v5`, which has not been independently validated as an optimal recipe.

The only planned actor difference is:

- `ppo-reference`: PPO-Clip actor objective from Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347, 2017), as implemented by Stable-Baselines3.
- `truly-ppo`: Truly PPO's KL-triggered rollback with `alpha=5`, `delta=0.03` (Wang, He, Wen, and Tan, *Truly Proximal Policy Optimization*, arXiv:1903.07940, 2019; extended 2020 version). When `kl(old||new) >= delta` and `ratio * advantage > advantage` (the authors' strict form, equivalent to `(ratio - 1) * advantage > 0`), the actor term becomes `ratio * advantage - alpha * kl(old||new)`. This is a KL penalty **inside the actor objective**. It is not a parameter, optimizer-state, or checkpoint restore. The paper's constant offset in the non-triggered branch is omitted because it has zero gradient. This is an SB3 port, not a bit-exact reproduction of the paper's training setup.

The rollback is a paper port, not a guarantee of higher return, sample efficiency, or robustness to perturbed dynamics. Held-out conditions change friction and actuator strength together, so they measure joint dynamics robustness rather than isolating either parameter. The trigger condition follows the authors' reference implementation: [wangyuhuix/TrulyPPO](https://github.com/wangyuhuix/TrulyPPO).

- `rpo`: Robust Policy Optimization's update-time perturbation, following [CleanRL's `rpo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py). Rollout collection samples from the plain Gaussian, and deterministic prediction returns its mean. Only the log-probability used by the PPO update evaluates the original rollout actions under a Gaussian whose mean is shifted by elementwise `Uniform(-alpha, alpha)` noise; entropy and the saved old log-probability remain unperturbed. The default is `alpha=0.5`; any finite non-negative value is accepted. `alpha=0` skips the noise draw entirely and is exactly PPO-Clip. This is an SB3 policy-level port, not a bit-exact reproduction of the paper's training setup, and it guarantees no performance gain.

## Cloud run: default arms

Requirements: Python 3.11 or 3.12, `uv`, and a CPU-only Linux instance. This project is developed with uv 0.10.10; use that version or a compatible newer one.

```bash
git pull --ff-only
uv sync --locked
bash scripts/run_matrix.sh
```

Defaults: the two default arms (`ppo-reference` and `truly-ppo`), seed 0, 1,000,000 steps per arm, periodic evaluation every 50,000 steps (10 episodes, periodic RNG seed 50,000), then the final deterministic evaluation over `nominal`, `heldout_low`, and `heldout_high` with 100 episodes per condition and per-episode seeds `40000 + i`. Runs are sequential. The matrix writes to the new root `runs/two_arm/<variant>/seed_<seed>/`; the retired three-group history under `runs/main/` is left untouched. The startup line and each run's `config.json` prominently record the variant and `shared_profile=ppo-reference`.

Review seed 0 before paying for more seeds, then expand explicitly:

```bash
SEEDS="0 1 2" bash scripts/run_matrix.sh
```

To train the experimental RPO variant explicitly (it is never added to the default matrix), request it by name:

```bash
VARIANTS="rpo" SEEDS="0" bash scripts/run_matrix.sh
# Optional: RPO_ALPHA=0.5 (default) sets the uniform-noise half-width.
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

- `bash scripts/pilot_reference.sh` trains **only** `ppo-reference` and writes a timestamped directory under `runs/reference_pilot/`. It is a single-arm pilot; use `run_matrix.sh` for the shared-profile comparison.
- `bash scripts/cloud_smoke.sh` runs the unit tests, trains the two default arms for 4,096 steps, and evaluates them briefly under a timestamped `runs/smoke/` directory.
- `bash scripts/benchmark.sh` trains the two default arms for 50,000 steps and writes timestamped throughput measurements under `runs/benchmark/`.

All scripts use `uv sync --locked` and do not modify dependencies or the lockfile. They run sequentially; there is no scheduler or concurrency layer.

## Results and evaluation protocol

Each run directory contains:

- `config.json`: variant, shared profile name, actor objective, all shared PPO settings, normalization, evaluation settings, source hashes, package versions, status, and resume history.
- `final_model.zip` and `vecnormalize.pkl`: the primary result and its matching frozen statistics.
- `checkpoints/`, `best/best_model.zip`, `best/vecnormalize.pkl`, `tensorboard/`, `monitor.csv`, `metrics.jsonl`, and `evaluations.npz`: diagnostics and resume material.
- `evaluation/episodes.csv` and `evaluation/summary.json`: final deterministic per-episode and per-condition results.

The **final model** is the primary result. The final evaluator loads matching VecNormalize statistics, keeps them frozen (`training=False`, `norm_reward=False`), evaluates raw environment return, and uses the same base seed protocol (`base_seed + episode`) for every arm. `summary.json` records the model and statistics SHA-256 hashes, variant, actor objective, RPO alpha (for `rpo`), condition set, episode count, base seed, and determinism flag.

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

`--resume` without a value picks `final_model.zip`, otherwise the latest checkpoint, together with the matching statistics. It never silently starts a new run when there is nothing to resume. Resume validates the configuration before training: variant, seed, shared profile (`ppo-reference`), all shared PPO settings, normalization, and the actor objective (including `alpha`/`delta` for Truly PPO and `--rpo-alpha` for RPO). For RPO, omitting `--rpo-alpha` resumes the alpha stored in the checkpoint; passing a different value is rejected. Loading a non-RPO checkpoint as the `rpo` variant is also rejected. Legacy `baseline`, `ppo-normalized`, and old 2048-step `truly-ppo` bases are rejected with a clear error instead of being resumed into the new profile. `--timesteps` remains the absolute budget; a resumed run trains only the remaining steps.

Limits: ordinary `--resume` does not restore simulator state or its RNG stream, so it is a close continuation, not a bit-exact one. RPO's debug packages additionally save complete pre-update rollout buffers and optimizer/RNG state for **fixed-buffer update replay**; these are not simulator-continuation checkpoints. With `n_steps=512`, an interrupted rollout in flight may still be lost.

## Legacy historical models

The old three-arm training plan (`baseline`, `ppo-normalized`, `truly-ppo` on the old recipe) is retired. `train.py` accepts only `ppo-reference`, `truly-ppo`, and the experimental `rpo` variant. Existing historical runs and models are left untouched. A historical unnormalized baseline model can still be evaluated offline:

```bash
uv run python evaluate.py \
  --model runs/pilot/baseline/seed_0/final_model.zip \
  --variant baseline \
  --episodes 100 --seed 40000 \
  --output-dir runs/pilot/baseline/seed_0/evaluation
```

This is evaluation-only compatibility, not a third training group. Do not mix those results with the new shared-profile comparison.

## Training diagnostics

Keep the whole run directory. `metrics.jsonl`, `monitor.csv`, `evaluations.npz`, and TensorBoard events are diagnostic, not the primary outcome. Useful signals:

- PPO update health: approximate KL, clip fraction, entropy, policy standard deviation, losses, explained variance.
- Exploration: entropy and policy std should not collapse immediately.
- Rollout episodes: raw episode return, length, and healthy-timeout rate.
- Normalization: action/observation clipping fractions and observation RMS scale.

For `rpo`, `train/approx_kl` and `train/clip_fraction` are computed against the saved unperturbed rollout log-probability using the perturbed update log-probability. They therefore include the perturbation itself and can be nonzero before any parameter update. They are not an analytic KL between the unperturbed policies.

These are correlational. A spike or collapse is a reason to inspect a run, not proof of a cause. View TensorBoard through an SSH tunnel (`uv run tensorboard --logdir runs/two_arm --host 127.0.0.1 --port 6006`); do not expose port 6006 publicly.

Before deleting a cloud instance, archive the entire run directory: `console.log` (when present), `config.json`, models, statistics, checkpoints, `best/`, TensorBoard events, Monitor/metrics files, and both `evaluation/` and `best/evaluation/`. `runs/` is git-ignored.

## RPO replayable flight recorder

RPO training through `train.py` automatically records **every rollout-level training update** under `debug/updates/` (one package per 512 environment steps with the shared recipe); it is not limited to periodic checkpoints. Each package covers all 20 epochs / 320 minibatch optimizer steps, with one measurement row per minibatch. It keeps pre/post models (including Adam), the full pre-update rollout buffer, Python/NumPy/Torch RNG state, matching VecNormalize statistics, full-buffer unperturbed distribution probes, update-aligned diagnostics and actual minibatch measurements. Manifests record hashes, status and partial failures. Recording failures stop training rather than silently leave missing evidence. No automatic retention/deletion policy is applied.

`debug/transitions/` stores compressed per-rollout traces: raw and normalized observations, raw Gaussian and clipped environment actions, old values/log-probabilities, raw and normalized rewards, episode boundaries, terminal observations, reward components and MuJoCo qpos/qvel. The normalized reward here precedes SB3's timeout bootstrap; the saved update buffer contains actual training rewards and GAE targets. Done-step next observations/qpos/qvel follow DummyVecEnv auto-reset and are explicitly labelled; terminal observation is separate, and its inverse-normalized reconstruction can lose information if VecNormalize clipped it. Environment XML, dynamics arrays and installed package versions are saved at each training segment start.

`debug/evaluations/` keeps the exact model/statistics pair and per-episode records at every periodic evaluation, not just the best. Periodic evaluation is during rollout, before the following optimizer update; records include `_n_updates` to disambiguate timing and use the existing fixed evaluation RNG stream (not `seed+i`).

After training, verify an update **on the same software/platform and CPU thread configuration**:

```bash
uv run python scripts/replay_rpo_update.py \
  runs/rpo_screening/rpo/seed_0/debug/updates/update_000000 \
  --trust-local --output runs/replay_update_000000.json
```

Only use `--trust-local` for your own training packages: model/buffer/RNG files contain executable pickle data. The verifier checks file hashes first and requires exact policy and Adam state equality after replay. It refuses to overwrite its output. A hash is an integrity check, not authentication of an untrusted package.

If performance deteriorates, preserve the whole run: locate the first degraded periodic evaluation, identify its preceding recorded updates, replay those exact buffers, then compare unperturbed KL/ratios, perturbed ratios, advantage signs, value fit, component gradients, gradient clipping and actual parameter steps. Test one proposed mechanism at a time on the saved data before proposing another long run. A fixed-buffer counterfactual diagnoses a local update; complete training-seed confirmation still requires actual experiments.

This recorder costs disk and runtime. A local CPU smoke using the full 512-step / 20-epoch recipe wrote 320 minibatch records, about **1.86 MB per update package**, plus about **0.32 MB per rollout trace** and **0.07 MB per periodic evaluation snapshot**. Multiplying by `ceil(1_000_000 / 512) = 1954` gives roughly **4.3 GB of debug data per million-step run** before normal checkpoints, TensorBoard, and other run artifacts. These are planning estimates, not size caps; reserve headroom. A separate local timing over two full-recipe updates (1,024 steps, including traces but excluding periodic evaluation) took 3.09 s without recording and 7.24 s with recording, about **2.35× runtime**. This single-machine smoke measurement is not a cloud throughput forecast. Summary quantiles describe the distribution of the recorded minibatch statistics, not pooled per-transition quantiles. Component-gradient probes run on the first minibatch of each epoch, not every minibatch. Do not delete old packages without explicit approval. `runs/` remains git-ignored and must be archived separately before deleting the server.

## Interpretation

Use nominal mean return from `final_model` as the primary outcome per seed. Report the paired per-seed difference `variant - ppo-reference` (for example `truly-ppo - ppo-reference`, or `rpo - ppo-reference` if you ran RPO) and both held-out conditions separately. Three seeds is a small sample: it cannot support significance claims, and one divergent seed dominates the mean and standard deviation. Prefer per-seed values and paired differences over a pooled aggregate. Any ranking is preliminary, and neither the Truly PPO rollback nor the RPO perturbation is expected to guarantee a gain.
