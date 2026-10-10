# POPGym Arcade reproducible control protocol

**Status:** Official baseline execution validated, not a completed SOTA comparison.
**Parent report:** `REPORT_PHASE6_2026-10-10.md`.
**Source:** https://github.com/bolt-research/popgym-arcade (authors' package `popgym-arcade==0.0.6`).

## Why this matters

Prior cycles used short or action-independent card tasks, researcher-supplied latent models, or synthetic reward overlays. A candidate for consequential autonomous representation discovery must survive *action-dependent, heterogeneous, visually partially observed control*, under unknown changes and realistic interaction budgets. POPGym Arcade offers a genuine independent environment suite with a common five-action interface, pixel observations, memory benchmarks, and published recurrent/non-recurrent baselines. It does **not** automatically provide controllable intervention APIs, unknown change-point semantics or a known full causal model; those must be declared and evaluated separately.

## Environment and provenance

Run in the project's isolated `.venv_arcade` (Python 3.12; JAX 0.11.2; `popgym-arcade==0.0.6`, `memax==0.1.1` and its baseline dependencies). Recreate using `research/requirements_arcade.lock.txt`. Main project `.venv` is untouched. Driver 580.126.09, RTX 3060 Ti (8 GiB), JAX CUDA 13 plugin and runtime from isolated packages. No W&B upload; `WANDB_MODE=disabled`. Official package source is examined in separate temporary clone and called via public `make_train`.

**Source maintenance issue:** The baseline PQN_RNN implementation in installed 0.0.6 computes an action-churn diagnostic by comparing arrays of incompatible shape after its learning phase (the bug was not hit by the project's four-step smoke test but arose when longer training was attempted). We preserve a **one-line, diagnostic-only fix** in `arcade_pqn_rnn_diagnostic_fix.patch` that replaces the broken churn-ratio comparison with a constant placeholder. The learning loss, model architecture, actor, optimizer and update stream are unmodified. This removes meaningful churn measurement; do NOT interpret that field. Applying this patch to the installed package is necessary to reproduce results until the upstream implementation is fixed. Do not claim the upstream baseline is fully unmodified.

To prepare an isolated environment:

```bash
python3 -m venv .venv_arcade
.venv_arcade/bin/python -m pip install -r research/requirements_arcade.lock.txt
# One-line metric fix, applied from project root to installed package:
cd .venv_arcade/lib/python3.12/site-packages
patch -p1 < /home/dtdav/dev/learn/research/arcade_pqn_rnn_diagnostic_fix.patch
cd /home/dtdav/dev/learn
```

Actual training call (one GPU seed, 32,768 environment steps; a **pilot only**):

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
.venv_arcade/bin/python research/cycle43_arcade_official_baselines.py \
  --algorithm PQN_RNN --task CountRecallEasy --seed 43221026 \
  --total 32768 --parallel-envs 8 --rollout 32 \
  --minibatches 4 --epochs 2 --test-envs 8 --partial \
  --output research/cycle43_CountRecallEasy_PQN_RNN_pilot.json
```

For memory-free comparison, set `--algorithm PQN`. Other tasks include `NoisyCartPoleEasy`, `BattleShipEasy`, `NavigatorEasy`, `CountRecallEasy` (see exact names exposed by the installed package). Both algorithms are exercised with raw 128×128 observations, a shared five-action space, and periodic heldout environment rollouts using separate evaluation random keys.

## Evaluation criteria and safeguards

The script records algorithm, task, seed, interaction count, wall clock, memory type, hardware, training completed-episode statistics and **separate evaluation-return timeseries**. Its final heldout score is the official authors' test-return metric, which may be normalized differently from raw training episode returns; never compare the two scales directly. A one-seed / 32k-step pilot cannot support general performance or novelty claims.

A **decisive study** needs:
- At least three heterogeneous task families (one noisy observation control, one symbolic-memory task, one action-dependent navigation/game) and multiple heldout mechanism shifts.
- A minimum of five independent seeds per method/task condition, with paired test environments where physically possible, and full confidence intervals including training run variation.
- Identical observation/action/diagnostic privileges across learners. No hidden simulator state, correct-rule labels or privileged branch indicators to proposed methods; advanced Bayesian baselines may use declared additional prior structure only if labeled as privileged.
- Comparable tuning budgets, multiple standard policy/representation baselines (published PQN and recurrent PQN/LRU, PPO with recurrent or attention memory, PSR/reward-machine/counterexample-guided methods where meaningful), plus compute and inference costs.
- Scale training interaction budgets toward the original paper's million-step regimes; do **not** claim superiority or impossibility from 2k/8k/32k-step pilot checkpoints.
- Blindly held-out environments/change schedules, including nuisance-only drift, reward/transition-mechanism changes and tests where no permitted experiment can identify the policy-relevant latent distinction.
- Report cumulative *task reward minus interaction and diagnostic costs*, false alarm rate, change recovery delay, compute, memory usage and model complexity; not just predictive loss.
- A preregistered, falsifiable distinctive prediction separating the new mechanism from existing BOCPD, model-based RL, learned reward/transition machines, in-context recurrent RL and active experiment design.
- Strong stopping conditions: if an existing matched algorithm explains the performance, reclassify as integration or cost trade-off; no synthetic task optimization to manufacture novelty.

**Current state:** Infrastructure is available and pilot returns are recorded. A newly proposed algorithm has **not** yet passed the burden above. This protocol is a prerequisite, not an established research discovery.
