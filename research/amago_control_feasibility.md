# Bounded prerequisite: maintained learned control competence

2026-10-21, started~02:34UTC after publishing the retired innovation screen at
374500e479748a672916013a9427ae9b11fe7cc3 (SSH remote verified). The next decision
is whether a maintained conventional history learner can acquire calibration and
tracking on the unchanged task. This is baseline research, not a rescue of H or
a new mechanism claim. Do not reopen the completed test or change its source.

AMAGO Section4 and current official code are the selected existing account to
check. Exact upstream main commit54c25ab6da9371614c47352f569a56c0fe938d3b was
read through Git SSH. Its core source/license/examples total~470KB, so a pinned
in-repository source copy is practical; no separate project or Git checkout is
needed. Optional benchmark environments/FlashAttention are unnecessary here.
Use the official configurable GRU trajectory encoder initially; this is not the
full Transformer result from the paper. No claim of competitive SOTA follows.

Preserve canonical .venv/bin/python and installed Torch2.14.1/Gymnasium1.4.0.
Upstream requires Gymnasium<=.29.1, so its compatible Gymnasium0.29.1 will reside
in a task-specific compatibility package directory INSIDE the existing .venv,
prepended only by the AMAGO process. This is not another Python environment and
does not downgrade the canonical import. Add only required small core support
packages after dependency resolution confirms no existing package replacement.
Disable external logging; do not start W&B runs/services or upload trajectories.

Bound feasibility/integration to~20minutes (~02:54UTC): source/dependency import,
ordinary-observation batch wrapper, one official forward/update and measured
runtime. If this cannot be achieved, preserve the specific obstacle and reconsider
implementation choice; do not spend hours maintaining research infrastructure.
No expensive fit until a concrete budget/competence criterion and train/validation/
fresh-confirmation separation are frozen. Do not derive claims from a smoke test.

Use the unchanged innovation_control_env.py mechanics, body distributions,
goals/cost/precision and ordinary information. No H feature. The maintained
agent may use its own counted exploration; no oracle/body labels or simulator
gradients. Offer full future objective. Check training competence, then independent
validation, then fresh physical draws if warranted. The prior held-out screen is
closed and cannot select this baseline's configuration.

Primary implementation and method:
https://github.com/UT-Austin-RPL/amago
https://arxiv.org/html/2310.09971v3
https://github.com/UT-Austin-RPL/amago/blob/54c25ab6da9371614c47352f569a56c0fe938d3b/pyproject.toml

Feasibility outcome~02:49UTC: pinned core imports; processcompatibility andall
pre-existing package versions preserved; official two-epoch/four-batch smoke and
fullB16 three-batch runtime profile pass. Terminal/interface critique passes.
Fullprofile2.20sec, peak1.37GB. Prefitserialization-view warning corrected using
Accelerator.save_state(safe_serialization=False), tiny CUDA GRU round-trip exact.
No expensive fit yet. Frozen competence plan is amago_control_plan.md.
