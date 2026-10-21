# Frozen prerequisite: official history-based learning competence

2026-10-21. This is a bounded established-method study, NOT our proposed new
learning mechanism and NOT a rescue of the retired innovation memory. Its
decision is whether conventional learned control is competent enough on our
unchanged instrument to make future learning-operation comparisons informative.
The earlier frozen control test is closed. No new ledger is justified by this plan.

## Existing account, information and actual update

Use the unmodified official AMAGO core at upstream
54c25ab6da9371614c47352f569a56c0fe938d3b (MIT license preserved). Adopt its
configurable GRU128, one recurrent layer, timestep MLP128/two layers, normalized
inputs. The official Agent provides replay-based actor–critic learning, PopArt,
multi-gamma values, four critics, target updates and filtered behavioral cloning
with its current defaults. Do not call this the full Transformer paper result.
Inspect the actual core/operative config; no homemade loss reproduction is used.

Ordinary observations: sin/cos q, v/2, all three future goals, phase, episode time
remaining, previous issued action and finite measured acceleration/10. The official
RL2 input additionally supplies previous action/reward. No H, probes, secret body
parameters, derivative labels, oracle actions or simulator gradients. Its own
sampled actions provide counted exploration. No compulsory32-step calibration or
externally fixed probe policy. All three goals/times remain offered.

Reuse innovation_control_env.py unchanged: same12sec,240steps,3goals/80steps,
cost/precision, hidden signs/permutation/gains, body distribution and OOD ranges.
The batched adapter changes only the ordinary interface/official automatic reset.
It records terminal trajectories before resetting the next body's observation;
official episode masks reset recurrent context. Neither hidden audit metadata
nor old terminal/new body labels enter policy inputs. Agent core is unmodified.

## Frozen resources and selection

Two independent settings: official learning rates .0001(default) and .0003,
same declared seed20261021 and matched training body/goal draw sequence. Each gets
128epochs,64batch bodies/240steps per epoch =1,966,080physical training interactions.
Per epoch50official replay minibatches of16full241-timestep sequences;6,400gradient
minibatches/fit, with the exact sampled/padded count retained in official logs.
Keep all collected replay within the9,000trajectory cap (8,192episodes/fit).
Initial replay warmup/default500LR-warmup gradient steps and all official Agent
loss/target settings remain as operative config records. No task curriculum.

Validation:64fixed independent bodies, deterministic mean actor, checkpoints after
32/64/96/128completed training epochs. Select ONLY by mean validation cost across
both rates/checkpoints. Fresh validation seed base2026102181; training base
2026102180 with batchk seed=base*1000+k. The four-body engineering smoke used
the beginning of validation, which is selection data, never confirmation. Smoke
weights are discarded. The held-out data of the retired innovation screen is
never used to select these settings.

If the checkpoint selected by MEAN VALIDATION COST has precision <75%, close this budgeted competence attempt without
spending interactions on confirmation. If >=75%, freeze selected weights/config,
then generate64fresh ID and64fresh OOD bodies with adapter seed base2026102182,
split-specific RNG roles (fixed seed per split). Score actual control and compare
to the same conventional RLS reference, forgetting selected on the NEW validation
set from .98/.995. Its32calibration actions remain counted. Do not tune on fresh
confirmation. Competence requires>=75%ID and>=50%OOD precise goals and mean cost
no worse than selected RLS on both splits. This is an early competence criterion,
not statistical proof or a general framework/competitive-SOTA claim.

After either failure, preserve all results and the specific unresolved training
obstacle; no automatic LR/network/normalization/budget rescue. After success,
identify a consequential limitation of the competent existing learner before
proposing another operation. Do not infer the innovation statistic is useless
from either outcome and do not resurrect H from these results.

## Runtime, environment and checks

One engineering smoke completed two official epochs/four gradient minibatches
in~3seconds on the local GPU, using four training bodies/epoch and a discarded
four-body validation batch. All ordinary episode interfaces/auto-reset timing
were checked through the official wrapper stack. This establishes executability,
not control competence. Complete full-batch runtime sampling before committing
to long training when feasible. Budget up to45elapsed minutes for BOTH fits;
report measured speed early and retain honestly unfinished work if the budget
cannot be met. Do not silently shorten only a rival or call partial work a success.

The canonical .venv/bin/python remains;37small core support dependencies were
added only after the resolver confirmed zero existing distribution replacements.
The before/after manifest verifies all pre-existing versions unchanged. Official
Gymnasium.29.1 is in .venv/amago_compat and is prepended ONLY in this comparator
process; canonical Gymnasium1.4.0 and Torch2.14.1 remain. W&B is disabled: no
external logging/service or upload. FlashAttention is unused with the official
GRU. TORCHDYNAMO_DISABLE=1 selects eager execution of the same official operations,
avoiding compilation infrastructure; no core algorithm source is patched.

Capture all fit losses, actual per-body control metrics, selected/unselected
weights, validation trajectories and official optimizer/RNG checkpoints. Keep
ordinary replay on the owner PC under existing ignored local_datasets; do not
delete it at run completion. Freeze source hashes before fitting. Independent
critique must challenge input leakage, terminal semantics, counting, checkpoint
selection, actual competence and the limited scope of this baseline study.

PREFIT integration notes: exact fullB16/L241 profile completed one64-body epoch,
three official minibatches and64-body validation in2.20sec, peak GPU allocation
1.37GB. Weights are discarded; this is runtime evidence, not a competence result.
Official Accelerate safetensors output warned that CUDA GRU's distinct flattened
parameter views were dropped as shared tensors. Before fits, checkpoint I/O was
changed to the same official Accelerator.save_state with safe_serialization=False;
raw torch weights retain all parameters. No core learning operation changed.
The precision thresholds are NOT proven physically attainable on all torque-
limited goal transitions. Failure only establishes failure of this budgeted
configuration, not intrinsic inadequacy of maintained learning or impossibility.

Primary methods read: AMAGO Section4 and official code, not just abstract:
https://arxiv.org/html/2310.09971v3
https://github.com/UT-Austin-RPL/amago/tree/54c25ab6da9371614c47352f569a56c0fe938d3b
