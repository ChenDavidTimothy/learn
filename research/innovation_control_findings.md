# Innovation response memory: actual control screen rejected

2026-10-21. **TESTED NEGATIVE.** Frozen protocol/source checkpoint
a1a3e24d70eec2ea30c370e3a8de73a8da6c2ff1 was SSH-pushed and remotely verified
before the four fits. The independent final result check passes. This is an
exploratory failure of a specified construction, not a new learning framework,
a general impossibility result or a v18 milestone. The ultimate Goal is active
and unachieved. This formulation is retired without a test-informed rescue.

## What actually learned and what was supplied

The candidate accumulates a 2x2 cross-moment of finite observed acceleration
with independent action probes during its first32 physical interactions. A
6,085-parameter policy/value network is trained by PPO to use that memory and
ordinary observations for three consecutive joint goals, without goal resets.
No inverse controller, true body parameters, model gradients or desired-action
labels are supplied to this network. Shared exact joint q/v, fixed statistics,
probe schedule, architecture and task cost remain substantial supplied priors.

The four fits use7,864,320 training interactions in total. The selected candidate
and full-history GRU each use1,966,080 training interactions at rate .0003 and
checkpoint128, selected by32validation bodies from both rates and four checkpoints.
The GRU has26,629 parameters. They are homemade screening learners, **not
demonstrated competent modern meta-RL baselines**. Correct replay/BPTT does not
establish effective optimization. RLS has no meta-training; it identifies online
from ordinary transitions, with the first32 calibration actions charged inside
the same240-step control episode. Its PD/inverse-control operation is supplied.

The plant remains the existing nonlinear horizontal two-link point-mass body.
New body masses, lengths, damping, gains and signed actuator permutations are
hidden. This calibration problem favors the moment bias. Exact q/v and every
future goal/time are supplied to all methods; no camera/body-dimension/general
robotics transfer is claimed. Each fresh split has32independent bodies, three
goals/body and12seconds/body. OOD shifts continuous physical coefficients;
signs and actuator permutations vary in both training and testing.

## Fresh results and frozen decision

Mean per-step body cost includes joint error, velocity and issued effort.
Precision requires both joints within .15rad and .15rad/s at every one of the
last20samples of a goal. All failed windows are retained.

| Method | Fresh ID cost | ID precise goals | OOD cost | OOD precise goals |
|---|---:|---:|---:|---:|
| Moment policy | .8607635317 | 2/96 | .9011114997 | 0/96 |
| Full-history PPO-GRU | .8697117941 | 2/96 | .9100885205 | 0/96 |
| Same moment policy, H zeroed | .8607429993 | 2/96 | .9011621217 | 0/96 |
| Conventional adaptive RLS | .2668823110 | 42/96 | .2890214531 | 33/96 |

ALL continuation gates fail on BOTH splits: absolute competence,10%cost
advantage over each rival and10%advantage over zero-H. Small cost advantages
over the GRU do not compensate for absent capability. Zero-H changes mean cost
by about2e-5/5e-5, so the trained construction demonstrates no useful dependence
on its proposed persistent memory. This does not prove the statistic useless.

RLS has roughly one-third the learned policies' costs without pretraining,
but also fails the declared75%ID/50%OOD precision thresholds. It does not fully
solve this task. No reliable control, modern SOTA competition, learned exploration,
compositional skill or general-learning claim follows from this screen.

## Actual resources and scope of verification

Four complete fits:1,966,080steps each,55.149/54.906sec for moment settings and
113.163/113.235sec for GRU settings. They ran concurrently on local CPU, so the
sum336.45sec is not elapsed campaign time. Training began~02:26UTC; selection
and fresh scoring completed~02:29UTC. This bounded control investigation reached
a first result~22minutes after its direction review began; no extra training,
hyperparameter rescue or simulator expansion is warranted.

Recorded study costs:7,864,320training +138,240validation +61,440fresh-test
interactions =8,064,000 /403,200simulated seconds. Three discarded engineering
one-update timing executions add46,080train interactions /2,304simulated seconds,
separate from the four fits (one completed moment update lost timing output to
an inline JSON print typo; the corrected fixture ran both methods). Engineering
checks and independent integrator evaluations are also separate numerical work.
Real hardware experience would not be free merely because simulation was fast.

The independent mathematical critic checked conditional Stein and Gaussian score
identities, a current-probe leakage counterexample, production GRU whole-sequence
replay/gradients and the first32probe/memory/evaluation timing using synthetic
data/mocked dynamics. The instrument author checked the torque map, six bounded
DOP853 flows (max6.34e-10 versus1e-7) and RLS/goal clock fixtures. These do not
independently verify all physical training transitions.

The separately authored final checker reconstructs costs and precision directly
from saved states/actions/goals for all26validation/test archives and199,680
transitions. Maximum cost discrepancy3.55e-15; all precision flags exact. It
checks paired bodies/goals, all16weight hashes, four reports, validation minima,
source provenance, interaction arithmetic and all gates. It does NOT regenerate
training optimization, network forward passes, complete physics or global optima.
The earlier fixture report is preserved separately from the final screen report.

## Corrections, prior work and next scientific decision

Before any substantive fitting, the math critic identified a confound in permanent
probing versus RLS's finite calibration. The frozen plan records the prospective
change to32probes followed by held memory and deterministic actor evaluation.
No test outcome selected that correction. No frozen source changed during fits.

**EXISTING RESEARCH:** response correlations, IV/Stein identities and plastic
outer-product memory are established tools. [Differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a/miconi18a.pdf)
Section2/Equations1–2 is a close memory precedent. [DEP-RL](https://arxiv.org/pdf/2206.00484)
Sections4.1–4.2 embeds sensor–actuator correspondence in its inverse map and
uses sensorimotor correlations for exploration. Estimating effects through our
known probes is a narrow distinction, not a foundational novelty claim.

The independent direction critic recommends retirement and a change in research
decision: **can a faithful maintained conventional history-based learner acquire
competent calibration and tracking on this unchanged task under a bounded budget?**
Both weak PPO learners, large remaining value errors and tiny late policy KL make
training adequacy a serious unresolved explanation; they do not diagnose one
optimizer cause. Another memory head would currently be judged through an
unproven learning procedure. Establish competence first, then identify a
consequential remaining limitation before proposing another mechanism.

[AMAGO's actual Section4](https://arxiv.org/html/2310.09971v3) supplies a serious
established account of off-policy sequence learning, actor/critic sharing and
loss scaling. Its maintained code/dependencies are being checked for feasibility;
no implementation is yet adopted or claimed reproduced. That prerequisite study
must use training/validation and later fresh confirmation, not recycle this test
for tuning or revive H. Do not widen the physics benchmark or start another
architecture menu. Previous prediction screens remain retired.

Reproduction: innovation_control_plan.md contains the complete frozen definition;
innovation_control_screen.py has fit/select/evaluate CLI modes. All four complete
fit directories retain reports, losses, selected/unselected weights and validation
trajectories. innovation_control_selection.json and innovation_control_results.json
freeze choices and decisions; all actual test trajectories are saved. Run
check_innovation_control_results.py for the bounded independent arithmetic check.
