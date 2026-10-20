# Active continuation after v14

The general-learning Goal is ACTIVE and unresolved. Latest independently verified
ledger: first_principles_robot_learning_ledger_v14.md, including complete v13 and
all earlier history verbatim. V13 SSH publication verified at
`aeb569a16688c7290af88169d654c0f809b6c2a6`; v12 at
`ace6041069d246656caa3bcffb3239db93a1bfa2`. Resolve current release commit via
`git log -1` and SSH origin/main (a document cannot contain its own hash).
Continue on the existing PC/.venv; a release is never a stopping condition.

## V14 findings and exact evidence

Generic periodic RBF relations learn coupled joint displacement from SAME noisy
rest-pulse data without true inertia/mass formula. Both encoder contexts supplied.
Broad acquisition supports new goals/poses; narrow acquisition extrapolates badly.
Inverse versus forward parameterization changes function/noise/conditioning: equal
features are not equal physical-function classes. No universal model necessity.

Canonical broad task MSE: constant .00898236, context forward3.18255e-5,
naive inverse8.98605e-6, calibrated inverse4.52803e-6. Eight acquired-data clusters
per condition. Broad naive-minus-calibrated paired difference4.45802e-6+/-1.08175e-6.
Narrow means .0194882/.0548094/.0144884/.0147816; inverse correction difference
-.000293217+/-.000602516 is unresolved. Zero-command analytic score .0145;
narrow inverse aggregate transfer is near that uninformed reference. Do not carry
over the archived comparison's stronger narrow harm claim to canonical results.

Known preparation pose/rest, ideal calibrated torque, angle topology, sensor law,
short duration and torque-affine finite basis remain supplied. No continuous
control/stopping, unknown-state inference or discovery of an observation language.
End rest calibration here; no more kernel/optimizer variants merely to improve it.

Source/design/results/NPZ/timing: coupled_learning_v14*. Independent theory:
coupled_learning_v14_theory_independent.md, v14_design_independent.md, literature
note. Artifact audit coupled_learning_v14: independent Hamiltonian RK4 at32/64steps,
all10240 acquisition+2048 predictive+12800 canonical task pulses,832 candidates,
PSD/ridge stationarity, validation/selection,10240 bounded decisions checked;
zero solver failures, box-optimum gap1.403e-12. Main/NPZ/audit reruns byte-identical.
Q1 symmetry is only approximate; both pooling definitions/results retained.

## Preserved information-use correction and development budget

Original coupled_learning_v14_two_reading_pilot* and corresponding audits retain
full first comparison. It used(o1-o0)/F although q0 was known exactly. Audit caught
avoidable initial sensor noise. Canonical ALL-method repair uses(o1-q0)/F,
variance .0004I versus original .0008I. No grid/goal/condition change; exact raw
records reused,160 nonresponse arrays bitwise identical and32 response arrays
recomputed. No new acquisition/predictive/oracle search. This is exploratory after
original outcomes, not a new blind benchmark. Both comparisons remain reproducible.

Total unique acquisition10240rest episodes, predictive2048. Each comparison tests
12800episodes; development total25600. Original oracle feasibility1456 simulator
rollouts once;160 task witnesses max residual3.24e-13 reused. All832 fits per
comparison=1664 preserved. 49152 acquired/predictive coordinate sensor readings,
102400 task readings. Preparation/reset capability and computation are separate
resources, not ignored hardware costs. No actual hardware experiment.

Original source is archived byte-for-byte; reproduce it with --output-prefix to
avoid overwriting current canonical results. Archived generated metadata names
refer to filenames at generation; archived audit resolves frozen paths. Canonical
main loads archived raw NPZ/oracle results; those files MUST remain present.
Dormant legacy experience() in canonical source still computes the old two-reading
statistic, but main never calls it. Do not reuse that helper with the repaired
covariance without explicitly changing the data model. See information_correction.

Mechanics pilot/audits are separate: coupled_context_v14_pilot, independent
Hamiltonian/Radau and short-pulse checks.179 unique episodes/21.025 seconds plus24
tighter reintegrations. Leading continuous displacement .5T²M^-1u+O(T4); special
poses0/pi hide quartic order. Finite coupling atpi is not exactly zero. Shared
noisy context and displacement have -R covariance; pilot trains no learner.

## Immediately next: relevant history in continuous coupled motion

Remove rest preparation. Question whether current positions/commands suffice for
the desired consequence, and how experience can reveal/revise an inadequate summary.
The same pose with different motion generally needs different actions. Derive the
simplest decisive aliasing/feasibility example before adopting a state architecture.
A history-based learner may act without naming velocity; a supplied physical state
is one competitor, not a mandatory foundation. Compare strong conventional
history/derivative-free methods, not only a deliberately under-informed baseline.

A promising first pilot: same two-link current pose, equal-prior opposite initial
velocities, same goal and bounded torque opportunities. Current-pose-only input is
identical, but a past-position history differs. Verify actual bounded commands can
meet both tasks separately; do not confuse infeasibility with missing information.
Avoid asserting a global finite-time lower bound from a grid/local optimizer.
Then construct actual learning/decisions in continuous coupled motion or an
experience-based summary-revision mechanism. No new release solely for a pilot.

Primary relevant baseline read: Romeres et al.2018, derivative-free inverse dynamics
using learned position-history features on iCub-arm data, arXiv1809.05074. This is
far stronger than rest calibration; do not claim reproduction or superiority.
Full literature links and limits are in coupled_learning_v14_literature.md.

For each meaningful verified milestone: independent audit, next sequential ledger
with complete v14 verbatim, executable actual artifacts, commit/push SSH/verify,
and immediately continue. Do not mark the broader Goal complete or blocked merely
because an execution cycle ends. Preserve precise continuation if interrupted.
