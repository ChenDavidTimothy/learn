# Active continuation after v13

The general-learning Goal is ACTIVE and unresolved. Latest verified living ledger:
`first_principles_robot_learning_ledger_v13.md`, with complete v12 and all earlier
history preserved verbatim. V12 SSH commit verified at
`ace6041069d246656caa3bcffb3239db93a1bfa2`. Resolve current release commit via
`git log -1`/SSH origin main (a document cannot contain its own commit hash).
No release is a stopping condition. Continue on the owner's PC/existing .venv.

## V13 verified bounded diagnostic, now CLOSED as an optimizer branch

A single prescribed exact-gradient fit of the SAME six-coordinate direct controller
on the32 existing v12 learned nonlinear models, using three fixed starts and
separate16-episode virtual validation, improves mean tracking cost on all4plants.
Fresh streams compare physicalCEM/virtualCEM/gradient/MPC. Means for gradient are
.265958/.275643/.306701/.289054 vs physicalCEM.313822/.330627/.402863/.352047;
MPC.266410/.271958/.266985/.263184. This constructively reveals a finite CEM search
limitation in the SAMEfamily. No new plant TRAINING. Extra learnedmodel computation
and supplied vocabulary remain; no universal architecture or optimality claim.

CRITICAL NEGATIVE: stationkeeping gradient1/64 vsphysical14/64,virtual16/64,MPC25/64.
62/64gradient trials violate velocity tolerance. Mean positioncost gains dominate
while velocitycost/effort rise; pooledlatepositionRMS nearlyunchanged, latevelocity
RMS.1107→.1606. Better average cost is not uniformly improved capability. Training
process0/testprocess.03, finitecost/event distinction, class and optimizer remain
separate possible causes. Do not add another scalaroptimizer to rescue this result.

Source/results/design: gradient_reuse_v13.py, JSON, NPZ, timings. One of96fit
attempts reachesmax160 (plant3rep7generic), retained and NOTselected. All32chosen
outputs are optimized (8/11/13starts). AllfreshMPCsolvesconverged, notglobalguarantees.
Optimizerrollouts6595168 virtualsteps;validation220160;replay88064;checks50912;
freshtests22016plantsteps/256resets. No hiddenplantreward in selection.

Audits: audit_gradient_reuse_v13.py/JSON/report independently reconstruct96fit
endpoints,160validationcandidates/choices,1024retainedvirtualepisodes,256tests,
5504MPCplans, hashes/budgets/pairedSEs. Main/NPZ and audit rerunsbyteidentical.
Theory gradient_reuse_v13_theory_independent.md, independentPyTorchAD
 audit_gradient_pathwise_v13.py/JSON: canonical16episodeJacobian1.1e-13agreement.
One initialcoarseFDcheck crossedaclippingboundary; preserve correctionmd/checktable.
Two smallerfixedstepswithunchangedmaskspass. IndependentADconfirmskink, and records
an extremehighsensitivityfloatingpointstresslimit honestly. 37checkrolls counted.

## Immediately continue beyond scalar gain tuning

Nextquestiondraft: coupled_context_v14_question.md. Investigate how experience can
learn a change in what an action means when articulated-body configuration couples
actuators, with NEW desired displacements/poses. Do not turn this into merely
identifying named masses or proving another abstractrankidentity.

Start simplestrest-pulse2linkhorizontalarm, then escalate towardcontinuouscoupled
motion. Draft initially uses semiimplicitEuler map dt²M^-1u; actualpilot should use
continuousDOP853 and derive displacement=.5dt²M^-1u+O(dt4) atrest/nofriction. Avoid
mistaking discretization artifact or suppliedcosinefamily for generallearning.
Theoryagent is independently criticizing this direction and derivingmechanics.
Compare strongjoint/context-awarekernelresponse, directinverse/retrievalalternative,
and simpleconstantresponse; exactphysicalfamily may only be privilegeddiagnostic.
Actualobservations/actions only to ordinarylearners, context/preparationnoiseexplicit,
feasibleboundedtargets, sameacquisitionbudgets. Firstpilotphysicsvalidity is NOTa
completedlearningmilestone. Donotpublishv14 until meaningfulauditedevidence.

For publication: nextledgerincludesv13verbatim; executableexperiments/actualresults,
independentaudit, SSHcommit/push/remoteverify, immediatelycontinue. GoalremainsACTIVE.

## Retained v12 findings needed for continuation

## What v12 actually establishes

- With the declared lagged timing, fresh iid zero-mean position noise and exogenous
  fixed goals/horizon/normalization, observation-only quadratic position/displacement/
  command fitness differs from hidden position/velocity cost by expectation offset
  1.6R/D. Biased/colored sensors or motor-first feedback can break policy ranking.
- Direct CEM improves a supplied six-parameter dynamic stationary controller.
  Shadow segmented raw-position OE followed by EKF/MPC gets lower mean tracking cost
  at both data checkpoints on all four supplied scalar plants from EXACTLY the
  same complete environment experience. This is a comparison of specified mechanisms,
  not equal priors/capacity/compute or universal model necessity.
- Feedback training OE is valid joint/sequential likelihood at zero process noise,
  known zero resets and all episodes retained. Do not claim full-command exogeneity
  or concatenate separate reset episodes. Do not fit only elites.
- Identical CEM inside the learned nonlinear model gives no consistent benefit to
  the SAME controller family. Capacity versus finite search remains OPEN. Learned
  linear and nonlinear MPC remain close; station keeping is unreliable.
- MPC uses remaining86-t; the direct class lacks a clock. Prescribed first70-step
  scores retain the MPC advantage before any horizon shortening, but class/compute
  differences remain. Feedback continues through86; this is not v10/v11 forcedcoast.

## Reproducibility and independent evidence

`shared_experience_v12.py`, NPZ, JSON, design, timing:32 training fits, four plants,
eight seeds each, all384 episodes/33024 training steps per seed. Checkpoints2/8 CEM
generations8256/33024 steps. Final direct means.314099/.331481/.401574/.357066;
nonlinear MPC means.267673/.274189/.266358/.274551. Main test384 episodes/22016plans.
`model_reuse_v12.py`, NPZ, JSON, design, timing:33024 VIRTUAL steps per seed, no new
plant training,64 extra test episodes; same-family costs.321943/.311855/.393994/.406712.

`audit_shared_experience_v12.py`/JSON/report independently reconstruct ALL24576
corrected physical+virtual training episodes,448 tests,22016 plans,384 fit records,
CEM draws/selections, seed domains, fits/Jacobians/EKF/metrics and clustered SEs.
No main implementation imported; eliminated dynamics independently implemented.
Audit passes: states<4e-13, tests<2.1e-14, CEMcoordinates/actions exact, summaries
exact. Main/reuse reexecution yielded byte-identical NPZ+JSON, independent audit
reexecution identical too. Raw metadata says auditpending because it records
pre-audit generation; the separate completed audit certifies it. Do NOT edit that
metadata casually: reuse JSON hashes the exact shared JSON bytes; audits verify it.

Independent theory/design: `audit_observation_loss_v12.md` and
`v12_comparison_design_independent.md`; older PD pilot and audit preserved.
First large run reused some adjacent-replicate noise seeds; caught before viewing
outcomes. All excluded source/data/compressedJSON/timing are retained as
`shared_experience_v12_seed_overlap_pilot*`. Never infer independent SEs from it.
Corrected roles globally disjoint; virtualdomain+1e10; test streams shared only for
intentional paired comparisons. Exact budgets/limitations are in ledger/design.

