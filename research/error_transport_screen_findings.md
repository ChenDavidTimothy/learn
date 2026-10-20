# First mechanism screen after v17: joint error transport rejected

2026-10-20. **TESTED EXPLORATORY NEGATIVE.** No v18 ledger or competitive learning
framework is claimed. The general-learning Goal remains active and unresolved.
This bounded candidate branch is retired; no additional training, architecture
changes or controller rollouts are justified to rescue it from these test results.
Independent reconstruction of the decisive statistics passes. See
check_error_transport_screen_results.json and its scoped report.

## Candidate and hypothesis

The candidate learns how an experienced prediction error changes consequences
elsewhere: a residual-conditioned signed 2x2 operator applied at three fixed
functional update iterations. Disjoint same-body query improvement trains the
update, rather than hand-supplying v17's similarity distance. This is a concrete
learning operation, but the broad principle is already present in MetaFun (2020).
The distinguishable hypothesis was a useful advantage from jointly conditioning
output mixing on query, support input and support residual.

An independent critic identified direct prior art before fitting and required a
matched separable functional update. That simpler comparator has learned scalar
query/support similarity and nonlinear local residual processing with cross-output
mixing. Both have the same base/encoder/three iterations and zero-residual/no-evidence
constraints. ALPaCA predictive-NLL adaptation and CoDA whole-network contextual
adaptation are stronger competing explanations, with their scoped reproduction
and training-loss differences explicitly documented.

The prior candidate of command-centered risk was rejected before implementation:
it does not itself change the information identified or provide a transfer
mechanism beyond competent residualized regression. We did not build a loss-only
framework or run old all-record/fork diagnostics.

## Frozen screen and actual outcome

One nonlinear controlled two-link family: 32 training, 8 validation, 8 held-out ID
and 8 out-of-range bodies. Masses, lengths, damping and actuator gains/coupling
vary; both passive motion and command response change. Eight 96-step trajectories
per body give 43,008 physical intervals / 2,150.4 simulated seconds, generated in about .93 seconds.
Exact observed q/v and finite observed delta-v/dt labels are supplied in this
noiseless screening instrument. Hidden physical parameters never reach learners.
This is cross-body parameter transfer, not another camera-interface improvement
or a general robotics demonstration.

Each method receives the same sampled train-body episodes. Support comes from
one continuous trajectory; query labels come from different trajectories. Fixed
validation/test adaptation uses trajectory 0 prefixes; query trajectories 1--7
never supply adaptation labels. Train-only normalization, 2 global learning rates,
2,000 gradient steps per fit, one discovery training seed. Validation chooses
checkpoints, global LR and ordinary adaptation LR. No test values select a model.
CoDA/pooled train support+query errors, while transport/ALPaCA use query-after-support
objectives: same sampled information, algorithm-dependent optimization.

Mean standardized coordinate MSE over eight held-out physical bodies:

| Mechanism | ID:16 observations | ID:64 observations | Out-of-range:16 | Out-of-range:64 |
| --- | ---: | ---: | ---: | ---: |
| Candidate joint error transport | .037175 | .026375 | .750616 | .640077 |
| Matched separable functional update | .033909 | .023653 | .677843 | .604399 |
| ALPaCA learned-prior core | .016837 | .014850 | .459555 | .378381 |
| CoDA contextual core | .024671 | .019824 | .359611 | .306448 |
| Pooled network + ordinary fine-tuning | .071777 | .061000 | 1.130118 | .960933 |

The candidate is about 2.09 times CoDA's OOD error at both 16 and 64 observations,
and loses to the simpler functional comparator too. It misses the prospective
10% improvement threshold decisively. Correct support does help the candidate:
OODerror falls from 1.93118 at zero support to .640077 with 64 observations, whereas
shuffled-body support gives 3.489565. Thus it learns a useful experience-conditioned
correction, but the proposed extra coupling does not earn its complexity against
competent alternatives. Correct support benefits the stronger adapters as well.

This is one fitted seed on a small physical family, with finite optimization and
validation tuning. It rejects an early practical-advantage hypothesis at the
specified budget; it does not prove the joint architecture intrinsically worse,
that all functional learning fails, or that the baselines solve general robotics.
No statistical superiority or controller capability follows from these errors.

## Costs, fairness and bounded verification

Parameter counts: joint 14,169; separable 15,192; ALPaCA 7,904; CoDA 44,050;
pooled 4,866. CoDA is larger and uses 100 local context Adamsteps; ALPaCA adapts
analytically and the functional methods take 3 evaluations/updates. Ordinary
fine-tuning uses 100 Adamsteps. Global and adaptation LR choice uses validation.
Summed two-fit invocation times (including validation) are approximately 23.53 seconds
joint, 17.50 separable, 15.52 ALPaCA, 16.74 CoDA, 36.70 pooled, with two learning-rate
processes concurrent. These process times are not sequential elapsed time,
independent experience or a precision real-time benchmark. Raw per-method test
adaptation/query times and every forecast are saved.

ALPaCA retains a learned full SPD prior and predictive Gaussian NLL but fixes output
noise; CoDA retains full-network low-rank adaptation/locality while replacing the
original integrated trajectory loss with observed-transition regression. These
are scoped method cores, not full published benchmark reproductions or claims
against every modern implementation. Scope notes and primary papers remain.

Before fitting, small independent fixtures checked posterior vs recursive solve,
complete CoDA parameter mapping/local adaptation, gradients, zero-evidence/residual
behavior, permutation/chunk consistency, support/query separation and train-only
scales. Physics checked the validated parent unit body and a few generalized
DOP853 comparisons; no exhaustive historical trajectory replay was performed.
This verification supports the screening decision, not a framework claim.

## Scientific decision and continuation

**REJECTED:** added joint query/support/residual output-matrix transport as a
worthwhile next mechanism under this test and budget. No follow-on control or
second-family test is warranted for this candidate, since it failed the declared
screen. No learning rate, support count, physical range or optimizer is changed
from held-out outcomes. No new seed is run merely to seek a win.

What changed scientifically: learned functional updates can condition on the
correct new body's experience here, but an initially plausible generalization of
that operation is unnecessary in this screen. Conventional learned-prior and
contextual adaptation are strong; a future candidate must address a limitation
they actually exhibit rather than compare only to pooled fitting or v17 lookup.

Do not infer a successor mechanism from these aggregate errors alone. In
particular, successful baseline adaptation does not establish a premise for
nonstationary-task detection or update rejection; changing the task solely to
make that idea necessary would repeat the old drift. Close this bounded attempt,
preserve code/actual results and reassess a consequential capability gap before
starting another. The ultimate Goal stays active; no invented success, new
ledger, pause or external blocker is declared.

Reproduction uses ownerPC/project.venv and fresh output names/checkouts because
completed outputs refuse overwrite. Main executables:
error_transport_data.py, error_transport_screen.py fit --method METHOD --lr LR
--seed 0 --steps 2000, then error_transport_screen.py score --seed 0. Use --help for arguments; do not run on the published artifact paths
unless deliberately reproducing in a preserved separate checkout. All source,
10 fit reports/weights, raw forecasts and ordinary/evaluator data are retained.
