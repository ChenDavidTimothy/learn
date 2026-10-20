# Independent moving-acquisition artifact audit

2026-10-20. **PASS** for the saved exploratory corpus, with the scope below. No
canonical acquisition file was edited and no new acquisition, learner, fork
forecast or controller was run. This is artifact verification, not a v17 milestone.

Executable source: `audit_cartesian_stream_v17_acquisition.py`. Actual results:
`audit_cartesian_stream_v17_acquisition_results.json`. They bind the ordinary/
evaluator NPZs, acquisition JSON, corrected source54f5b710..., frozen design,
plant source and independent physics helper by SHA256.

## Verified information and randomness

All40 intended initial q/v states reproduce exactly from the three independent
per-trajectory RNG roles; intended Cartesian velocity matches to2.78e-17. All120
role keys are unique and include split, trajectory and role. All commands, initial
zero prefixes, held-two commitments, sensor draws, split/trajectory IDs and noisy
camera positions reconstruct exactly. The audit checks moving initial preparation;
this corpus does not contain matched A/B twin trials or supplied rest.

The ordinary file has EXACTLY these keys: observed, commands, split,
trajectory_index, dt, sensor_sd. The evaluator file holds states, sensor_noise,
plant_nfev and seed_paths. This schema does not by itself certify later learner
code: its actual dataflow, features/labels and heldout selection still need review.
No true velocity, noiseless p0, physical branch or response matrix is ordinary data.

## Independent physics rather than repeated source execution

No canonical acquisition/mechanics module is imported. The previously independent
Hamiltonian helper is reused, explicitly not claimed as another new derivation.
It advances q and conjugate momentum using

    qdot=M(q)^-1 P,
    Pdot1=u1,
    Pdot2=u2-sin(q2)*qdot1*(qdot1+qdot2).

This differs from the canonical velocity/Coriolis integration. RK4 uses128
substeps per.1sec interval. ALL10,240 actual intervals are checked from their stored
starts, with maximum state discrepancy1.3451e-11. In addition ALL40 complete
25.6sec trajectories are advanced from initial conditions WITHOUT resetting to
intermediate stored states. Their maximum state discrepancy is2.4654e-9; maximum
Cartesian position discrepancy2.7224e-9, far below camera SD.0002m.

A separate adaptive Radau integration of the Hamiltonian equations checks54
intervals: every trajectory's initial interval, ten spread mid-trajectory spots,
and extrema chosen by speed, phase increments and small Jacobian determinant.
Maximum state discrepancy is9.7345e-13. No broader/repeated integration was needed
after these appropriate checks passed.

## Coverage and counters

All saved energy/work and support quantiles reconstruct exactly. The corpus has
10,240 intervals/1,024simsec;320passive-prefix intervals/32simsec;9,920actuated
intervals/992simsec;10,280camera reads/20,560independent scalar noise draws. Stored
RHS counters sum to285,728. This reconstructs the stored historical counter; another
solver cannot independently observe the original wall clock or DOP853 call count.

Maximum energy/work residual is7.0517e-12. Maximum sampled Cartesian speed is
1.9059m/s and joint speed norm4.6812rad/s. Maximum joint coordinate increment is
.4244rad and distal phase increment.2391rad. These are descriptive sampled
properties, not a uniform continuous-time or temporal-aliasing certificate.

There are67sampled states with |detJ|<.01. The independent additional diagnostic
finds182consecutive-sample sign changes of sin(q2), across39/40trajectories. These
are bracketed chirality changes, not182exactly localized singular events or a
hidden-state filter. All trajectories/outliers remain. Initial slow/interior
preparation is plainly not a later trajectory envelope. Physical inertia remains
positive definite; minimum sampled eigenvalue.1715729137.

## Consequence and limitations

A persistent episode-level elbow or constant response matrix would be a false
physical assumption for this corpus. This is an opportunity to learn/revise useful
history-dependent relations through actual motion, not evidence that a particular
latent-mode mechanism is required. A serious conventional history predictor or
retained estimator may suffice.

The next separately frozen learner/fork protocol should test action-dependent
consequence contrasts before interpreting absolute next-position accuracy as
learned control knowledge. Renewals remain.2sec commitments. Full-history causal
randomization does not make arbitrary finite windows sufficient under policy
shift, and40streams do not give10,240independent contexts or general-body evidence.
No future fork outcomes should become hidden hyperparameter selection. The broader
goal still requires useful acquired relations and honest charged decisions.
