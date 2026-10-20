# Independent audit of the preliminary continuous-history pilot

**TESTED / independently verified, 2026-10-20. Not a completed learning comparison or a release milestone.** Reproduce from the project root:

```sh
.venv/bin/python research/audit_history_context_v15_pilot.py > research/audit_history_context_v15_pilot_results.json
```

The audit imports no canonical research modules. Its JSON records the source, question, result and v14 parent hashes. The v14 parent hash is checked against the previously completed independent v14 audit. All eight ordinary controller relations are exactly the information-corrected broad-data calibrated inverse fits in their original replicate order. There is no new model training or learned history-selection mechanism.

## Physics, histories and feasibility

I reconstructed the two-link mechanics in general initial momentum coordinates, using `p=M(q)v`, `qdot=M^-1 p`, `pdot1=u1` and `pdot2=u2-sin(q2)(v1^2+v1*v2)`. Separate vectorized Hamiltonian RK4 integrations at 32 and 64 substeps checked all 48 backward history states, forward continuation from each earliest past state to all later history samples/current state, all 12 bounded oracle witnesses, and all 4,608 saved future pulses. Maximum 32/64-step discrepancy was `3.655e-14`; maximum discrepancy from saved future states was `1.777e-15`. Past-state discrepancy was `1.111e-15` and forward reconstruction error `2.221e-15`. These are legitimate continuous zero-torque prehistories, rather than disconnected fabricated past observations.

All 12 oracle commands independently return their respective moving initial state to the specified current position within `1e-8`, respecting torque bounds. An independent optimization replay with the separate physics implementation also converged in all 12 cases and reproduced the source's 132 total residual calls, including its two final residual evaluations per world. This is an individual feasibility check. It does not make a global claim about the best action when motion is hidden, or about stationkeeping.

The ballistic/no-Coriolis surrogate identity in the question is algebraically valid for that explicitly simplified model. With fixed nonzero velocity, the actual nonlinear acceleration is `M^-1(u-h(q,v))`; omitting `h` is not the exact quadratic Taylor expansion in time. The current canonical result correctly names its scalar reference `ballistic_surrogate_risk_reference_not_nonlinear_certificate`. Zero torque is a descriptive baseline, not a finite-time global optimum or error lower bound for the nonlinear instrument. For the true paired endpoints, the midpoint/half-separation squared-error identity is exact; turning a small-time separation expansion into a numerical uniform bound requires an explicit remainder bound. This pilot does not provide one.

## Information and ordinary action reconstruction

The ordinary controller sees four noisy past positions, exact current preparation position and the goal equal to that same position. The times and anchored slope arithmetic are supplied:

\[
\widehat v={\sum_i t_i(o_i-q_0)\over\sum_i t_i^2},\qquad
a_*=-2\widehat v/T.
\]

Every random reading and slope was independently reconstructed. Measurement RNG matched exactly and maximum slope discrepancy was `3.47e-18`. With independent coordinate noise variance `1e-8`, the slope's sensor covariance is `1e-8/(sum t_i^2) I = 1.333333e-7 I`; finite-history curvature can additionally bias the slope as an estimate of instantaneous velocity. The pilot does not learn or validate the choice of this summary.

I independently evaluated the retained v14 RBF coefficients using the equivalent cosine chord-distance expression. The ordinary command minimizes `||pinv(K(q0))*u-a_*||^2` over `[-1,1]^2`. A separate two-dimensional box boundary/interior enumeration reproduces all 1,536 history commands to `4.00e-15`, with objective discrepancy below `3e-30`. All saved commands satisfy their bounds. True current velocity, inertia and future outcomes do not enter this ordinary computation. Those quantities construct the simulated worlds and the explicitly privileged oracle only.

The source does not retain per-oracle `nfev` or ordinary solver status records. Independent endpoint replay and exhaustive bounded quadratic certificates establish the saved command properties; they do not reproduce every internal numerical iteration or establish global nonlinear optimality. There is no future measurement noise or feedback during the one future pulse.

## Freshness and the excluded seed collision

All 1,536 canonical sensor streams are unique, spanning `50020261020` through `50027371035`. An explicit enumeration of 24,881 v12/v13/v14 RNG roles, including physical/virtual CEM, validation banks, checks and heldout evaluation, has maximum seed `43029161021`. The new range is strictly above that maximum and has no intersection.

The preserved excluded seed-collision pilot is not fresh independent evidence. The audit finds **24** exact previous-role matches: eight broad-v14 acquisition seeds and sixteen v12 heldout-test seeds. The latter occur for world index 10, measurement repeats 0/1, across the eight model indices. All seed values are retained in the audit JSON. This does not affect the corrected 50-billion namespace.

## Scores, uncertainty and limits

All 1,536 rows and three future outcomes per row were retained. Endpoint position error and endpoint velocity norms were independently recomputed, as were all means/quantiles/MCSE. The reported position MSEs are:

| Method | Mean endpoint squared position error | MCSE across eight parent-model clusters |
|---|---:|---:|
| Pose-only zero torque diagnostic | `1.66667096217e-5` | `0` |
| Learned relation plus supplied history slope | `2.20125513511e-8` | `3.71420176138e-9` |
| Privileged current-state feasibility oracle | `1.47272254051e-30` | `0` |

Each cluster averages twelve fixed worlds and sixteen independently drawn measurement records. The eight clusters correspond to independently acquired v14 relations, not 1,536 independently learned models. Zero and oracle outputs are deterministic across these model/measurement repetitions, explaining their zero MCSE. This is not zero uncertainty about arbitrary worlds or a population performance guarantee.

The pulse returns near a position while retaining motion: mean final velocity norms are `0.0201776`, `0.0203072` and `0.0201776` for zero/history/oracle. Maximum history endpoint speed is `0.0253374`. Consequently this is not a stopping result. It contains no process noise, continuous goal sequence, online feedback or learned revision of an insufficient summary. It shows that a supplied causal history statistic can use a previously learned rest relation for this small return-position instrument. It does not demonstrate autonomous discovery of velocity, a general state representation or a general learning architecture.

## Actual computation and data costs

The canonical source performs 48 backward integrations, 12 forward-return checks, 12 zero-reference integrations, 132 privileged oracle residual calls and 4,608 future evaluation integrations: **4,812 ODE calls**, with **958.8 total absolute simulated time units** including backward durations. Its 1,536 history records consume **12,288 scalar past-coordinate sensor readings**. Eight v14 learned relations are reused; new learned-model training samples are zero. Many zero/oracle trajectories are repeated deterministic evaluations, not additional distinct physical worlds.

The excluded collision run performed the same 4,812-call protocol and 12,288 sensor readings. Counting both development runs gives 9,624 ODE calls, 1,917.6 absolute simulated time units and 24,576 sensor readings, before independent numerical audit compute or any repeated reproducibility execution. These are owner-PC simulations, not hardware experiments. Their cost is not hidden learner training or a runtime-efficiency claim.

The meaningful next experiment must ask whether continuous experience can select or revise useful history, compare serious conventional history-based alternatives, and test consequences under unsupported motion without giving only one ordinary method privileged velocities. The present pilot alone should not produce a v15 milestone.
