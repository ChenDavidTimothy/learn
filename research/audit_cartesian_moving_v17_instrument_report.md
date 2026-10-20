# Independent moving-Cartesian framing instrument audit

2026-10-20. **TESTED:** physical matched twins, acceleration separation, finite passive trajectories, specified Gaussian information curves and resource counts independently reconstruct. **SCOPE:** evaluator support/framing instrument only. No learner, actual noisy measurement set, classification result, control task or v17 learning milestone is established. No canonical source/results were edited.

```sh
.venv/bin/python research/audit_cartesian_moving_v17_instrument.py
```

The executable/results share this report's prefix and preserve exact source, result, question and independent-helper hashes. No canonical project implementation is imported. Hamiltonian RK4 and Radau check physics; a separately reconstructed q/v right-hand side with DOP853 checks the reported adaptive-solver evaluation counter.

## Matched preparations and physical distinction

For radial/tangential Cartesian velocity (vr,vt), c=cos(phi/2), s=sin(phi/2), set a=vt/(2c), b=vr/(2s). Independent polar construction gives joint velocities (a+b,-2b) and (a-b,2b) at the two opposite elbow poses. All 54 stored preparations match this construction within 2.8e-17; both present Cartesian positions match and both velocities equal the declared V. They are evaluator-prepared twins. Exact joint/Cartesian velocities and physical derivatives are not inputs to an ordinary learner.

Independent Hamiltonian differentiation verifies the supplied Cartesian acceleration formula to 5.3e-18. For branch B minus A, at the local radial/tangential frame,

    delta_radial = -2*s*vr*vt / D,
    delta_tangent = (s*vt^2 + c^2*vr^2/s) / D,
    D = 2-cos(phi)^2.

For 0<phi<pi, c,s,D are positive, so any nonzero matched Cartesian V gives strictly positive tangential difference. At exact rest it is zero. The common-input response difference is radial; several additional common torques independently preserve the tangential discrepancy within 8.7e-16. Thus a common torque cannot cancel the entire initial acceleration difference of these moving twins. This is a structural statement about this fixed body and these matched alternatives, not a general theorem that all hidden physical circumstances become observable passively.

The result correctly challenges extrapolating v16's stationary aliasing argument to motion. Under smooth dynamics, matching p and V removes constant/linear separation, leaving .5*delta_acceleration*t^2+O(t^3). Special configurations/velocities can cancel higher terms; moving remainders must not be assigned the resting quartic expansion automatically.

## Independent finite trajectories and numerical limits

All 1,080 passive intervals replay with vectorized Hamiltonian RK4; maximum state difference is 9.4e-15. All 324 short Taylor flows also replay. Radau independently checks 66 passive intervals and every one of the 324 Taylor flows; maximum passive state difference is 9.6e-15 and maximum Taylor Cartesian-separation difference 8.9e-16. Energy drift is below 4.9e-17 and cyclic angular-momentum drift below 1.2e-16.

Whole trajectories rotate correctly under the .7-radian global rotation: maximum Cartesian mismatch 1.8e-15. Isotropic-noise information curves are rotation invariant to accumulated numerical error (maximum squared-Mahalanobis difference 4.5e-10). These are not independent samples of a population merely because both orientations are tabulated.

Some short Taylor remainders are at machine-precision scale. Thirty-eight of 162 paired Taylor checks—including rest—have saved remainder below 2e-14 m. Cubic/higher-order empirical ratios cannot be resolved reliably for those entries; their tiny values are not high-relative-accuracy certificates. The exact acceleration calculation and robust finite .1–1 second trajectories establish the relevant distinction without that extrapolation.

## Gaussian calculation: what it actually measures

Let each specified world's deterministic sampled positions form a known mean vector mu_A or mu_B. Assume independent coordinate/time noise N(0,SD^2 I), identical across the two hypothesis laws, and equal prior probabilities. The log likelihood ratio is Gaussian with variance d^2 and means plus/minus d^2/2, where

    d^2 = ||mu_B-mu_A||^2 / SD^2.

Consequently KL(B||A)=d^2/2 and equal-prior optimal binary error is Phi(-d/2). These are the implemented formulas. Independent accumulated arithmetic and an erfc-based normal tail agree with the saved squared distances, KL and errors to 4.6e-13, 2.3e-13 and 1.2e-16 respectively. Every error/summary curve recomputes, including monotonic cumulative information and nonincreasing known-pair error.

| Cartesian speed | Grid pairs | Mean ideal error at .1 s | At .5 s | At 1 s |
|---|---:|---:|---:|---:|
| 0 | 6 | 0.5 | 0.5 | 0.5 |
| .03 m/s | 24 | .497763 | .430374 | .197066 |
| .1 m/s | 24 | .475137 | .054423 | .0000178844 |

At .03 m/s the one-second known-pair errors range from .058276 to .381381 across the declared grid. Structural difference therefore does not imply useful short-window resolution. The acceleration signal is quadratic in speed; its leading squared-distance contribution is fourth order in speed for a fixed short window. Arbitrarily slow nonzero motion can remain arbitrarily difficult with finite noise/time.

No noise is actually sampled, no classifier is trained, and no realized success fraction exists here. The table is a deterministic known-pair likelihood calculation with a hypothetical sensor law. Estimating initial motion, pose, body parameters, unknown timing or correlated sensing creates a different composite inference problem. The grid mean is a descriptive average, not a universal Bayes lower bound or population performance estimate.

## Active falsification of an overly broad passive-information interpretation

**Counterexample under an enlarged body family:** multiply both point masses by k>0 while keeping lengths, matched initial q/v and calibrated torque units fixed. Then M_k=k M and h_k=k h. With zero torque,

    qddot_k = (kM)^-1*(-k h) = -M^-1 h,

so every passive Cartesian history is exactly unchanged for all k. Under a known nonzero torque, however,

    qddot_k = -M^-1 h + (1/k)*M^-1 u.

Absolute torque-response gain differs. The independent executable checks the passive scaling identity numerically; its maximum RHS discrepancy is below 3.5e-18. This does not contradict the fixed-unit-body elbow result. It shows why passive identification of a context must not be conflated with acquisition of intervention consequences: those can still require action data or additional prior assumptions. Nor does it prove every task requires identifying mass; common adequate actions can exist.

These recorded sequences are forward passive observation windows after preparation, not pre-existing histories supplied at a present decision. Waiting for such evidence consumes time and changes the actual moving state. There is no charged goal/control task here with which to decide whether waiting, probing or ordinary task motion is preferable. A learner may eventually exploit an already available causal history, but this instrument does not construct or score it.

## Counts and conclusion

Counts independently match: 54 paired contexts, 108 passive episodes / 1,080 intervals / 108 seconds; 162 paired Taylor checks / 324 flows / 1.89 seconds; 1,404 total plant flows. The independently reconstructed q/v DOP853 pass reports the same 40,104 RHS evaluations. There are zero actual sensor draws and 2,160 hypothetical scalar sampled-position opportunities. Audit reintegrations are validation computation, not acquired learner data.

The framing correction is sound: stationary passive aliasing does not persist automatically when natural motion changes the output law. Finite-noise usefulness and acquisition of action relations remain unresolved. Use this as support for a less supplied learned history/experience construction and a fair conventional comparison, not another observability-only release or a reason to prescribe a favored architecture.
