# Independent frozen Cartesian reuse pilot audit

2026-10-20. **TESTED:** saved physical trajectories, fit records, posterior updates, bounded continuations and scores independently reconstruct. **NUMERICAL LIMITATION FOUND:** the order-nine first-action quadrature is not adequately converged for a competitive comparison. This is a construction pilot audit, not a release recommendation or evidence of general Cartesian control. No canonical source/results were edited.

Reproduce from the repository root:

```sh
.venv/bin/python research/audit_cartesian_reuse_v16_pilot.py
```

The checker imports no canonical project implementation. Exact source/design/data/results hashes are in `audit_cartesian_reuse_v16_pilot_results.json`, together with its executable source hash. It resolves the frozen pilot design explicitly because generated metadata names the mutable canonical design basename. Audited frozen source SHA256: `0c3ea4a359ee83cfe6c761a00fdcfbe82f16fa763bf8488a223e430d737a6426`. Audited raw NPZ SHA256: `014c0350acbb828e1300afb709773a81ee6416f6a8bf3377b748d1087624c6b4`.

## Independent reconstruction and information access

The source-level dataflow separates mechanics/preparation from learning. Fits receive only issued commands and noisy Cartesian readings with supplied exact initial Cartesian position/rest. They do not receive joint coordinates, true B, branch labels, or evaluator velocities. Whole-group validation selects mixture and covariance-floor candidates. Test groups do not select fits. Privileged nonlinear witnesses are evaluator diagnostics. First-action design integrates the learned prior, not the actual hidden branch. No hidden reset occurs after probing; real motion is carried through all three pulses. After the first observation, the model conditions B once and computes position/velocity consequences from that same uncertain B rather than initializing a separate exact state from the noisy reading.

All 384 acquisition RNG paths, preparations, commands, observations and group exposures reproduce. Training/validation/test paths are disjoint, and disjoint from the 96 shared control-noise paths. Pairing across methods and branches deliberately reuses each control noise path. This is one acquired fit with 16 sensor repeats per branch/goal, not 192 independently trained systems or evidence across unrelated environments.

Physics is reconstructed from Hamiltonian equations: qdot=M^-1 p, pdot_1=u_1, pdot_2=u_2-sin(q_2)*v_1*(v_1+v_2). Vectorized RK4 with 128 steps per interval reproduces all acquisition states within 2.9e-15 and all 1,728 control trajectories within 4.0e-15 in their saved state coordinates. Independent Radau checks of six acquisition intervals and ten control intervals agree within 4.5e-16. Acquisition energy minus applied-torque work is below 1.1e-15. All twelve saved nonlinear endpoint/stop witnesses reproduce using Radau, with normalized residual differences below 1.1e-12. These checks verify the generated mechanics in this regime; they do not make the constant-B learner exact.

All 17 EM candidate fits, histories, initializations, validation likelihoods and the selected record reconstruct; maximum matrix difference is 2.7e-13. Sixteen candidates report convergence; the unconverged candidate remains retained. The selected K=3 fit has weights 0.0057983, 0.4825777 and 0.5116240. Component count is selected from a supplied menu; the physical branch count is not a supervised target.

All three Gaussian moment candidates and floor selection reconstruct. The moment operator has rank ten and condition 12.68494. Raw covariance eigenvalues are approximately (-0.0030071,-0.0002095,0.0565288,1.1677090); PSD projection is a consequential estimation operation, not evidence that raw moments were already a valid covariance. The Gaussian baseline is moment fitting, not optimized Gaussian marginal likelihood; any later refinement is a separate preserved artifact.

Every saved control posterior, E[B'B], continuation command, projected-gradient residual and expected cost reconstruct. Maximum continuation action difference is 6.4e-15. Independent L-BFGS-B checks of 54 representative strictly convex QPs reach objective differences below 1e-8; maximum action difference from face enumeration is 6.9e-8. All record costs, component scores, success flags and endpoint velocities recompute. No dropped trajectories or clipping were found.

## Actual pilot outcomes

Every row below has 192 episodes. These are descriptive results from one learned prior, six small goals and paired sensor repeats.

| Prior / first action | Mean actual cost | Precision successes |
|---|---:|---:|
| Mixture / directed grid | 0.000947949 | 192 |
| Mixture / ordinary task | 0.666607659 | 64 |
| Mixture / random direction | 0.178850321 | 148 |
| Gaussian moments / directed grid | 0.074139521 | 160 |
| Gaussian moments / ordinary task | 0.702017014 | 62 |
| Gaussian moments / random direction | 0.272991842 | 112 |
| Mixture-matched Gaussian / directed grid | 0.036181712 | 165 |
| Mixture-matched Gaussian / ordinary task | 0.666607155 | 64 |
| Point mean / ordinary task | 591.414627911 | 64 |

The point-mean failure is algorithm/family-specific. Its first command consumes leading stopping authority in 128 episodes. It is not a theorem that all conditional-mean learners fail, nor a lower bound for the exact nonlinear body. All executed probes from the other priors satisfy the leading stopping-feasibility check.

## Numerical failure: first-action integration

All 900 saved candidate scores reproduce at order nine, establishing implementation agreement but not integral accuracy. Recomputing complete menus at orders 17 and 33 changes the general-Gaussian winner for goals 2, 3 and 5 (zero-based indices): saved candidates 36,44,47 become 35,43,46. At goal 5 the maximum whole-menu score change is 0.047388 from 9 to 17 and another 0.017760 from 17 to 33. At order 33, candidate 46 costs 0.536985 versus 0.549748 for saved candidate 47.

Higher-order checks of those selected/competing candidates confirm the direction but do not certify arbitrary precision. At goal 5 their scores at order 65 are 0.535429/0.550581 and at order 129 are 0.536377/0.551519. At goals 2/3 the alternative converges near 0.195110 while the saved choice is near 0.196221. Thus selection discrepancies are not merely roundoff ties. Other Gaussian selected values still shift by about 0.0014–0.0018 between orders 65 and 129.

Mixture and matched-Gaussian winners remain unchanged at orders 9/17/33 for all six goals. Nevertheless, some whole-menu values change materially: maximum 9-to-17 differences are about 0.01251 and 0.06682, respectively; maximum 17-to-33 differences are about 0.00233 and 0.03405. Stable winners do not establish converged candidate values. The pilot must retain its original actions/results; a fresh competitive design needs a disclosed accuracy criterion and stronger integration/ranking checks, including its refined Gaussian baseline. Directed selection is a finite-grid numerical approximation, not global dual optimality.

## Prescribed diagnostics and deviations

The design prescribed heldout conditional prediction but source saved only joint likelihood and mean MSE. This audit reconstructs normalized response NLL for the second observation conditional on the first:

| Prior | Joint group NLL | Second given first NLL |
|---|---:|---:|
| Mixture | -9.399380 | -4.908059 |
| Gaussian moments | -6.754629 | -3.525720 |
| Matched Gaussian | -7.111606 | -3.837180 |
| Point | 402.234951 | 370.519134 |

Negative continuous-density NLL is possible and these values use normalized displacement coordinates. This is predictive evidence within the declared supplied local family, not a universal representation ranking.

The design also prescribed checking every candidate's leading stopping feasibility before controls. Source instead checked only executed probes afterward. This is a protocol deviation even though the audit now finds all 50 candidates feasible for both branches in each of the 18 non-point prior/goal menus. The point prior has no saved design menu and its 128 infeasible executed probes remain explicitly retained. Posthoc verification does not retroactively satisfy the precomputation requirement, and leading feasibility remains distinct from a finite-time nonlinear witness.

Physical generation counts reconstruct: 384 acquisition episodes / 768 intervals / 76.8 simulated seconds, 2,304 scalar acquired sensor readings (including the unused noisy initial read); 1,728 control episodes / 5,184 intervals / 518.4 seconds, 3,456 scalar intermediate readings. The twelve privileged witnesses report 227 actual residual calls, hence 454 additional flow calls; their optimizer-reported 47 nfev is a different counter. Reintegrations performed by this audit are validation computation, not extra learner experience.

This supports a useful acquired-prior engineering construction under supplied rest/preparation and a local response approximation. It does not yet demonstrate sustained feedback, independent fit-replicate reliability, cross-pose generality, superiority to strong modern robot learners, or a general mechanism of intelligence. Preserve the numerical failure and deviations before any stronger fresh study.
