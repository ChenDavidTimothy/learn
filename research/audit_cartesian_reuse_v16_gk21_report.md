# Independent Cartesian batch numerical repair check

2026-10-20. **TESTED:** the immutable `cartesian_reuse_v16_gk21.py` batch agrees with the independently reconstructed complete menu. This is an engineering-pilot numerical validation, not a release recommendation or globally optimal control claim.

```sh
.venv/bin/python research/audit_cartesian_reuse_v16_gk21.py
```

The guarded checker/results verify exact source, design, raw-data, parent and independent-audit hashes. All 900 candidate values agree with the independently implemented likelihood/posterior/response/box-QP integrator within 9.97e-12. The independent integrator uses `gk21` with tighter tolerances; both use SciPy's adaptive engine. Earlier Genz-Malik discrepancies were also independently corroborated with scrambled Sobol integration, as preserved in `audit_cartesian_reuse_v16_adaptive_report.md`.

All 18 directed winners and direction-averaged random radii agree. The new batch's complete controls, priors, fit records, heldout predictions, witness records and resources are exactly identical to the independently audited adaptive parent. Its raw NPZ is also identical. Consequently the prior physical/posterior/score verification carries through without repeating those trajectories. No new calibration was acquired; this reuses the exploratory evaluation cohort.

Every new integral records the intended `gk21`, atol1e-8/rtol1e-7 and reported convergence. Those estimated errors remain numerical estimates. The analytically checked tail envelope applies under the declared local probabilistic model; it does not certify cubature error. This validates the observed finite-menu repair and preserves the failed methods. One fit, tiny Cartesian goals, supplied local response structure and three pulses with one intermediate adaptation remain the scope. Sustained feedback requires its own audit.
