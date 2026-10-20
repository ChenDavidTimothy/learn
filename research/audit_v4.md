# Independent v4 audit, 2026-10-20

Auditor: independent_theory subagent, independently derived mathematics and reran artifacts. No canonical edits by auditor.

- The no-invariant-law proof passes: positive-probability quadratic escape contradicts recurrence under any hypothetical invariant probability.
- Exact ODE solution independently checked against 20,000-step RK4 for initial velocities -20,-4,-.3,0,.3,4,20: maximum discrepancy 2.49e-14.
- Complete `dissipative_flow_v4.py` rerun parsed exactly equal to saved JSON, all 128 seeds.
- Independently reconstructed position by cumulative sum, added sensor errors, then differenced and refit seed20661020. H0=1.1608909543532206,H1=.8485344298978034 agree with saved values within 1.3e-15.
- Indexing correct: y[t]=v[t+1]+e[t+1]-e[t]; tag a[t-1] acts through f(v[t]), so derivative(v[t]) is the correct finite-time index.
- 6000 retained transitions yield 5997 regression rows after excluding the first three lags.
- Flow semigroup maximum error 4.44e-15; finite-difference derivative error 8.27e-7. The latter is largest near zero, where the second derivative changes side.
- Exact flow changes Euler's numerical coefficients. It describes unforced damping plus subsequent impulses, not exact continuously forced dynamics.
- Four-MC-SE assertion is only a smoke test, not equivalence proof or calibrated error control.
- For stationary derivative g=f', synchronous coupling gives initialization error <=2*r*alpha*r^t*E|xi|/(1-r), below 6.5e-66 at burn600 in both conditions. This supports burn-in under the replacement assumptions.

No unresolved mathematical or numerical disagreement. This correction is established numerical-analysis territory and does not meet the overall foundational-advance completion criterion.
