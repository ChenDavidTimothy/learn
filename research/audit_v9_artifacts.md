# Independent v9 artifact audit

Auditor: `v9_artifact_audit`. No canonical simulator, estimator or result was edited.
The independent code is executable with the existing `.venv/bin/python`:

```bash
.venv/bin/python research/audit_force_gauge_v9.py
.venv/bin/python research/audit_nonlinear_identification_v9.py
```

## Equivalence, timing and interventions

The pilot and expanded final gauge artifact use **lagged** actuator timing, unlike
v8's motor-first timing. Independent reconstruction eliminates motor state:

```text
v[t+2] = f(v[t+1]) + lambda*(v[t+1]-f(v[t])) + beta*u[t] + xi[t].
```

Initial v[0]=v[1]=0 follows from initial v=m=0. The pilot's first measured
displacement uses an artificial previous observation zero, equivalently e[-1]=0;
it is not a stationary initial sensor difference. The independent audit integrates
positions, adds identical sensor errors, and reconstructs the feedback commands
from those observations. Maximum position/observation discrepancy is 7.82e-14,
velocity 9.55e-15, command 7.11e-15, and reconstructed force gauge 8.44e-15.
Both canonical gauge programs independently rerun to the saved JSON exactly.
Those reruns establish reproducibility; the eliminated-recurrence implementation
provides the distinct computational verification.

For the expanded 48-seed/policy artifact, independently reconstructed force RMS
and maxima agree within 6.39e-16. Its largest saved connected observational
equivalence error is 3.98e-13. Independent motor-removal preparation, replay,
response distance and lower bounds agree within 2.22e-16. Response distance is
0.2733856697523759, giving expected-RMSE bound 0.13669283487618795.

The transformed actuator is explicitly causal: its tanh argument rho*v+q is
already determined from present state before xi. It is two-state, input-affine,
and retains the same additive actuator innovation. No extra velocity process
noise is present. Adding independent velocity noise would generally introduce
an additional transformed actuator innovation; the convenient same-noise claim
must not be generalized to that different model.

The audit actively checked initialization and intervention counterexamples.
At v=1, forcing both m=0 and q=0 yields a one-step gap 0.15231883119115297;
compatible initialization instead maps q=m-c*tanh(v). A coordinate change alone
does not produce a distinct physical intervention: original physical m=0 maps
to q=-c*tanh(v), giving exactly equal outputs. The final artifact correctly adds
**different physical drag locations/component boundaries** before defining
removal: body-mounted drag remains in A; motor-mounted drag is removed in B.
Its component-transfer bound is conditional on those explicit physical worlds,
not a consequence of renaming a hidden state.

## Restricted identification and observation error

All 108 clean regressions were independently rebuilt from physical indices
t=1,...,n-2. Their coefficient discrepancies are <=1.91e-13 and singular-value
discrepancies <=5.69e-14. Full regressor rank at c=0 does not identify physical
pole allocation: the coefficient-to-parameter inverse still has two branches.
For c!=0 the ratio labels lambda only within the supplied passive-tanh plus
velocity-independent linear-actuator family. Small c and narrow velocity support
make that inference fragile. Nonlinear feature errors do not satisfy the linear
errors-in-variables proof merely because additive sensor errors are independent.

All 72 noisy runs, all **432** recorded output-error/structured-regression
optimizer objectives, all selected-start choices, heldout replay metrics and
summary statistics were independently checked. Recorded objectives agree within
9.95e-13. Parameter-derived error/ratio values agree within 5.97e-13 after scaling
by max(1,magnitude). Heldout metrics and summaries agree within 4.07e-13 on that
scale. Raw unconstrained OLS has enormous unstable extrapolations, so absolute
floating-point discrepancies can exceed 1e61 despite this relative agreement;
they are preserved in audit JSON, not hidden or treated as additional failures.
An initial audit absolute-error assertion was corrected to use this appropriate
scale; no canonical experiment was altered.

Trajectory derivatives were checked against finite differences of the independent
eliminated recurrence, including a high-persistence parameter case. Maximum scaled
error is 4.13e-9 (absolute 9.47e-5 for large accumulated position sensitivities).
The structured regression Jacobian independently agrees within 4.51e-10.
Fresh numerical-Jacobian fits from an additional start reproduce a strong-signal
seed's output-error SSE 20.8403405031211 versus saved 20.8403405031210, and
structured-regression SSE 112.4063060060772 versus saved 112.4063060060772.
This supports the implementation; neither multistarts nor convergence certify
a global optimum for all seeds.

Output-error least squares is the appropriate Gaussian output-error likelihood
under this experiment's process-noise-free deterministic dynamics, exogenous
open-loop commands, iid position errors and exact known zero initial state.
Those are substantial supplied assumptions. OE and constrained one-step
regression get identical commands/observations and the same four-parameter family,
bounds, starts and evaluation budget; selection uses training data only. Their
different objectives test observation-error treatment while controlling the
family/stability regularization. The raw five-coefficient OLS also lacks those
ties/bounds, so its extreme failures cannot alone establish the observation
model's contribution. Heldout replays are clean and start at known zero state;
they do not evaluate unknown-state filtering, closed-loop learning or goal control.

No novelty, broad robot-learning generalization, superiority to optimized modern
methods, or completion of the persistent scientific objective follows. No
unresolved implementation discrepancy was found in the final audited artifacts.

## Final ledger wording audit

Both independent auditors checked the new scientific section before publication.
Corrections specify clean known-zero-state preparation/removal evaluation, measurable
phi for trajectory laws, stable actuator/contractive passive velocity rather than
contractive integrated position, and distinguish guaranteed label identification
from accidentally choosing the correct c=0 optimizer branch. No remaining material
proof, metric or reproducibility discrepancy was identified.
