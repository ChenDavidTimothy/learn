# Independent v11 deadline-support audit

Auditor: `v9_artifact_audit`. Execute with the existing project environment:

```bash
.venv/bin/python research/audit_deadline_control_v11.py
```

Actual evidence is in `audit_deadline_control_v11_results.json`. No canonical
controller/source/result was edited by this auditor.

## Consequential protocol correction

The original executor forced commands to zero at time70, while its planner still
optimized eight free commands even immediately before shutdown. Thus it could
rely on future actions that would never execute. The new aware planner uses
N=min(8,70-t), consistently with the executor. Its argument is explicitly the
announced remaining action opportunities; future goal values and physical states
remain hidden. ACTIVE=70 was already present in the execution wrapper, so this
is most precisely a correction to planner/executor task-constraint consistency,
not acquisition of new physical information.

Both added variants have sixteen predicted zero-command coast steps. Comparing
the old twelve-step prediction against unaware16 isolates tail length; comparing
unaware16 against aware16 isolates use of the cutoff. The source's original
optimizer and sensitivities use len(actions), so shrinking/truncating the warm
vector correctly changes the free-action support. Shifted warm starts shrink
without reset. All ninety-six paired aware/unaware episodes have exactly equal
commands through t62 and equal physical/filter histories through the first
shrinking decision at t63. This is a direct check that only the intended planner
constraint changes behavior.

## Independent full-artifact reconstruction

The audit independently reconstructs all192 added episodes through an eliminated
motor recurrence, including sensor/process streams, current-position Kalman
estimates, metrics and success criteria. It independently differentiates all
13,440 saved variable-length plans and verifies selected costs, shifted/truncated
warm costs, projected gradients, first commands, announced horizons and bounds.
Twelve independent seed clusters, each averaging the two tasks, reproduce all
summary means/SEs/quantiles. Stop counts remain24 trials per cell.

Maximum discrepancies: physical states/observations 1.60e-14, filter estimates
8.96e-16, task metrics 1.51e-14, endpoint diagnostics 7.11e-15, warm costs
4.55e-13, selected costs 5.69e-14, projected gradients 2.95e-12. All first
actions agree exactly. No unresolved implementation discrepancy was found.

## Conditional endpoint mechanism

For the **known-parameter** Kalman filter only, independent state rollout gives

```text
b69 = [A17 * posterior_mean69]_x + beta*[A16]_(x,m)*u69 - final_goal.
E[(x86-final_goal)^2 | usable_history69] = b69^2 + V.
```

The information variance V is unchanged across deadline policies. The exact
paired-noise check is stronger than matching averages: realized endpoint residual
`x86-final_goal-b69` is identical across old12, unaware16 and aware16 within
3.11e-15. The known-parameter filter error evolves independently of issued
observation-adaptive controls, so this is the expected identity. Differences in
the paired endpoint therefore arise from conditional means, not improved sensor
information or new noise realizations.

Fixed16 alone changes the original result negligibly. Deadline awareness reduces
the empirical mean of squared conditional endpoint bias from1.676571 to.001556
at sensor SD .2, and from.603351 to.000664 at SD .02. These are averages over24
observed histories, not exact population risks. The corresponding variance V
remains about.343422 and.193027. It is therefore incorrect to attribute most
original endpoint error to irreducible uncertainty merely because a nonzero
information lower bound exists.

The diagnostic last-action adjustment `u69-b69/(beta*h16)` exactly centers the
conditional endpoint mean with earlier controls/history fixed. Its feasibility
does not certify whole-stop success, low intermediate error, optimal active
tracking or global policy optimality. The whole event constrains many positions
and velocities, while the endpoint diagnosis concerns only the last position.
The source correctly applies this exact conditional decomposition only to the
known-parameter filter; fitted-model beliefs do not automatically have that law.

The original covariance and Gaussian probability bounds remain valid. Their
existence does not explain every failure, and the repaired stopping rates still
do not attain their terminal-marginal probability ceilings. The scientific
correction is to separate avoidable task/planning bias from conditional
uncertainty before concluding that additional sensing or learning is required.
