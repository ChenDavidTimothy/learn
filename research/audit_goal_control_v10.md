# Independent v10 control-artifact audit

Auditor: `v9_artifact_audit`, continuing independently from v9. Canonical
simulator/controller sources and results were not edited by this auditor.

```bash
.venv/bin/python research/audit_goal_control_v10.py --pilot
.venv/bin/python research/audit_goal_control_v10.py
.venv/bin/python research/audit_goal_control_PD_v10.py
```

Corresponding audit result JSON files preserve the actual checks.

## Timing, gradients and information access

The controller and physical plant consistently use lagged motor timing. Current
u changes mnext; it has exactly zero direct effect on xnext and vnext. Independent
motor-eliminated simulation reproduces the trajectory objective, and finite
differences verify action gradients (maximum scaled error 7.44e-10, including a
high-persistence case whose gradients exceed 5e5). The planning continuation has
eight optimized commands followed by twelve zero commands. Only the current goal
is supplied to the planner. No physical state, filter, warm start or PD smoothing
state is reset at the goal switches.

EKF updates use the current noisy position and the preceding predicted belief,
then propagate through the chosen issued command. Process noise enters the motor
coordinate after the transition. Its covariance is included for the seventy
active steps and disabled for the sixteen-step coast, exactly matching the
simulator. Sensor noise continues throughout. Known-zero initial means/covariances
are an explicit supplied capability. Only `oracle_state_MPC` reads the true state;
`oracle_model_EKF` additionally knows true parameters but uses the same noisy
observation channel. Nonlinear EKF/finite-horizon certainty-equivalent planning
is not an optimal stochastic-control theorem.

Training is process-noise-free with noisy positions; testing additionally has
motor process noise during active control. All learned controllers use identical
512-command physical training data. PD tuning uses only synthetic experience on
the fitted nonlinear model and supplied noise scales. True-plant evaluation
occurs after gain choice. No heldout goals or true states are passed to the
ordinary controller's action selection. Supplied model families, bounds,
stopping tolerances, timing and noise scales remain substantial prior knowledge.

## Exact linear pole symmetry and the corrected optimizer

For c=0 define T(x,v,m)=(x,v,m+(rho-lambda)*v). The linear transition matrices
satisfy T*A(rho,lambda)=A(lambda,rho)*T. Input direction, position observation and
motor-process covariance are unchanged by this transform. Starting from mapped
beliefs, Kalman posterior means and covariances therefore remain mapped under
common measurements/actions. The independent filter check agrees within
3.56e-15. Every planned x/v/action-cost vector is identical, so the MPC objective,
gradient and exact optimum are identical. The .03 action penalty makes the
linear problem strictly convex. A separate bounded-linear-least-squares solver
recovers swapped optimal actions within 2.25e-14.

The original forty-iteration L-BFGS-B pilot failed an initial literal-action
invariance check: maximum closed-loop action difference .010663421670913442,
state difference .007806357460745816, primary-cost difference 1.64e-5. Its
recorded projected-gradient residuals reach 2.0 across the full experiment.
This was a numerical optimization limitation, not a failure of the symmetry
proof. The initial strict action-equality assertion was rejected and its
diagnostic preserved. The main researcher preserved the full original source
and 672-episode output as `goal_control_v10_lbfgs_pilot.py` and corresponding JSON.
The pilot did not establish publishable optimizer-quality evidence.

The final source uses bounded trajectory-residual least squares for **all** MPC
models. The independent paired canonical run now agrees within 1.64e-14 for
actions/states, with primary-cost difference 2.23e-16. All 33,600 final MPC solves
report convergence; maximum independently checked projected-gradient residual is
3.165123638160239e-5. Convergence does not prove global optimality for nonlinear
plans. The linear strictly convex case admits the stronger independent check
above. The action objective and experiment/scoring design were preserved while
fixing the solver.

## Full final-artifact reconstruction

Independent eliminated recurrence replays **all 672 final episodes**, not merely
a selected favorable episode. It reconstructs each exogenous sensor/motor stream,
physical state, observation, EKF estimate, PD smoother, every task metric and all
summary statistics. It checks the twelve fitted-model seed clusters explicitly:
the two task outcomes are averaged within seed before Monte Carlo SE, while
success totals correctly retain 24 task trials per plant/controller.

All saved training sequences, fit objectives and selected fits, clean-prediction
scores, and coarse offline PD objectives are independently reconstructed. Every
saved final eight-command plan is audited through a distinct eliminated-state
sensitivity recursion: selected cost, projected-gradient residual, first issued
action and shifted warm-start cost. No warm-start cost reset occurs at switches.

Maximum discrepancies: state/observation 1.43e-13, estimate 1.38e-14, task metric
1.99e-13, selected plan cost 2.28e-13, projected gradient 2.43e-12, warm-start cost
1.37e-12, training fit SSE 4.50e-10. No optimizer fallback occurred. Pilot/final
training data and all fitted-model records are exactly identical; optimizer
repair did not change the physical learning data.

## Stronger PD baseline repair

The continuous-PD extension is explicitly exploratory after observing that the
initial gain grid was weak. It uses Powell gain tuning on the learned model,
including an active-cost objective and a separately declared stop-aware weight-5
objective, with extra offline compute. Neither objective queries the real plant.
The original coarse candidate is retained. Nonconverged Powell records are not
discarded; 38 such records occur, and neither convergence nor selected gains
certify global gain optimality.

An independent audit recomputes all **288** saved virtual candidate objectives,
chosen gains, **192** additional physical test episodes, command/estimate traces,
every task metric and cluster summary. Pilot and final training fits agree
exactly. State/observation/metric discrepancies are <=7.11e-14; commands,
estimates and summaries agree exactly. The largest virtual-objective absolute
difference is 1.28e-8 in the independently ordered simulation; all checks pass.

## Interpretation limits

The nominal MPC is one specified untuned model, not every robust or adaptive
controller without learning. Linear fitted MPC matches or outperforms nonlinear
fitted MPC in several cells. Physical pole names are unnecessary for ordinary
command objectives when predictions agree. No general superiority of nonlinear
model complexity or component attribution follows.

Poor stopping with true parameters and improved stopping with true states
implicate estimation/control design, but do not prove an information-theoretic
impossibility or isolate estimation as the only cause. Even true-state MPC fails
some tasks; finite-horizon objective, constraints, noise and stopping criterion
remain consequential. Repaired PD has a supplied restricted gain family and
exploratory tuning, so these experiments do not compare every strong feedback
alternative or establish a new general learning mechanism.

No unresolved final implementation discrepancy remains in the audited artifacts.
