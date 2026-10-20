#!/usr/bin/env python3
"""Build audited v10 once, preserving the complete v9 history."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
prior=(root/'first_principles_robot_learning_ledger_v9.md').read_text()
data=json.loads((root/'goal_control_v10_results.json').read_text())
pd=json.loads((root/'goal_control_PD_v10_results.json').read_text())
doc=r'''# First-Principles Robot Learning: Living Research Ledger v10

**Date:** 2026-10-20, Europe/Berlin. **Status:** constructed and independently audited experience-to-control pipeline; retained failed numerical pilot and stronger baseline repair. Complete v9 and all previous ledgers follow verbatim. Prior v9 commit `426524a38a041b7e566fece00831b073cfdddab6` was pushed and verified on existing SSH origin/main before this cycle. The persistent scientific Goal remains ACTIVE.

## Outcome: useful behavior without resolving physical names

**TESTED:** Conventional output-error identification from noisy position, state filtering and bounded receding-horizon planning form a working experience-to-goal-action pipeline on four declared actuator-memory plants. Controllers handle new goal sequences without resetting hidden velocity or motor force at switches. Learned linear models already track well, including the nonlinear plants. This is construction with supplied dynamics families and standard methods, not discovery of a general architecture.

**TESTED NEGATIVE:** On the nonlinear plants, nonlinear models' large improvement in clean command-sequence prediction does not reliably improve the selected closed-loop tracking objective. Accurate physical model parameters are not the only bottleneck: true-parameter filters have stopping failures too, and more informative true-state access changes performance. Finite-horizon planning, state estimation, process noise and the success definition remain distinct limitations.

**CORRECTED BEFORE RELEASE:** Initial L-BFGS-B planning frequently failed to converge, even in convex linear cases. The complete failed source/results remain. Replacing only the action optimizer with bounded least squares removes all final convergence failures and restores linear pole-swap invariance numerically. A weak coarse PD grid was also challenged by broader continuous tuning using only learned dynamics; it improves some comparisons but remains imperfect. Neither correction is concealed or called a preregistered method contest.

The strongest next uncertainty is whether obtaining more training examples is useful when current state information and future disturbances dominate goal precision. Adding active model-disagreement machinery is premature until this is resolved.

## First-principles question and a fixed experimental contract

V8/v9 showed that component labeling can be impossible even while command consequences are predictable. Do useful actions actually require those labels? Conversely, does a more accurate nonlinear predictor necessarily make better goal decisions? Test behavior directly, rather than score hidden parameter names.

The physical plant and observation channel are

~~~text
vnext=rho*v-c*tanh(v)+m
mnext=lambda*m+beta*u+xi
xnext=x+vnext, o=x+e.
~~~

Four plants: rho=.65, beta=1.2, lambda=.2 or .7, c=0 or .2. Units/timestep/tanh scale, model order, lagged timing and independent Gaussian noise laws are supplied. Training has no process noise and position-sensor SD .2. Test active control has motor-process SD .03 and sensor SD .2. Every episode begins at known x=v=m=0; there is no hidden-state reset at goal changes. These assumptions remain substantial built-in structure, not discoveries.

Each plant has12 independent fit seeds `20261020+1700000+10000*plant_index+100*i`. For each, generate512 commands in[-1,1] by choosing independent uniform amplitudes held for randomly selected1,3 or8 steps, truncated at512. Both model families receive **identical data within plant/seed**:513 noisy positions and512 commands. Command streams differ across plants. Neither family receives latent v/m, simulator parameters, sensor realizations or test outcomes.

Learned nonlinear OE uses the frozen v9 four-parameter fitting procedure. Learned linear OE fixes c=0, fitting its two poles and beta by the same raw-position objective, parameter bounds, three starts and evaluation budget. Linear labels are not scored. All fit objectives, starts and chosen models are saved. The experiment supplies the function vocabulary; no operator discovery, automatic model-order growth or online parameter revision is demonstrated.

## F44. The decision-relevant accuracy requirement [PROVEN UNDER ASSUMPTIONS]

At a given observable history h, let J(a) be the true expected trajectory loss for each admissible future action sequence, and Jhat(a) a learned estimate. If `sup_a |Jhat(a)-J(a)|<=epsilon` and the selected action sequence is within delta of the **global** infimum of Jhat, then

~~~text
J(selected) <= inf_a J(a)+2*epsilon+delta.
~~~

Insert Jhat on either side and use the uniform bound twice; an approximate infimum sequence gives the same inequality if a minimizer does not exist. No physical-coordinate identification is needed for this statement. But a one-step RMSE on a sampled distribution does not supply the uniform bound over sequences selected by optimization, and a projected gradient is not a global delta certificate for a nonconvex problem. This elementary established approximate-optimization fact is not a new learning theorem.

V10's planner uses a deterministic certainty-equivalent rollout with a fixed current goal; it does not compute the true noisy expected loss or anticipate unknown future goal switches. No tested epsilon/delta or repeated-history error bound is supplied. Thus the proposition explains the uncertainty that matters; it is not a guarantee for this implemented closed-loop controller.

## F45. Exact linear prediction/control need not choose a pole label [PROVEN / TESTED]

At c=0 let `q=m+(rho-lambda)*v` and swap rho/lambda. With s=[x,v,m], A=[[1,rho,1],[0,rho,1],[0,0,lambda]], B=[0,0,beta]^T and C=[1,0,0], the triangular transform T obeys `A_swapped*T=T*A`, `T*B=B`, `C*T=C`, and `T*Q*T^T=Q` for motor-only Q. Transforming prior mean/covariance therefore preserves Kalman observation/velocity predictions. Exact known-zero initial state/covariance already maps correctly.

Every planned loss using x,v,u and the same zero-command tail is invariant. Penalizing arbitrary hidden m instead would break that property without defining a physical task. Independently derived similarity/filter identities and a paired implementation check verify this. Final swapped-label MPC actions and physical states differ by about1.63e-14. The initial truncated solver differed by .010663 in action; the exact objective had never changed. Numerical optimization artifacts do not provide evidence identifying the poles. Nonlinear EKF is not claimed invariant to arbitrary nonlinear coordinate changes.

## Controller construction and fair access

All predictive controllers use the same eight bounded action variables followed by12 prescribed zero commands, minimizing the sum of `(x-goal)^2+.3*v^2+.03*u^2` across the20 predicted transitions. Only the first chosen command is executed; the remaining plan shifts for the next warm start. The planner sees only the current goal. It gets no future target schedule or true state except in the explicitly labeled state oracle.

Action sensitivities follow the actual lagged timing: current u changes only mnext, not next x/v; its first position effect is two steps later. Final action optimization uses bounded residual least squares (TRF), analytic Jacobians, max80 evaluations, ftol/xtol1e-10 and gtol1e-8. Every selected8-action plan, warm/selected cost, projected-gradient norm, termination record and fallback is retained. This solves the linear convex cases accurately; nonlinear global optimality/stability is not established.

The observer is a conventional EKF with state[x,v,m], position measurements, known noise scales and exact-zero initial mean/covariance. Measurement update precedes command selection; prediction follows the issued command. The transition Jacobian is `[[1,f'(v),1],[0,f'(v),1],[0,0,lambda]]`, Q=diag(0,0,.03²), and position covariance uses a Joseph update. Estimated parameters remain frozen during tests. Parameter uncertainty and nonlinear future noise effects are ignored by this certainty-equivalent design.

Comparators are fixed PD; PD tuned offline on the learned nonlinear model; learned linear and nonlinear MPC; and a fixed nominal-parameter MPC(.6,.4,.1,1) with no training. The nominal comparison concerns that specific mismatch, not every robust no-learning controller. Two privileged diagnostics have true plant parameters with either the same observation-only EKF or the full current physical state. They are not same-information learning competitors.

PD uses causal smoothed position and a second smoothed difference, then clips `kp*(goal-smoothed_position)-kd*smoothed_velocity` to[-1,1]. The initial18-gain grid is kp=.08,.2,.5, kd=.2,.6,1.2, smoothing=.25,.6. Its two synthetic goal schedules are(1,-1,.5) and(-1,1,-.5). Each virtual rollout uses only the learned nonlinear model and supplied noise law; **no extra physical observations** tune PD. Thus this is model-tuned feedback, not a model-free baseline. It optimizes active tracking, whereas the MPC surrogate also includes predicted coasting; stopping comparisons alone cannot establish a planning advantage.

After poor coarse-grid PD pilot results, an exploratory baseline repair broadens bounds to kp[.002,.5], kd[0,4], smoothing[.05,.95], with Powell optimization from the coarse winner and(.02,1,.8), up to350 evaluations per start. Selection also retains the coarse candidate. One version tunes active tracking, another adds five times normalized late squared position/velocity error, explicitly testing a stopping-aware objective. All tuning remains virtual on the same fitted model and separate goal schedules. This increases offline compute, not physical data. Thirty-eight Powell starts hit their evaluation limit and are retained. Gain optima are not certified globally, and no claim against the best possible PD controller follows.

## E26. New goal sequences, active disturbances and genuine coasting

Each fitted model is tested on goal sequences(2,-2,1) and(6,-4,3), revealed at steps0,6,28. There are70 active steps, then16 forced zero commands at the final goal. Sensors remain noisy throughout. Motor disturbances stop during the final coast so the score tests stored motion/actuation rather than a continuing uncontrollable random walk. The last active innovation remains in motor state; the coast is not a known-rest reset. Turning disturbance off is a declared diagnostic evaluation intervention, not a universal real-world stopping guarantee.

Test seeds are fitseed+2000000+task_index. Exogenous sensor/motor streams are paired across controllers, but their actions and observation histories differ. No test goal or noise is used for fitting/tuning. There are96 task trials per controller, with two sharing each of48 learned models. The main run retains672 episodes; the stronger PD repair adds192. Arrays of actual position, velocity, hidden motor state, observations, estimated states, commands and goals are saved. True state enters only the physical simulator/scorer and the explicitly privileged state oracle.

Frozen primary score: mean active `(position_error²+.3*v²+.03*u²)` divided by `1+mean(active_goal²)`. Stop success requires position error<=.3 and |v|<=.15 for every final5 active and16 coast samples. Also report late position/velocity error, action energy and settling diagnostics. The saved settling delay is a **zero-based action-index delay**, not elapsed physical transitions. Success rates are empirical task counts, not calibrated confidence guarantees.

Each cost mean first averages the two tasks within each fit seed, then averages12 independent seeds; MC SE uses those12 seed clusters. Counts out of24 share fitted models and are not24 independent Bernoulli fits.

{{COST}}

Cost table shows means. Full MC SEs and individual outcomes are in JSON. The broader PD repair substantially improves the coarse PD comparison in some slow-actuator cases, but can still be unstable or get stuck in poor synthetic tuning basins. This limits interpretations of relative PD performance.

{{STOP}}

Stopping table gives successes/24. True-parameter EKF and learned-model EKF behave similarly here. True-state access improves these particular controllers, but even that privileged controller can fail. No information-theoretic impossibility or uniquely causal attribution to state estimation follows just from these comparisons: planning horizon, objective, noise, filtering approximation and forced-coast timing all matter.

## E27. Prediction accuracy and control performance disagree [TESTED]

Each model also predicts a fresh clean256-command uniform[-1,1] replay from known rest, seedfit+3000000. This evaluation never selects models or gains. It measures prediction under a changed temporal command pattern, not the controller's future conditional objective.

{{PREDICTION}}

The nonlinear model improves command-replay prediction by large factors on c=.2, yet its primary control cost is slightly higher than the simpler learned linear model in both nonlinear cells. This is a finite-grid result, not a theorem that linear models suffice for general nonlinear robotics or that prediction is useless. It falsifies the inference that lower ordinary replay RMSE necessarily improves this control pipeline's selected task metric.

## F46. Even a true-state controller cannot know the final fresh motor innovation [PROVEN UNDER ASSUMPTIONS]

For c=0, condition on all information before the final active motor innovation xi69, including the chosen u69 and policy coins. At final coast time86, `x86-goal=mu+h16*xi69`, where `h16=(A^16)[0,2]`. xi69 is independent Gaussian SD .03; future process noise is zero. The probability of any length.6 interval under this conditional Gaussian is maximized when mu=0. Thus **any nonanticipating controller**, even a true-state one, has whole-stop success probability no larger than

~~~text
P(|x86-goal|<=.3) <= erf(.3/(sqrt(2)*.03*abs(h16))).
~~~

For lambda=.2/.7, h16=3.567238258/9.406435335, giving population bounds .994941569/.712264292. The whole event includes additional constraints and may be much less probable. Finite empirical fractions can exceed a population bound by chance. This result is specific to linear Gaussian noise, the fixed shutdown timing and success tolerance; it is not asserted for c=.2. It explains why even perfect model/state access does not imply100% strict coast success and why adding offline training cannot reveal that future noise sample.

## Independent audit, failed pilot, and existing work

The theory auditor independently derived lagged action sensitivities, filter/physical-label invariance, noise/coast counterexamples and the final-innovation bound; proof and scope are in `v10_theory_independent.md`. The artifact auditor independently eliminated motor state and reconstructed physical trajectories, observation streams, EKF estimates, PD virtual tuning, prediction scores, fit objectives, every task metric and clustered summary. It checked all33600 final selected plans, warm/selected costs and projected gradients from saved action vectors. Full audit sources/results are `audit_goal_control_v10.py` and JSON plus `audit_goal_control_v10.md`; the stronger PD audit is retained as `audit_goal_control_PD_v10.py` and its JSON.

The **failed pilot** source/complete JSON are `goal_control_v10_lbfgs_pilot.py` and `_results.json`, with independent audit JSON. It had large projected gradients, many40-iteration terminations and parameter-order-dependent actions despite identical linear objectives. After diagnosis, only the action optimizer changed; training rows/fits, plant grid, goals, bounds, noise streams, primary metrics and success tolerance remained fixed. The final33600 solves all report convergence, no fallback, maximum projected-gradient norm about3.17e-5. This is numerical verification, not a nonlinear global-optimum certificate. Predictions/plant replays and all negative outcomes remain available.

Before seeing results, the independent auditor also exposed two score-design confounds: a three-step coast misses slow stored force, and continued process noise during forced zero command measures uncontrollable drift. The frozen executed protocol uses12 planned and16 evaluated coast steps, and distinguishes noisy active control from disturbance-free coast. The additional continuous/stopping-aware PD search is explicitly **post-pilot exploratory baseline repair**, preventing a claim against an unnecessarily weak gain grid. No released history is overwritten.

**EXISTING RESEARCH:** [Tedrake's trajectory-optimization notes](https://underactuated.mit.edu/trajopt.html) describe direct shooting, exact gradients and receding-horizon control. [Nguyen et al., Output-Feedback MPC with Online Identification](https://arxiv.org/abs/2009.10631) is a serious observable-history alternative to a separate latent observer, including sensor noise and task-relevant identification; not implemented here. [Zhou and Tzoumas, Simultaneous System Identification and MPC with No Dynamic Regret](https://arxiv.org/abs/2407.04143) provides richer online learning/control under explicit technical assumptions and real quadrotor experiments. Their results do not validate ours; they establish how much broader existing work already is. Our construction is an elementary conventional baseline with known structure, not a new learning/control method or competitive robotics result. [Schoukens and Ljung](https://arxiv.org/abs/1902.00683) already distinguish model structure, noise treatment and application-specific validation.

## Reproduce and continue

Existing owner's PC and `.venv` only: Python3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl. No installation/services. Full outputs are intentionally retained, including the failed pilot, so the artifact files are larger than earlier releases.

~~~bash
.venv/bin/python research/goal_control_v10.py > /tmp/goal_control_v10_check.json
cmp research/goal_control_v10_results.json /tmp/goal_control_v10_check.json
.venv/bin/python research/goal_control_PD_v10.py > /tmp/goal_control_PD_v10_check.json
cmp research/goal_control_PD_v10_results.json /tmp/goal_control_PD_v10_check.json
.venv/bin/python research/audit_goal_control_v10.py > /tmp/audit_goal_control_v10_check.json
~~~

This cycle closes an earlier construction gap: acquired command consequences are used for actual goal-changing feedback control under hidden motor memory. It also rejects a tempting nonlinear-model success narrative, and independently catches numerical and weak-baseline confounds. But it supplies the state dimension, observation model, noise scales, function family, controller class, objective and exploratory data. No autonomous language expansion, coupled articulated body, unknown observation law or competitive modern general-learning framework has been demonstrated.

**Immediately next:** Test whether more offline experience versus more informative current sensing changes the observed precision failures. Begin with the exact linear Gaussian case: derive the best conditional uncertainty after all observation history, propagate it through the stopping decision, and compare its information limit to actual learned/oracle controllers. Then run an equal-protocol data-versus-sensing ablation, keeping model family and planner fixed. Additional samples cannot reveal a future innovation, but finite learned-model error and recoverable state uncertainty need distinct tests. Avoid interpreting another parameter fit as a solution to the task; the question is which available experience or measurement changes useful decisions. Broader mechanism construction remains OPEN, and the Goal stays active after publication.

---

# Historical archive: Complete original v9 ledger, including all preceding versions

'''
rows=['| lambda | c | Nominal MPC | Linear learned MPC | Nonlinear learned MPC | Continuous PD active |','| ---: | ---: | ---: | ---: | ---: | ---: |']
for p,q in zip(data['plants'],pd['plants']):
    r,lam,c,b=p['true_theta'];s=p['summary']
    vals=[s[k]['normalized_tracking_cost']['mean'] for k in ('nominal_MPC','linear_MPC','nonlinear_MPC')]
    vals.append(q['summary']['continuous_PD_active']['normalized_tracking_cost']['mean'])
    rows.append(f'| {lam} | {c} | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
doc=doc.replace('{{COST}}','\n'.join(rows))
rows=['| lambda | c | Linear MPC | Nonlinear MPC | PD active / stop-aware | True-model EKF | True-state oracle |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for p,q in zip(data['plants'],pd['plants']):
    _,lam,c,_=p['true_theta'];s=p['summary'];z=q['summary']
    rows.append(f"| {lam} | {c} | {s['linear_MPC']['stop_successes']}/24 | {s['nonlinear_MPC']['stop_successes']}/24 | {z['continuous_PD_active']['stop_successes']}/24 / {z['continuous_PD_stop']['stop_successes']}/24 | {s['oracle_model_EKF']['stop_successes']}/24 | {s['oracle_state_MPC']['stop_successes']}/24 |")
doc=doc.replace('{{STOP}}','\n'.join(rows))
rows=['| lambda | c | Linear replay velocity RMSE | Nonlinear replay velocity RMSE | Linear minus nonlinear control cost +/- paired SE |','| ---: | ---: | ---: | ---: | ---: |']
for p in data['plants']:
    _,lam,c,_=p['true_theta']
    pred=[np.mean([r['clean_prediction'][k]['velocity_rmse'] for r in p['runs']]) for k in ('linear','nonlinear')]
    delta=np.array([np.mean([t['controllers']['linear_MPC']['normalized_tracking_cost']-t['controllers']['nonlinear_MPC']['normalized_tracking_cost'] for t in r['tasks']]) for r in p['runs']])
    rows.append(f'| {lam} | {c} | {pred[0]:.6f} | {pred[1]:.6f} | {delta.mean():.6f} +/- {delta.std(ddof=1)/np.sqrt(len(delta)):.6f} |')
doc=doc.replace('{{PREDICTION}}','\n'.join(rows))
assert '{{' not in doc
out=root/'first_principles_robot_learning_ledger_v10.md'
if out.exists():raise SystemExit('Refusing to overwrite released v10')
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
