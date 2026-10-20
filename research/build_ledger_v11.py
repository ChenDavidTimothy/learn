#!/usr/bin/env python3
"""Build v11 once, preserving all v10 history verbatim."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
prior=(root/'first_principles_robot_learning_ledger_v10.md').read_text()
info=json.loads((root/'precision_information_v11_results.json').read_text())
deadline=json.loads((root/'deadline_control_v11_results.json').read_text())
doc=r'''# First-Principles Robot Learning: Living Research Ledger v11

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited information bounds, data/sensing ablation and a consequential correction to task-framing interpretation. Full v10 and every earlier ledger follow verbatim. V10 was published and verified on SSH origin/main at `450272cc1f34282b7dd2521db3972280982c2c7e` before this milestone. The persistent general-learning Goal remains ACTIVE.

## Principal result: question the task contract before adding learning machinery

**TESTED NEGATIVE:** In the selected slow linear actuator case, extending the same physical training trajectory from512 to8192 commands barely changes this controller's performance. Reducing current position-sensor SD tenfold helps, but strict stopping remains unreliable. The model family, noise laws and initial state remain supplied.

**PROVEN UNDER ASSUMPTIONS:** Known-parameter Gaussian conditioning gives a controller-independent terminal uncertainty floor and a strict upper bound on the protocol's stop probability. Even perfect position history cannot reveal the newest motor disturbance before its effect reaches position. Longer history and more offline training cannot recover that independent unseen innovation.

**CORRECTED INTERPRETATION / TESTED REPAIR:** Those valid bounds did **not** explain most of the original endpoint error. The executor already encoded a step70 cutoff, but its planner was not passed that constraint: it kept optimizing actions that would never be executed. Independent decomposition found that avoidable conditional-mean error dominated the average predicted endpoint MSE over the retained histories. Passing the already available executor deadline into the planner and removing unavailable planned actions reduces that bias by orders of magnitude, using the same model, data, sensors and optimizer. Merely increasing the predicted coast from12 to16 changes almost nothing. No new physical state or nonlinear feature is needed for this repair.

This extends v10's evidence rather than rewriting it: its actual numerical comparisons remain valid for its frozen pipeline, but its forced-coast results must be read as involving **unannounced command cutoff**. They do not establish an optimal stopping limit or a pure sensing failure. The earlier suggestion to prioritize sensor-selection machinery is superseded until planning actually respects the task's available actions.

The scientific gain is a falsified failure diagnosis and a constructive context repair, not a novel control algorithm or completion of general intelligent learning.

## Why this is the next question

V10's learned and true-parameter filters behaved similarly, while full-state diagnostics improved stopping. It was tempting to attribute the residual mainly to sensing. First distinguish three possibilities: inaccurate learned dynamics, genuine uncertainty unavailable in the current history, and a wrong description of the future task/action interface. They imply different repairs. More training, better sensors, and a corrected task contract are genuinely competing explanations here.

Use only the simplest case that supports exact uncertainty analysis: rho=.65, lambda=.7, c=0, beta=1.2 with lagged motor timing, initial x=v=m=0. This case was selected **after** v10's failures for explanation, not as a new independent benchmark for a general performance claim. Analytic tables also include lambda=.2; new control trials use only lambda=.7.

Physical law: `vnext=rho*v+m`, `mnext=lambda*m+beta*u+xi`, `xnext=x+vnext`, `o=x+e`. Training has xi=0, sensor SD .2. Active testing uses independent motor SD .03; after70 actions the next16 commands are zero and no new motor noise enters. The last active innovation remains. Future goals are not announced; the current goal sequence is revealed at steps0,6,28 as in v10.

## F47. Conditional information at the last possible action [PROVEN UNDER ASSUMPTIONS]

Let s=[x,v,m], A=[[1,rho,1],[0,rho,1],[0,0,lambda]], B=[0,0,beta]^T, Q=diag(0,0,sigma_xi²), C=[1,0,0]. Let H69 include **all position measurements through o69**, issued commands, independent policy coins and the supplied task information. The decision u69 is made using H69. There are no later nonzero commands.

With known parameters/initial state and independent Gaussian process/sensor noise, conditional s69 given H69 is Gaussian with mean mu69 and deterministic posterior covariance P69. This remains true for arbitrary causal observation-based control: condition on the observed history so each earlier input is a known shift; Gaussian prediction/update induction supplies the conditional law and Riccati covariance. Controls do not change this covariance in this declared linear additive-noise channel. An extra private sensor, anticipatory disturbance information or a state-dependent noise/measurement law would change the argument.

After one last active transition and16 disturbance-free coast transitions,

~~~text
E[x86-goal | H69] = C*A^17*mu69 + C*A^16*B*u69 - goal = b
V = C*A^17*P69*(A^17)'*C' + C*A^16*Q*(A^16)'*C'.
~~~

Thus `E[(x86-goal)^2|H69]=b²+V>=V`, and

~~~text
P(whole stop event | H69) <= P(|x86-goal|<=.3 | H69)
                        <= erf(.3/sqrt(2V)).
~~~

The interval probability is maximized at b=0. V is a lower bound on conditional MSE, not necessarily an achievable minimum with bounded commands. Even feasible endpoint centering does not optimize the whole path/velocity event. Future coast observations cannot retroactively improve the already-issued u69; the history cutoff is essential. These are elementary Gaussian-conditioning/control facts, not new theory or an optimality certificate for MPC.

{{LIMITS}}

The bounds are population upper bounds; finite trial fractions can exceed them by chance. They apply to the exact linear Gaussian protocol, not arbitrary nonlinear robotics or unknown noise laws. Knowledge of true parameters is more informative than finite independent training about those same parameters, so additional offline samples cannot remove these particular live-innovation terms.

## F48. Perfect positions still arrive too late for one motor innovation [PROVEN UNDER ASSUMPTIONS]

With exact x0:69 and known parameters, v69=x69-x68 and m68=v69-rho*v68 are exact. But

~~~text
m69=lambda*m68+beta*u68+xi68.
~~~

xi68 has not yet affected x69. Therefore P69=diag(0,0,sigma_xi²), even with arbitrarily long noiseless position history. Terminal error contains two independent unobserved innovations: xi68 with coefficient `(A^17)[0,2]` and xi69 with `(A^16)[0,2]`. For lambda=.7 this gives terminal SD .399789 and whole-stop probability upper bound .546984, versus .712264 with full current physical state known. Position precision cannot remove this timing gap. A timely additional motor-force measurement or changed action opportunities supplies different information; it is not equivalent to more precise positions. This conclusion depends on fresh independent motor noise and the declared lagged interface, not a universal requirement for force sensors.

## E28. More experience versus current sensing [TESTED]

`precision_information_v11.py` retains the exact12 v10 slow-linear fit seeds and all original512 commands/513 noisy positions. Each physical trajectory continues to8192 commands without a state reset. The extension uses independent seeded bounded held commands(1,3,8 steps), seedfit+4000000, and sensor SD .2. All old observations stay unchanged; the new first observation is at time513, so the boundary is neither duplicated nor overwritten. Raw-position linear OE is refit using the same bounds/starts as v10. These extra samples are a sixteenfold interaction increase, not a same-budget method comparison.

With short fits, long fits and true parameters, test position-sensor SD .2 or .02. The sensing intervention changes both actual noise and the supplied filter variance, with identical underlying standard-normal draws and motor noises. Training sensor SD stays .2. Same goals, test seeds, bounds, controller/optimizer and scoring as v10; no true state to ordinary controllers. There are144 episodes/10080 plans,12 fit-seed clusters; two task trials share a fit. The added training and improved sensor's physical/economic costs are not equated.

{{DATA}}

Cost means/MC SEs average tasks within seed before summarizing12 seeds. Late position RMSE combines the final10 active and16 coast states, while endpoint MSE below concerns time86 alone; these metrics must not be conflated. All per-episode trajectories/selected plans,8192-row training data and fit records are retained. The .2 short-fit and known-parameter episodes exactly reproduce v10's counterparts.

Increasing data changes mean active cost by +.0000173 +/- .0000281 paired MC SE at sensor SD .2; this is evidence of little benefit in this case, not a formal universal equivalence result. Lowering sensor SD with the short fit changes it by -.0124751 +/- .0043141. Whole-stop outcomes remain0/24 or3/24; the small counts do not establish calibrated success guarantees or general benefit from sensor upgrades.

## F49. An uncertainty floor is not the cause of all observed error [DERIVED / TESTED]

The fixed-horizon planner at time69 still optimizes eight future commands before its predicted zero-command tail. The actual protocol applies u69 and then forbids all commands. Neither MPC's action function nor its fixed horizon receives the cutoff deadline. This is a planner/executor constraint mismatch in our benchmark implementation: future controls are being optimized that will not be available.

Independently re-filtering saved known-parameter episodes gives the b from F47. Over the24 retained known-parameter histories at each sensor level, at SD .2 mean b²=1.676571 while V=.343423: about83% of predicted endpoint MSE b²+V is conditional-mean error. At SD .02, b²=.603351 versus V=.193027, about76% is conditional-mean error. These are finite averages of conditional biases/risks, not exact population expected MSEs. Neither valid uncertainty floor explains that excess. Observed0/24 or3/24 whole stops cannot be attributed solely to those endpoint ceilings.

Counterfactually replacing just the final command by `u69-b/(C*A^16*B)` centers the conditional endpoint mean; every checked known-parameter case permits this within[-1,1]. This is a diagnostic holding earlier history fixed, not a claim that the replacement meets every stop constraint, is optimal for the full loss, or could use future information. It demonstrates that a substantial part of the existing bias was avoidable with the already available measurements.

## E29. Announce the deadline and remove unavailable planned actions [TESTED]

`deadline_control_v11.py` compares two added variants, both predicting16 zero commands after the planned control sequence. One keeps eight planned actions. The other receives the known time70 cutoff and uses `N=min(8,70-t)` action variables, truncating the warm plan causally. At t69 the one optimized command followed by16 zeros matches the actual available action schedule. All state/filter/smoother variables continue normally; no reset is introduced.

Use the same12 seeds, both goals, sensor SD .2/.02, short512-fit or true parameters, with no additional training. This adds192 episodes and13440 bounded optimizations. Only current goals and, for the aware variant, the shutdown deadline are supplied; future goal changes remain hidden. The shutdown time was already encoded in the executor. Passing it through the planning interface uses a supplied task constraint; it acquires no new external physical information, learns no new law and improves no sensor. The comparison is an exploratory diagnosis after v10, not an independent preregistered algorithm contest.

{{DEADLINE}}

The original12-step predicted coast and fixed-eight/16-step coast have nearly identical results. Removing unavailable future controls changes the conditional mean dramatically while **leaving V unchanged**. The learned512-row and known-parameter variants remain similar. This refutes the immediate need for additional physical model complexity or many more training examples to correct this particular bias.

The primary active tracking cost changes little because it largely scores the earlier70-step tracking phase. Large improvements in coast error need a separate task score; optimizing/reporting only a convenient primary scalar can obscure a capability failure. Stopping still fails on many trials; conditional uncertainty, path constraints, the finite quadratic objective and bounded/causal control remain. The scalar terminal bound is not a complete characterization of achievable whole-stop probability, and the deadline-aware controller is not asserted optimal.

## Audit, corrections, and existing research

Independent theory derived the adaptive-policy conditioning, exact-position two-innovation floor, and the b²+V decomposition. It challenged the initial sensor-first interpretation and independently re-filtered saved data in `audit_precision_cutoff_v11.py`/JSON. `v11_theory_independent.md` preserves the earlier suggestion and its explicit correction, rather than silently replacing research history.

The artifact auditor independently constructed batch Gaussian conditioning, including Moore-Penrose/nullspace handling for exact position measurements, rather than just rerunning the same Riccati loop. It reconstructed nested trajectories without hidden-state resets, all144 experience/sensing episodes,10080 plans, filter histories, metrics and clustered summaries; source/results/report are `audit_precision_information_v11.py`/JSON/Markdown. Deadline-aware experiments and conditional-mean diagnostics are independently reconstructed in `audit_deadline_control_v11.py`/JSON and its audit report. Audit scope distinguishes fresh derivations/implementations from repeated execution.

**CORRECTED BEFORE RELEASE:** A JSON field originally named minimum conditional MSE was renamed conditional MSE **lower bound**, since bounded centering need not be feasible generally. Known-parameter variance is not a success guarantee. The unannounced-cutoff issue corrects interpretation of v10/v11 initial stopping failures; it does not invalidate the actual saved numbers or Gaussian bounds. The final source/result metadata records audit completion; the earlier experiments and released ledgers remain preserved.

**EXISTING RESEARCH:** [Kalman1960](https://people.math.harvard.edu/archive/116_fall_03/handouts/Kalman1960.pdf) establishes recursive estimation-error covariance and optimal linear prediction. [Georgiou and Lindquist, The Separation Principle in Stochastic Control, Redux](https://arxiv.org/abs/1103.3005) examines the subtleties of information under feedback; our finite discrete-time induction explicitly supplies the history/causality conditions. [Tedrake's MPC/direct-shooting notes](https://underactuated.mit.edu/trajopt.html) describe planning with specified dynamics/constraints. These precedents preclude novelty claims. Gaussian filtering or LQG references do not turn this bounded, fixed-horizon, multi-step stopping event into an optimal-control theorem. No advantage over modern general learners is established.

## Reproduction and drift audit

Owner's existing `.venv`: Python3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl. No installation or service changes.

~~~bash
.venv/bin/python research/precision_information_v11.py > /tmp/precision_information_v11_check.json
cmp research/precision_information_v11_results.json /tmp/precision_information_v11_check.json
.venv/bin/python research/deadline_control_v11.py > /tmp/deadline_control_v11_check.json
cmp research/deadline_control_v11_results.json /tmp/deadline_control_v11_check.json
.venv/bin/python research/audit_precision_information_v11.py > /tmp/audit_precision_information_v11_check.json
.venv/bin/python research/audit_precision_cutoff_v11.py > /tmp/audit_precision_cutoff_v11_check.json
.venv/bin/python research/audit_deadline_control_v11.py > /tmp/audit_deadline_control_v11_check.json
~~~

This cycle resolves an operational uncertainty relevant to learning: when a task fails, extra experience or model capacity may be irrelevant because the learner/planner is solving the wrong future-interaction problem. The experiments distinguish actual parameter uncertainty, live hidden innovations and omitted action-availability context. Correcting the latter is simpler and more effective than building new sensing-policy machinery in this case.

The mechanism still does not autonomously discover the failure diagnosis or task metadata; independent scientific investigation supplied the correction. It does not learn an observation language, select among unknown physical families, infer preferences from nothing, or demonstrate nonlinear articulated-body intelligence. More isolated covariance identities would risk drift. Further work should test whether acquired predictive consequences support reliable decisions when the function vocabulary, coupled dynamics or action availability must themselves be inferred from experience, using strong established alternatives rather than imposing an architecture prematurely.

**Immediately next:** Reassess the complete learning loop beyond this supplied scalar family. Preserve the distinction between learned experience and supplied task/interface facts. Prefer a decisive test of which missing distinction actually changes successful action: task-context inference across episodes or coupled dynamics requiring model revision are legitimate alternatives. Derive the simplest discriminating case first, compare conventional simple mechanisms, and only add complexity if the simpler explanation fails. The broader Goal remains active; publication is not a stopping condition.

---

# Historical archive: Complete original v10 ledger, including all preceding versions

'''
rows=['| Position sensor SD | Terminal conditional SD | Endpoint/whole-stop ceiling |','| ---: | ---: | ---: |']
for x in info['information_limits']:
    if x['lambda']==.7:
        rows.append(f"| {x['sensor_sd']} | {x['terminal_position_conditional_sd']:.6f} | {x['whole_stop_population_probability_upper_bound']:.6f} |")
doc=doc.replace('{{LIMITS}}','\n'.join(rows))
rows=['| Test sensor SD | Model knowledge | Active cost +/- clustered SE | Late position RMSE | Whole stops |','| ---: | --- | ---: | ---: | ---: |']
for sd,d in info['summary'].items():
    for name,s in d.items():
        c=s['normalized_tracking_cost']
        rows.append(f"| {sd} | {name} | {c['mean']:.6f} +/- {c['mc_se']:.6f} | {s['late_position_rmse']['mean']:.6f} | {s['stop_successes']}/24 |")
doc=doc.replace('{{DATA}}','\n'.join(rows))
rows=['| Sensor SD | Same known-parameter model / metadata | Late position RMSE | Mean conditional endpoint bias squared | Whole stops |','| ---: | --- | ---: | ---: | ---: |']
for sd,d in deadline['summary'].items():
    for name in ('known_parameters_unaware16','known_parameters_aware16'):
        s=d[name]
        rows.append(f"| {sd} | {name} | {s['late_position_rmse']['mean']:.6f} | {s['mean_squared_conditional_endpoint_bias']:.6f} | {s['stop_successes']}/24 |")
doc=doc.replace('{{DEADLINE}}','\n'.join(rows))
assert '{{' not in doc
out=root/'first_principles_robot_learning_ledger_v11.md'
if out.exists():raise SystemExit('Refusing to overwrite released v11')
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
