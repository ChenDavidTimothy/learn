#!/usr/bin/env python3
"""Build v12 once, preserving v11 and its complete lineage verbatim."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
prior=(root/'first_principles_robot_learning_ledger_v11.md').read_text()
main=json.loads((root/'shared_experience_v12_results.json').read_text())
reuse=json.loads((root/'model_reuse_v12_results.json').read_text())
doc=r'''# First-Principles Robot Learning: Living Research Ledger v12

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited shared-experience comparison, observation-only loss derivation, negative same-family model-reuse diagnostic, and preserved seed correction. Complete v11 and all earlier ledgers follow verbatim. V11 was published at `90b7e9b8198a1854982588a2ed3553c51681129c`; the subsequent audited preparatory pilot/handoff was published at `6dc0c5037b5f23ee62ed04e07914ac595a7fc6bb`, both verified on the existing SSH origin/main. The general-learning Goal remains ACTIVE.

## Principal findings: different uses of identical experience

**PROVEN UNDER ASSUMPTIONS:** A causal learner can rank expected goal behavior directly from noisy positions and its issued actions, without hidden-state reward or fitting dynamics, in this declared lagged system. The observed position/displacement quadratic loss differs from the true position/velocity loss by a policy-independent expectation offset. Timing, sensor independence/zero mean, and exogenous fixed-horizon objectives are essential. This is an elementary expectation identity, not a universal learning principle or novelty claim.

**TESTED:** Direct CEM search in a supplied six-parameter dynamic feedback family improves behavior. Conventional output-error models, EKF and bounded MPC fitted to the **exact same complete experience**, including poor policy trials, yield lower mean tracking cost at both retained interaction checkpoints on all four declared plants. The models do not receive true parameters or extra training episodes. The result compares specified mechanisms, with different supplied vocabulary and computation; it does not show that direct learning generally requires models or that these toy methods compete with modern robotics learners.

**TESTED NEGATIVE:** Repeating the identical direct CEM search inside the learned nonlinear model, with no extra simulator-plant training, does not consistently improve the same controller family. Thus the current evidence does not isolate model reuse, policy capacity, or finite search as the cause of the MPC advantage. Using the correct nonlinear family still does not consistently beat the learned linear model in tracking. Neither predictable behavior nor low average cost ensures precise station keeping.

**CORRECTED BEFORE RELEASE:** Expanding the first run's candidate budget accidentally exceeded its replicate seed spacing. Some adjacent training replicates reused sensor streams. Source inspection caught this before outcomes were reviewed; the full excluded run is retained, corrected disjoint seed domains are verified, and all reported inferential results use the rerun. No failed run was concealed.

## Why change the framing now?

The preceding sequence successfully used predictive consequences despite ambiguous physical labels, then corrected a planner/task-interface mismatch. That route could prematurely turn 'learning' into 'fit a supplied physical model.' Ask instead whether experience must be used that way. Directly improving behavior from observed consequences is a genuinely different possibility: no hypothetical dynamics, latent-state estimate, or online planner is required by its algorithm.

A fair discriminating experiment must separate data acquisition from its use. Comparing a direct learner's feedback traces to a model learner's specially designed excitation would change both. Here the predictive learner is a shadow of the direct search: it reads every action/observation already collected, respecting all episode resets. Its favorable result concerns reuse of the same data within strongly supplied families, not a necessary foundation of intelligence. This is a constructive learning-mechanism comparison, rather than another isolated scalar identification theorem.

## F50. Observation-only expected goal loss [PROVEN UNDER ASSUMPTIONS]

Use the lagged channel

~~~text
v[t+1]=f(v[t])+m[t]
m[t+1]=h(m[t],v[t],u[t])+xi[t]
x[t+1]=x[t]+v[t+1], o[t]=x[t]+e[t].
~~~

The sensor errors are fresh iid, zero mean, variance R<infinity, independent of initial state, physical noise and policy coins. Commands can depend causally on observations through the present. Goals, weights, horizon and normalization are fixed/exogenous, and expected losses finite. Because v[t+1] and x[t+1] depend on commands only through u[t-1], they are independent of e[t],e[t+1]. Expanding the squares gives

~~~text
E[(o[t+1]-g[t])^2+.3*(o[t+1]-o[t])^2+.03*u[t]^2]
 = E[(x[t+1]-g[t])^2+.3*v[t+1]^2+.03*u[t]^2]+1.6R.
~~~

For the fixed episode normalizer D=1+mean(g²), the mean episode offset is1.6R/D. Knowing R is unnecessary to rank expectations. No Gaussianity is needed for the identity; finite fitness variance generally needs fourth-moment conditions. Neighboring stage errors are dependent; independent training seeds, not timesteps, are the uncertainty units.

**Counterexamples / limitations:** Biased sensors add a policy-dependent position cross term. Colored sensing, state-dependent sensor laws, random stopping or data-dependent goals/normalizers can break the argument. With MOTOR-FIRST one-step v1=beta*u0 and bounded u0=-k*tanh(e0/SD), the full expected gap is `1.6R+.6*beta*k*E[e0*tanh(e0/SD)]`, dependent on k. The preserved pilot independently checks Gaussian quadrature and one million Monte Carlo samples: at SD.2/beta1.2, k0/.3/.9 gaps are.064/.09016648/.14249943.

The identity applies to a fixed causal policy on fresh noise. Adaptive repeated optimization of the same finite future-noise bank can invalidate that fixed-policy training interpretation. Even fresh noise per candidate leaves optimistic minimum selection. All final policies are frozen before fresh test episodes; their finite observed/true score gaps need not equal the expectation offset trajectory by trajectory. No hidden true-state reward enters policy selection.

## F51. Predictive fitting of feedback-collected experience [PROVEN UNDER ASSUMPTIONS]

Training has ZERO process noise and each episode begins at supplied known physical zero. Given theta and the issued command prefix, the position x_theta is deterministic. For the actual causal search/controller, the joint/sequential experiment likelihood factors as

~~~text
p_theta(o0:T,u0:T-1)
 = p_theta(o0) * product_t pi_t(u_t | recorded history through o_t)
                       * Normal(o[t+1];x_theta[t+1](issued prefix),R).
~~~

Conditional on recorded history, the actual action rule and preceding CEM updates are independent of the hypothetical physical theta. Deterministic policies supply theta-independent point masses on the recorded support. Therefore raw-position SSE over **all** known-reset segments is a valid sequential/joint Gaussian likelihood objective. Reset simulated states and parameter sensitivities at every segment; do not join separate physical resets into a continuous trajectory.

This does NOT assert that the likelihood conditional on the entire future command sequence is an iid Gaussian product: those future commands encode past sensor errors. It does NOT justify fitting only elites selected by subsequent outcomes. Nonzero unobserved process noise would require a latent-state likelihood or separate argument. Valid likelihood does not imply excitation, unique physical labels, a global optimum or successful decisions. These distinctions were derived independently in `audit_observation_loss_v12.md` and challenged again in `v12_comparison_design_independent.md`.

## E30. Preserved preparatory direct-learning pilot [TESTED, LIMITED]

`direct_experience_v12_pilot.py`/JSON and independent executable audit retain eight seeds on theta(.65,.7,.2,1.2). CEM searched a three-parameter supplied PD/smoothing class using8256 simulated-plant transitions per seed. Fresh test mean true cost.678067 +/- .185905 clustered SE;3/16 station-keeping events. A handpicked fixed PD was much worse, but was explicitly a weak descriptive comparator. This pilot alone did not warrant a new ledger or any method-superiority claim. It motivated a stronger dynamic class and an actual shared-data competitor.

## E31. Shared-experience controller comparison [TESTED]

Design was written before inspecting new outcomes in `shared_experience_v12_design.md`; its source-inspection seed correction is separately disclosed. Four lagged plants rho=.65,beta=1.2,lambda in{.2,.7},c in{0,.2}. Sensor SD.2; train motor process0; test motor process SD.03 on every step. Known zero reset per86-step episode. Current goals switch at6,28 without resetting any physical/controller memory. Feedback remains available throughout all86 test steps. This differs from v10/v11 forced-coast stopping and must not be called a solution of that task.

The direct policy is

~~~text
sx += alpha*(o-sx)
sv += alpha*((sx-sx_previous)-sv)
u = clip(kp*(goal-sx)-kd*sv+bv*sv_previous-bu1*u[t-1]-bu2*u[t-2],-1,1).
~~~

Six searched coordinates are logkp,logkd,logitalpha,bv,bu1,bu2. Bounds kp[.005,.5],kd[.02,4],alpha[.1,.98],remaining coefficients[-2,2]. It includes pilot PD exactly. In the alpha1 noiseless linear limit, m_t=lambda(v_t-rho*v[t-1])+beta*u[t-1]; clipped state feedback Kp*error-Kv*v-Km*m maps to kp=Kp,kd=Kv+Km*lambda,bv=Km*lambda*rho,bu1=Km*beta,bu2=0 when gain restrictions permit. Alpha1 is OUTSIDE the actual search bound. This useful limiting capacity check does not prove arbitrary observer/MPC containment, especially with noisy measurements or nonlinear drag. Positive signs, smoothing, vocabulary and finite bounds are supplied biases.

Diagonal CEM uses8 generations x24 candidates x2 actual simulated-plant training episodes, goals(1,-1,.5) and(-1,1,-.5);6 elites, fixed initial distribution/SD floors detailed in the design/source. Each candidate gets fresh noise; previous winner is retested as candidate0. Select current-generation winner, not a stale best-ever noisy score. No true state or test result enters selection. Save every poor trial and CEM draw/elite. Checkpoints after2/8 generations consume8256/33024 transitions and96/384 resets; the earlier checkpoint is a prefix, not extra acquisition. Eight training seeds per plant,32 clusters total.

At each checkpoint, shadow linear/nonlinear OE fits ALL observations/commands using separate reset segments, v9's three starts/bounds/max160 evaluations, analytic Jacobians, minimum raw SSE. Supplied families include the true one. Each model controls with EKF (supplied noise scales/zero prior) and bounded TRF MPC:16 free actions, truncated to min(16,86-t), current goal held constant in forecasts, no zero-command tail, same quadratic stage cost. Save fit records/curvature diagnostics and all plans/status/selected costs/gradients. All384 fit attempts and22016 action solves report convergence; success is not a global nonlinear optimality certificate. Maximum selected projected action-gradient magnitude is3.568e-5.

Fresh evaluations use goals(2,-2,1),(6,-4,3), common underlying noise across compared policies/checkpoints within seed, separate training/evaluation streams. Primary cost averages86 true-state stages then divides by1+mean(goal²). True states are external evaluator-only. Late10 station keeping requires all |position error|<=.3 and|velocity|<=.15. Means/SE first average both tasks within each training seed, then summarize eight seeds per plant. Counts have16 dependent-within-fit tasks; they are descriptive, not calibrated success probabilities.

{{MAIN}}

The final direct-minus-nonlinear-MPC paired mean differences are.046426+/-.008424,.057292+/-.011525,.135216+/-.009540,.082515+/-.014324 by plant order, with SE across eight seed clusters. These are Monte Carlo summaries for the fixed protocol, not a distribution-free superiority theorem. Learned linear and nonlinear control remain close; adding the correct nonlinear vocabulary does not consistently improve the task score. The original direct policy can have more station-keeping successes despite worse average cost; no single scalar certifies all capabilities.

**Metadata/capacity caveat:** MPC consumes remaining episode time; the direct policy is stationary without a clock feature. Both have current goals, bounds and reset/noise facts, but their internal vocabulary/use of metadata differs. The prescribed first70-step diagnostic avoids horizon truncation entirely: MPC still has its full16 actions through t70, so end shortening cannot causally explain prefix performance gaps. All four final prefix costs favor the model planners too; this rules out that one explanation, not capacity/compute confounding. The final direct command can only add effort, but broader differences cannot be reduced to this small single-action effect.

## E32. Same-family simulated experience reuse [TESTED NEGATIVE]

Before inspecting corrected outcomes, `model_reuse_v12_design.md` prescribed repeating exactly the six-coordinate CEM protocol on each learned final nonlinear model. Same training goals, sensor SD.2 and motor process0; independent virtual seeds,33024 additional VIRTUAL transitions per fit, no new simulated-plant training. Each selected stationary policy is tested on the same heldout actual-plant streams. This controls controller family and time features, but adds model-based computation and a strongly supplied physical vocabulary. Virtual-state access never supplies true plant-state reward.

{{REUSE}}

No consistent gain emerges from this one virtual-CEM restart; the large spread and eight seeds forbid treating small differences as equivalence. It does not prove simulation reuse is useless or the direct family incapable. Search can fail within an adequate family; a local simulation objective may not transfer; a model can be accurate but the chosen optimizer unhelpful. The MPC advantage therefore remains a result for complete specified mechanisms. One exact-gradient same-family capacity/search diagnostic is a defensible next investigation, after which move toward coupled/missing-vocabulary learning rather than endlessly polishing scalar gains.

## Audit, exact budgets, failures and reproduction

All plants are simulated on the owner's existing PC/.venv; these are not hardware robotics results. 'Plant experience' distinguishes actual benchmark environment transitions from rollouts inside a fitted model. The corrected shared experiment collects1056768 unique training transitions/12288 resets total. Its two-checkpoint test consumes33024 transitions/384 resets, never used to update a learner. The virtual-reuse diagnostic adds1056768 learned-model transitions/12288 virtual resets and5504 plant test transitions/64 resets, zero new plant training. Earlier PD pilot and excluded overlap pilot are additional development work, not hidden from the interaction record. Reproduction reruns repeat existing seeded instances and are not new independent scientific replicates.

The excluded first large run is `shared_experience_v12_seed_overlap_pilot.py`, full NPZ, gzip-compressed full JSON and timing. Its replicate spacing10000 overlapped candidate episode offsets spanning19101. It is not used for reported MCSEs. Corrected seeds use plantstride1000000000,replicatestride1000000,trainingoffset100000+100*index+task,testoffset500000+task,CEMbase; virtual runs add10000000000. Audit checks global role/replicate disjointness. True outcome similarity would not rescue the invalid independence assumption.

Independent theory actively challenged policy capacity, sensor assumptions, feedback likelihood and horizon metadata. The artifact audit uses separately implemented dynamics/controllers/sensitivities/filter/plan loss, reconstructs all24576 corrected/virtual training segments, all448 fresh test episodes and22016 plans, checks all384 fit records, CEM randomness/selection, seeds, budgets and clustered summaries. It distinguishes independent implementation from simply re-executing the source. The earlier PD pilot audit independently reconstructed800 episodes. Sources/results/reports are retained alongside the main experiments; no unresolved audit discrepancy may be treated as a release.

The raw generated metadata retains its original pre-audit execution status; the separate subsequent audit report/result records completed verification. The main and reuse experiments were each re-executed byte-for-byte for NPZ and result JSON. Timing is separately logged and is inherently nondeterministic. On the saved original corrected run, total wall sums are3.48s plant-training simulation,4.19s linear fits,5.79s nonlinear fits,15.29s linear-control tests,15.52s nonlinear-control tests; reuse virtual training3.43s. These are implementation-specific local measurements, not a fair runtime contest or scaling claim. BLAS threads limited to1; Python3.12.3,NumPy2.5.3,SciPy1.18.1, existing threadpoolctl; no environment changes.

~~~bash
.venv/bin/python research/shared_experience_v12.py --output-prefix /tmp/shared_v12
cmp research/shared_experience_v12_results.json /tmp/shared_v12_results.json
cmp research/shared_experience_v12_data.npz /tmp/shared_v12_data.npz
.venv/bin/python research/model_reuse_v12.py --output-prefix /tmp/reuse_v12
cmp research/model_reuse_v12_results.json /tmp/reuse_v12_results.json
cmp research/model_reuse_v12_data.npz /tmp/reuse_v12_data.npz
.venv/bin/python research/audit_shared_experience_v12.py
~~~

**EXISTING RESEARCH:** [de Boer/Kroese/Mannor/Rubinstein's CEM tutorial](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf) supplies established distribution/elite-based optimization. [Mania/Guy/Recht2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf) demonstrates serious simple policy search and emphasizes broad seed evaluation. Our algorithm is CEM, not ARS; eight-seed scalar cells are not comparable with their robotics results. OE, EKF, shooting/MPC and pathwise derivatives are conventional tools; no originality or modern benchmark superiority is claimed.

## Research direction audit and exact continuation

This milestone changes the evidence about constructed learning mechanisms: experience can improve behavior through observation-only scores, while the same full experience also supports conventional predictive fitting and better mean transfer under the present supplied model/planner. It does not establish a necessary architecture. It also gives a useful negative diagnostic: adding simulation rollouts with unchanged finite CEM is not automatically enough. Scientific consequences concern how data, vocabulary, computation and decision objectives must be distinguished when comparing learning.

The limitations are substantial: scalar fixed-family systems; known reset, control sign, order/timing/noise and functional vocabulary; no autonomously learned distinction language; no coupled bodies, unknown contact or genuine robot transfer; controller class/search not exhausted; no modern learner superiority. A release does not achieve the general-learning objective.

**Immediately continue:** One prescribed, independently checked exact-gradient optimization of the SAME controller family on learned dynamics, with fixed starts, virtual training/validation separation and NEW physical test streams, can constructively test whether finite CEM search is limiting. Success would show useful family capacity, failure would not prove incapacity. Do not add an unbounded ladder of optimizers. Then choose a consequential coupled or missing-function-vocabulary problem: derive competing minimal mechanisms, test from observation/action experience under explicit priors and strong applicable baselines, and retain negative evidence. Keep the Goal ACTIVE across publication.

---

# Historical archive: Complete original v11 ledger, including all preceding versions

'''
rows=['| lambda,c | Plant training steps/seed | Direct cost +/- SE | Linear OE/MPC cost +/- SE | Nonlinear OE/MPC cost +/- SE | Station keeping: direct/linear/nonlinear |','| --- | ---: | ---: | ---: | ---: | --- |']
for s in main['summary']:
    vals=[]
    for name in ('direct','linear','nonlinear'):
        v=s['controllers'][name]['cost'];vals.append(f"{v['mean']:.6f} +/- {v['mc_se']:.6f}")
    success='/'.join(str(s['controllers'][n]['stationary_successes']) for n in ('direct','linear','nonlinear'))
    rows.append(f"| {s['theta'][1]},{s['theta'][2]} | {s['generations']*24*2*86} | "+' | '.join(vals)+f' | {success} of16 |')
doc=doc.replace('{{MAIN}}','\n'.join(rows))
rows=['| Plant index | Model-reused same-family cost +/- SE | Original direct minus reused +/- paired SE | Station keeping |','| ---: | ---: | ---: | ---: |']
for s in reuse['summary']:
    c=s['reused_cost'];d=s['paired_original_direct_minus_reused']
    rows.append(f"| {s['plant']} | {c['mean']:.6f} +/- {c['mc_se']:.6f} | {d['mean']:.6f} +/- {d['mc_se']:.6f} | {s['stationary_successes']}/16 |")
doc=doc.replace('{{REUSE}}','\n'.join(rows))
assert '{{' not in doc
out=root/'first_principles_robot_learning_ledger_v12.md'
if out.exists():raise SystemExit('Refusing to overwrite released v12')
if not (root/'audit_shared_experience_v12_results.json').exists():raise SystemExit('Audit results required before building release')
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
