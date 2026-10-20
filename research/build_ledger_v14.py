#!/usr/bin/env python3
"""Build compact coupled-learning milestone once; preserve all v13 history."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
prior=(root/'first_principles_robot_learning_ledger_v13.md').read_text()
r=json.loads((root/'coupled_learning_v14_results.json').read_text())
doc=r'''# First-Principles Robot Learning: Living Research Ledger v14

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited first coupled articulated learning instrument, context-dependent relation transfer, negative extrapolation results and explicit measurement corrections. V13 was published and verified on the existing SSH origin/main at `aeb569a16688c7290af88169d654c0f809b6c2a6` before continuing. Complete v13 and all earlier ledgers follow verbatim. The broader general-learning Goal remains ACTIVE.

## Principal result: relation choice and context coverage both matter

**TESTED:** With broad pose experience, conventional generic context-conditioned regression acquires useful two-joint action/displacement knowledge and uses it for new displacement targets/poses, without receiving the true inertia formula or named physical parameters. A learned inverse relation performs better than the learned forward relation within the selected finite basis. This accepts established learning tools rather than claiming a new foundation or superior general robot learner.

**TESTED NEGATIVE:** Matched interaction budgets confined near one pose do not support broad-pose transfer. The more flexible context learners can extrapolate worse than the full coupled constant baseline; some inverse relations produce severe prediction outliers. Correcting known response-measurement noise helps inverse decisions in the broad condition but does not reliably improve the narrow condition’s out-of-support score. Statistical moment correction is not reliable extrapolation, and local validation is not evidence for unseen-context validity.

**DERIVED / EXISTING RESEARCH:** Which relation is approximated changes its noise placement and physical-function class. Forward and inverse RBF coefficient counts being equal does not make their approximation capacity identical: the leading inverse ACTION relation K(q)=M(q) is affine in cos(q2), while the forward response M(q)^-1 is rational. A learned inverse relation remains a model; the common bounded solver uses its implied forward relation. This is not a model-free-versus-model-necessity result.

**LIMITATION:** All episodes begin at supplied exact preparation pose/rest, have calibrated ideal torque, known duration/noise/angle topology, and measure joint-angle displacement after one short pulse. Pulse endpoints are moving. This is not continuous reaching/stopping, unknown-state inference, model-language discovery or general articulated intelligence. End this calibration branch after the compact test and move to continuous coupled motion or consequential missing-history revision.

## Why this changes the research question

The scalar program supplied a correct four-parameter function family and concentrated increasingly on controller optimization. V13 bounded that branch: a better optimizer finds lower average cost in the same policy family, but precise station keeping collapses. Adding another scalar optimizer would not resolve the deeper dependency on supplied vocabulary and context.

Start from an intervention rather than a preferred architecture: does the same command have the same consequence at different body configurations? A coupled body gives a clear counterexample. Learn a reusable relation from positions/actions, then use it to choose commands for goals never issued during acquisition. Forward consequences, inverse action relations and episodic/direct alternatives are logically possible; states/world models are not assumed universal foundations. Here a compact conventional comparison is enough to decide whether a new mechanism is immediately needed. The evidence favors conventional sufficiency inside the sampled regime and exposes unsupported transfer outside it.

## F53. Continuous coupled rest-pulse mechanics [DERIVED / EXISTING RESEARCH]

Consider a horizontal two-link point-mass body with unit masses/lengths, direct bounded generalized torques, no gravity/friction. From Cartesian kinetic energy, Lagrange's equation gives

~~~text
M(q2)=[[3+2cos(q2),1+cos(q2)],[1+cos(q2),1]]
M(q2)*v_dot + h(q2,v)=u, q_dot=v
h=sin(q2)*[-(2*v1*v2+v2^2), v1^2].
~~~

M is SPD: detM=2-cos²(q2)>=1, and its global minimum eigenvalue is3-2sqrt(2)=.1715728753. For fixed torque from rest, time-reversal symmetry and smoothness give

~~~text
q(T)-q0 = .5*T^2*M(q0)^-1*u + O(T^4).
~~~

At q2=0 the leading torque map is [[1,-2],[-2,5]], while at q2=pi it is identity. Thus joint2 torque has a nonzero leading joint1 effect at the first pose and zero leading coefficient at the second. The finite-time effect at pi is NOT identically zero: evolving q2 restores coupling at higher order. At generic q2, a frozen linear response has a quartic absolute remainder. At0/pi its quartic term vanishes, hiding that generic rate. Adding damping or actuator lag would change the expansion and cannot be silently imported.

An early design used semi-implicit Euler's T² coefficient; BEFORE execution the instrument was changed to continuous DOP853 and the .5T² coefficient, preserving the draft/correction. This avoids mistaking discretization for physical response. Classical mechanics is a tool, not a novelty claim or a supplied formula for the ordinary learner.

## E34. Independently checked coupled mechanics instrument [TESTED, NO LEARNER]

`coupled_context_v14_pilot.py`/JSON retains153 rest pulses over17 q2 poses/three command directions/three durations,24 non-rest energy cases and2 coupling examples:179 distinct episodes,21.025 simulated seconds. Twenty-four tighter numerical reintegrations are solver checks, not extra independent data; total203 solver calls. Joint positions are measured at pulse BEGINNING/END in time, not Cartesian end-effector positions. Pilot sensor SD1e-5.

Generic duration-halving error ratios are15.936..16.064 then15.984..16.016, approaching16 as quartic remainder predicts. Symmetry cases are higher order or roundoff limited. Independent theory computes fourth time derivative of q at0=(1.75,-3) at q2=pi/2,u=(0,1). Energy change equals constant torque work, and cyclic momentum p1 obeys p1_dot=u1. The artifact auditor reconstructs all cases using separately derived Hamiltonian(q,p) dynamics and Radau; max state discrepancy2.62e-12, energy-work1.66e-12, cyclic momentum8.27e-13; measurement RNG exact. An independent short-pulse implementation also checks the special higher-order effect. Sources/results/reports remain.

**Measurement correction before learning:** If the same noisy initial encoder reading supplies context and the displacement baseline, Cov(e0,eT-e0)=-R. Independent context reads remove that shared-noise path but still leave noisy-context EIV. The pilot contains that correlated record, but trains no learner. The subsequent comparison openly supplies exact PREPARATION pose to every method; it does not pretend to infer exact context from noisy sensors. Both joint coordinates are supplied; selecting q2 alone would reveal the relevant context dimension. Angle periodicity is separately supplied to the generic basis.

## F54. Fair inverse regression needs a measurement model [DERIVED / EXISTING RESEARCH]

Define measured normalized response d_obs=(oT-q0)/k, k=.5T², using supplied EXACT preparation pose. At T=.1 and encoder SD1e-4, response-noise covariance is S=.0004 I. An earlier two-reading comparison used (oT-o0)/k and S=.0008 I, adding avoidable initial measurement noise; its complete source/data/results are preserved below. Let phi(q) be fixed features. A forward relation d≈B(q)u has exact issued-command regressors and noisy response outputs. An inverse u≈K(q)d instead has noisy regressors: naive least squares generally shrinks an inverse even with unlimited samples. It is not a fair sole direct baseline.

With feature ordering(context,input), X=phi⊗d_obs. Independent zero-mean response noise gives

~~~text
E[X_obs'X_obs | noiseless data] = X_true'X_true + sum(phi*phi') tensor S
E[X_obs'u | noiseless data] = X_true'u.
~~~

Subtract the known covariance term for corrected moments. These are unbiased MOMENTS, not unbiased finite coefficient estimates. Inversion, regularization, finite data and selection retain bias. The source explicitly projects negative corrected-Gram eigenvalues to zero then adds ridge; it does not solve an unregularized indefinite moment equation. The field named training_normal_equation_residual refers to unregularized G and need not vanish under this regularization.

For any fixed inverse candidate K evaluated on fresh validation measurement noise,

~~~text
E[||K*d_obs-u||² - tr(K*S*K')] = ||K*d_true-u||².
~~~

This correction is valid even when finite-pulse nonlinearities make the linear inverse family approximate; it removes measurement risk, not approximation error. Corrected validation values can be negative, and choosing their minimum is optimistic. All candidate scores/eigenvalues/floors are kept; independent fresh tasks, not a lucky validation minimum, determine evidence. These are standard errors-in-variables identities, not new statistical theory.

## E35. Compact same-experience relation-learning comparison [TESTED]

The original `coupled_learning_v14_two_reading_pilot_design.md` was prescribed before learning outcomes. Current design and `coupled_learning_v14_information_correction.md` preserve the later exploratory information-use repair, with all tuning grids/tasks unchanged. Two acquisition conditions: q1 uniform[-pi,pi] in both; q2 broad[-pi,pi] or narrow[-.4,.4]. Eight independently seeded datasets per condition,512 training+128 separate validation pulses each. Commands uniform[-1,1]^2, independently randomized. Each pulse has T=.1, zero initial velocity, known EXACT preparation pose supplied to every learner, and independent noisy initial/final joint-angle measurements. Ordinary learners receive preparation context, commands and these measurements; true states only score outcomes. No mass/inertia/Coriolis/relevant-encoder formula is supplied. Canonical learners use the available exact preparation rather than a noisy baseline when constructing their response.

All methods use the identical complete dataset within replicate. Alternatives:

- FULL coupled constant forward matrix, a serious simple context-omission baseline rather than independent joints.
- Generic context forward B(q), using65 periodic Gaussian RBF features (8x8 center grid over BOTH encoders plusconstant).
- Same feature vocabulary for naive inverse K(q), a descriptive measurement-bias baseline.
- Moment-corrected inverse with the known response covariance and PSD/ridge regularization.

Width(.5,1,2,4) and ridge(1e-6,1e-4,.01,.1) are chosen by separate declared validation; constant matrix has the ridge grid only. Forward selection uses response prediction SSE; both inverse variants use noise-corrected inverse risk. All832 candidate fits/coefficients/eigenspectra/selection scores per comparison are saved (1664 across original and information-corrected pipelines). Selection itself remains finite/noisy; neither moment subtraction nor equal feature counts gives equal physical prior or universal optimality.

For ALL ordinary methods, commands minimize the SAME bounded displacement metric. Forward uses B(q); inverse uses pinv(K(q),rcond1e-8), retaining conditioning diagnostics. This eliminates a command-clipping versus goal-metric projection confound. It also means the inverse is a learned relation used in planning, not a pure model-free policy search.

Fresh tasks: q1[-2.7,-1.1,.4,2.2] crossed with q2[-pi,-2.1,-1.2,-.4,0,.4,1.2,2.1] and five COMMON desired normalized displacement vectors(.1,0),(0,.1),(.1,-.1),(-.1,.1),(.05,.1), total160 context-goal pairs per fit/method. An initial proposed +pi duplicate was replaced by0 BEFORE execution, making the periodic poses unique and including narrow-region interpolation. Desired actual displacement=k*z. Test goals are not generated from hidden pose-specific random torques, so they do not encode the action relation being tested.

A privileged continuous endpoint optimizer constructs a bounded command witness for EVERY task, max residual3.24e-13 in normalized displacement,1456 recorded solver rollouts in the original comparison; canonical correction reuses the same witnesses with no new oracle search. It receives true mechanics and is excluded from ordinary learning comparisons. Leading-order required torque<=.5 supplies margin; exact numerical witnesses establish feasibility for these finite tasks, not a global reachability theorem.

Primary score is ||actual_displacement/k-z||², averaged over160 tasks then summarized across eight independent acquired-data seeds. Truth is evaluator-only. Task dynamics are deterministic after command issuance; noisy test readings are retained but do not enter one-pulse decisions or primary score. MC SE concerns acquired-data variation, not160 independent fits. Additional128 fresh broad-context/random-command pulses per fit test predictive transfer without updating learners.

{{TABLE}}

In broad context, naive-minus-corrected inverse paired task MSE is +4.458017e-6 +/-1.081753e-6 SE. Corrected inverse therefore improves this finite declared pipeline, while the forward-vs-inverse contrast also changes physical-function approximation and decision conditioning. In narrow context, the same difference is -.000293217 +/- .000602516: the corrected mean is slightly worse, with large paired uncertainty. The stronger negative difference in the original two-reading comparison does not carry over as a strong claim about the repaired estimator. It does not repair missing context coverage. No philosophical conclusion about model necessity follows.

**Secondary support diagnostic:** Narrow acquisition still fits decisions inside q2[-.4,.4]: context-forward mean task MSE2.44037e-5, corrected inverse9.95402e-6 over60 familiar-region tasks. On100 outside-region tasks these become.0876804 and.0236446. The constant model is9.84515e-5 inside/.0311221 outside. This isolates support-related transfer failure descriptively; it does not prove no method could extrapolate with appropriate physical assumptions. Broad acquisition uses the same640 pulses per seed, not more interactions.

**Symmetry limitation:** Physics is invariant to q1 global rotation, but learned context relations have nonzero command variation when the SAME q2/goal is tested at different q1. Lower task error does not prove exact symmetry discovery. Both q1 diagnostics/pooling conventions and all commands are preserved independently. No supplied q2-only feature selection was passed to the ordinary learner. Unsupported extrapolation can amplify this nuisance dependence and nearly singular inverse predictions.

## Pre-publication information-use correction [CORRECTED / TESTED]

The first complete comparison used the unbiased two-reading displacement (oT-o0)/k even though q0 was supplied exactly. Independent audit identified that o0-q0 is observable sensor noise: using (oT-q0)/k removes that independent error and halves S from.0008 to.0004 I. The old statistics remain valid for their stated estimators, but that noise was not irreducible and those estimators did not use available information efficiently.

Before publication, correct ALL methods without changing features, width/ridge grids, conditions, validation logic, targets or bounds. Reuse EXACT original raw acquisition/validation/predictive data and original endpoint feasibility witnesses. Original source/default output metadata, full data/results/timings/design/theory/audits remain as `coupled_learning_v14_two_reading_pilot*` and corresponding audit names. The canonical source records exact parent hashes. All160 non-response arrays are bitwise identical;32 response arrays change only by the declared statistic. No new acquisition/predictive data or oracle search is added. New commands require12800 additional task episodes.

This repair follows an information audit after original outcomes were read; call it exploratory, not a fresh blind benchmark. Heldout tasks/priors are unchanged and no task outcome selects settings. Original broad task MSEs for constant/forward/naive/corrected are.00898141/7.02875e-5/1.79176e-5/9.41635e-6; original narrow values.0196274/.0455312/.0231309/.0254856. Lower effective measurement noise improves all broad context-aware means in the canonical run, but does not uniformly improve out-of-support selected pipelines. Selection/regularization still matter.

A post-hoc analytic reference is zero torque: rest remains exactly at rest, so average task score equals mean||z||²=.0145 with no extra simulation. Narrow inverse means(.0144884 naive/.0147816 corrected) are close to that uninformed reference at their large SEs; their aggregate broad-grid transfer is not reliable learned goal achievement. This reference is not a tuned new method or an equivalence test. Broad context-aware scores are orders of magnitude smaller. No more calibration variants are added.

## Audit, history corrections, budgets and reproduction

All source/design/pilot/actual results/independent reports and failed or rejected candidates remain. Original two-reading and canonical repaired JSON/NPZ re-execution are byte-identical within their respective pipelines. Original pre-audit metadata is retained; separate completed audit records subsequent verification. No numerical/task case is silently dropped. All ordinary bounded action solver statuses and selected condition numbers are audited; successful numerical optimization does not make an extrapolated relation correct.

Acquisition:16 datasets x640 pulses=10240 episodes/resets,1024 simulated seconds. Predictive tests2048 episodes/204.8 seconds. Each original/corrected comparison has12800 task episodes/1280 seconds (five methods including privileged endpoint oracle), total25600 episodes/2560 seconds during development. Oracle feasibility optimization additionally uses1456 deterministic plant-simulator rollouts/145.6 seconds once, not ordinary learner training. There are49152 joint-coordinate scalar sensor readings in the reused acquired/predictive episodes and102400 total in both task comparisons; the preparation mechanism supplies exact context/rest beyond those noisy readings. Pilot179 episodes/21.025 seconds and solver-refinement computation are additional development, not hidden acquisition. No hardware experiments are claimed. Wall timings are retained separately and do not constitute a runtime-efficiency theorem.

Independent theory derives Lagrangian mechanics, continuous short-pulse order, sensor-noise coupling, EIV/validation identities, source tensor ordering and physical-function-class limitations. The artifact audit independently reconstructs physics, RNG/data, all fits/validation/selection, commands, predictive and decision scores, budgets and clustered summaries, including extrapolation failures. It uses a separate implementation and qualifies numerical conditioning rather than rerunning source and calling that independent verification. See `audit_coupled_context_v14_pilot`, `audit_coupled_shortpulse_v14`, `audit_coupled_learning_v14`, q1-variation audit and `coupled_learning_v14_theory_independent` reports/results. Audit scope and unresolved concerns must be read before extending any claim.

~~~bash
.venv/bin/python research/coupled_context_v14_pilot.py > /tmp/coupled_pilot_v14.json
cmp research/coupled_context_v14_pilot_results.json /tmp/coupled_pilot_v14.json
.venv/bin/python research/coupled_learning_v14.py --output-prefix /tmp/coupled_v14
cmp research/coupled_learning_v14_results.json /tmp/coupled_v14_results.json
cmp research/coupled_learning_v14_data.npz /tmp/coupled_v14_data.npz
.venv/bin/python research/audit_coupled_learning_v14.py
~~~

**EXISTING RESEARCH:** [Tedrake's multibody derivation](https://underactuated.mit.edu/multibody.html) supplies classical mechanics context. [Nguyen-Tuong/Peters2011](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Nguyen_CP_2011.pdf) explicitly discusses forward/inverse/mixed robot models and ill-defined inverses. [Jordan/Rumelhart1992](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1603_1) addresses learning when unknown dynamics intervene between actions and outcomes. These precedents exclude novelty claims. [Romeres et al.2018](https://arxiv.org/abs/1809.05074) tests learned position-history features for inverse dynamics on real iCub-arm data, a relevant stronger method for subsequent continuous motion. Our calibration is not a reproduction or performance competitor. The literature note distinguishes accessed full papers from publisher abstracts.

## Research-direction reassessment and exact continuation

This is a modest construction beyond the supplied scalar/tanh family: a generic conventional learner acquires context-dependent coupled behavior and chooses new actions from noisy experience. The same data can support forward or inverse relations, with different function/noise/conditioning behavior. Statistical corrections and local validation have explicit boundaries. Simple established methods suffice for the supported pulse task; do not invent an architecture to rescue novelty.

The scientific value is conditional and limited: exact rest/pose preparation, ideal torque, fixed body, joint-angle semantics/periodicity, torque-affine finite basis, known sensor noise and short isolated episodes are all supplied. No unknown component language, cross-body transfer, continuous stability or articulated stopping is demonstrated. A successful calibration is not the ultimate scientific objective. Further calibration/cosine-fitting releases would drift.

**Immediately continue:** Remove rest preparation and investigate continuous coupled motion or how a learner detects/revises a summary that omits relevant history. Derive a decisive aliasing/transfer example, ensure actual task feasibility, and compare serious history-based/derivative-free conventional methods rather than giving only one method true velocities or accelerations. Learn from actual available positions/actions; distinguish a learned conditional predictor from a supplied physical state. Preserve first-principles alternatives and strong baselines. Complete and audit meaningful results, append complete v14 in the next ledger, commit/push SSH/verify and continue. The general-learning Goal remains ACTIVE.

---

# Historical archive: Complete original v13 ledger, including all preceding versions

'''
rows=['| Acquisition | Relation | Task MSE +/- clustered SE | Broad-context prediction MSE +/- SE | Saturations /1280 |','| --- | --- | ---: | ---: | ---: |']
for cell in r['summary']:
    for label,s in cell['models'].items():
        a=s['task_mse'];p=s.get('predictive_mse');pred='privileged feasibility witness' if p is None else f"{p['mean']:.6g} +/- {p['mc_se']:.6g}"
        rows.append(f"| {'broad' if cell['condition']==0 else 'narrow'} | {label} | {a['mean']:.9g} +/- {a['mc_se']:.6g} | {pred} | {s['saturations']} |")
doc=doc.replace('{{TABLE}}','\n'.join(rows));assert '{{' not in doc
out=root/'first_principles_robot_learning_ledger_v14.md'
if out.exists():raise SystemExit('Refusing to overwrite released v14')
if not (root/'audit_coupled_learning_v14_results.json').exists():raise SystemExit('Independent completed audit required')
audit=json.loads((root/'audit_coupled_learning_v14_results.json').read_text())
assert audit['artifact_sha256']['coupled_learning_v14_results.json']==hashlib.sha256((root/'coupled_learning_v14_results.json').read_bytes()).hexdigest(), 'Current-result audit hash required'
out.write_text(doc+prior);assert out.read_text().endswith(prior);print(out)
