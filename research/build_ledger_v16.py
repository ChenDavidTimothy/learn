#!/usr/bin/env python3
"""Assemble an audited acquired-Cartesian-response milestone; retain all v15 bytes."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=ROOT/'first_principles_robot_learning_ledger_v16.md'
    assert not target.exists(),'Released ledgers are immutable'
    aggregate=json.loads((ROOT/'cartesian_feedback_v16_aggregate_results.json').read_bytes())
    for rep in range(100,108):
        a=json.loads((ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_results.json').read_bytes())
        assert a['status']=='INDEPENDENT COMPLETE CORPUS REPLAY PASSED; NUMERICAL INTEGRALS AUDITED SEPARATELY'
        assert sha(ROOT/'audit_cartesian_feedback_v16_study.py')==a['audit_source_sha256']
        for name,value in a['artifact_sha256'].items():assert sha(ROOT/name)==value,name
    # Final aggregate and numerical audits are additional publication gates.
    audit=json.loads((ROOT/'audit_cartesian_feedback_v16_aggregate_results.json').read_bytes())
    assert 'PASS' in audit['status'] or 'VERIFIED' in audit['status'],audit['status']
    numerical=json.loads((ROOT/'audit_cartesian_feedback_v16_study_numerical_results.json').read_bytes())
    assert numerical['status'].startswith('PASS'),numerical['status']
    assert sha(ROOT/'audit_cartesian_feedback_v16_aggregate.py')==audit['audit_source_sha256']
    assert sha(ROOT/'aggregate_cartesian_feedback_v16_numerical_audit.py')==numerical['aggregate_audit_source_sha256']
    for record in (audit,numerical):
        for name,value in record['artifact_sha256'].items():assert sha(ROOT/name)==value,name
    assert numerical['scope']['corpora']==8 and numerical['scope']['initial_histories']==64
    assert numerical['scope']['initial_top_three_checked_everywhere']
    assert numerical['scope']['later_histories']==128 and numerical['scope']['integrals']==1440
    assert not numerical['threshold_failures'] and not numerical['checked_ranking_changes']
    s=aggregate['summary'];table=['| Same-data prior/action rule | Mean phase cost +/- corpus SE | First-goal cost | Later-goal cost | Precise endpoints /384 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,row in s.items():
        table.append(f"| {name} | {row['cost']['mean']:.9g} +/- {row['cost']['se_across_eight_corpora']:.3g} | {row['first_goal_cost']['mean']:.8g} | {row['later_goal_cost']['mean']:.8g} | {row['precision_count']} |")
    selections=[c['selected_mixture_components'] for c in aggregate['corpora']]
    doc=r'''# First-Principles Robot Learning: Living Research Ledger v16

**Date:** 2026-10-20, Europe/Berlin. **Status:** acquired Cartesian response variation, purposeful evidence, and persistent local goal control; independently audited conventional constructions and explicit limits. V15 was SSH-published and remotely verified at `e64e48c72bb9bcde3a85257d9d54ee843c6394ac`; audited post-v15 payload evidence/question handoff at `2023af57f978ff0baab22e7a144a44f7cce6c095` preceded this investigation. Complete v15 and every earlier ledger follow verbatim. The broad general-learning Goal remains ACTIVE and unresolved.

## Principal finding and boundary

**TESTED:** Unlabeled action/Cartesian experience can acquire alternative local responses and useful covariance structure, then use a small amount of new evidence for bounded physical decisions while retaining knowledge and motion through six successive goals. Ordinary learners receive no joint encoders, evolving true velocity, mass, branch label or true response matrix; initial Cartesian position/rest are supplied. General Gaussian, matched Gaussian, finite mixture and isotropic diagnostics share the acquired records, camera/action clock, loss and correlated continuation solver. Their full failed cases remain.

**PROVEN UNDER ASSUMPTIONS:** More stationary Cartesian history cannot resolve the two present hidden preparations. Under the declared local coefficient model, single-action population marginals also need not identify coherent cross-action consequences. Fixed open-loop quadratic decisions need only first/two moments; anticipating evidence-conditioned decisions can depend on more of the prior law. These are separate statements. The general covariance counterexample does NOT prove grouped experience is necessary for this arm's restricted rank-one variation.

**LIMITED / EXISTING RESEARCH:** This is conventional hierarchical regression, joint Gaussian/mixture conditioning and a small finite-candidate dual-control calculation, not a new universal architecture. Exact initial Cartesian preparation/rest, known noise/timing, persistent grouping and a constant-response vocabulary are supplied. The articulated ODE generates actual nonlinear motion, but millimeter goals remain near one preparation. Two-stage feedback is repeated without goal resets; it is not arbitrary per-pulse feedback or stationkeeping. No modern robot-learning benchmark superiority, real-time feasibility or general intelligence is established.

## E38. The payload premise failed before adding adaptation [TESTED NEGATIVE]

After v15, `payload_revision_v16_pilot.py` attaches distal mass1->2 at the first6sec goal switch, preserving q/v through an external matched-velocity load. The change is not announced to the frozen controller. The single parent v15rep100richH4 and entire nine-setting causal tanhPD bank are retained; no online learner or active probe is added. Four fixed contexts and paired fresh role-tagged sensor paths give80episodes/15,040intervals. This instrument still uses noisy JOINT positions, not Cartesian sensing.

Frozen MPC passes the same7/8postevent precision windows with and without the load. Postevent positionMSE changes .0107440728->.0124800700 and late velocityRMS .0079659600->.0068941482. This does not prove noninferiority or universal robustness: it is one learned fit/four contexts. It does reject the premise that this specification already requires adaptation or probing. No disturbance was increased until a desired method won.

Independent variable-mass Lagrangian/Hamiltonian audits check torque/work, external momentum/energy exchange, unit-parent recovery, all80episodes,1,440MPCgradients and12,960PDactions. Maximum replay discrepancy8.77e-11; common preevent states and first loaded actions match exactly. Sources, actual results, audit and rejected premise remain. Goal/change clocks coincide and cannot later be described as an unknown-time detector experiment.

## F57. An intervention can expose what more passive history cannot [PROVEN UNDER ASSUMPTIONS]

For two unit links p(q)=(cos(q1)+cos(q1+q2),sin(q1)+sin(q1+q2)). Preparations qA=(-phi/2,phi),qB=-qA,0<phi<pi have the same p0=(2cos(phi/2),0). At zero motion/torque their Cartesian histories are exactly identical for any duration, including with the same independent noise law. Offline population experience cannot identify the CURRENT preparation from that stationary record alone.

The inertias agree because M depends on cos(q2). With B=J*M^-1, at phi=pi/2,

~~~text
BA=[[1,-3],[1,1]]/(2sqrt(2)),
BB=[[-1,3],[1,1]]/(2sqrt(2)).
p(T)-p0=.5*T²*B*u+O(T4), starting at rest.
~~~

Each B is invertible, while its equal-preparation mean has a zero leading radial row. Averaging can erase an attainable direction. The EXACT finite-time mean need not have a zero radial response: curvature contributes higher-order effects. Nor does this prove every common open-loop sequence fails. Tangential goals admit a common leading action u proportional to(3,1); resolving the hidden distinction is useful only when it changes relevant available decisions enough to repay the experience. See `cartesian_branch_v16_theory_independent.md` for derivation and counterexamples.

## F58. Marginals are not always a reusable relationship [PROVEN UNDER ASSUMPTIONS / DESIGN CORRECTED]

Let b=vec_row(B), H(u)=I2 tensor u^T and A=[[0,1],[-1,0]]. For0<epsilon<1, the two positive-definite Gaussian covariances

~~~text
Gamma_plus/minus = I4 +/- epsilon*[[0,A],[A^T,0]]
~~~

give H(u)*Gamma*H(u)^T=||u||²I2 for EVERYu, hence identical single-action marginals, even with the same sensor noise. Their cross-action laws differ: Cov(B e1,B e2)=+/-epsilon*A. An observation under one action therefore has different implications for another. No amount of independent one-pulse population data identifies this general missing covariance direction.

Both independent auditors raised this BEFORE Cartesian acquisition. The design was repaired to group two consecutive actions under the same hidden context, without an intermediate reset. From supplied p0/rest, normalized endpoints satisfy

~~~text
(p1-p0)/T² = B*(.5u0),
(p2-p0)/T² = B*(1.5u0+.5u1).
~~~

Their2x2 effective-input determinant is .25det(u0,u1), nonzero almost surely for independent continuous commands. The resulting full group exposes all coefficient coordinates under the local model. Nearly collinear groups remain statistically weak; invertibility is not finite-data precision. The code fits whole group likelihoods/moments rather than treating inverted noisy coefficients as truth. This general theorem does not establish grouping necessity in the special arm, whose leading variation can already be found from suitable single-action second moments under extra restrictions.

## F59. Useful conditioning must retain motion/response dependence [DERIVED / EXISTING RESEARCH]

Use normalized z=[s,w,b],s=(p-p0)/T²,w=v_cartesian/T, with one fixed p0. Under the supplied constant-B model, known issued u gives s'=s+w+.5Bu,w'=w+Bu,b'=b. This is linear in the augmented quantity. Gaussian conditioning is exact WITHIN that model; mixtures condition full components and their weights. Real finite nonlinear flow can invalidate the model. No process noise/drift repair is inserted in this study.

After the first pulse, with remaining controlsU=(u1,u2), terminal s=s_now+2w+B[1.5I,.5I]U and w_terminal=w+B[I,I]U. Expected quadratic loss depends on G=E[B^TB], h_d=E[B^T(s+2w-g)] and h_w=E[B^Tw]. The cross-covariance terms are necessary. The state-uncertainty constant E||s+2w-g||²+.25E||w||² is irrelevant to the INNER optimizer but necessary when comparing FIRST actions. A noisy camera reading cannot also be inserted as an independent exact state.

Positive effort weight makes the remaining boxQP strictly convex. The solver checks KKT and enumerates all81faces when the unconstrained four-coordinate solution is infeasible. Cautious three-pulse plans use the analogous six-coordinate problem. Mixture and FULL-moment matched Gaussian have identical fixed open-loop quadratic decisions; shape matters through conditioning or its anticipation. The outer first-action menu is a heuristic, not a globally optimal dual controller. Independent augmented sigma-point/Torch/direct raw-second-moment checks establish the formulas and active-bound gradients; all actual filters/continuations are independently replayed.

## E39. Acquired priors and preserved numerical failures [TESTED / CORRECTED]

The exploratory corpus contains256fit/64validation/64test independent two-action groups atT=.1, cameraSD=.0002, |u_j|<=.5. A group chooses one hidden rest preparation, independent uniform commands and noisy Cartesian endpoints. No q/v/B/branch labels enter ordinary fitting. The mean/full covariance and K1/2/3mixtures are learned from raw groups. Seventeen generic EM starts, three moment/PSD Gaussian choices and two full marginal-likelihood refinements are retained. Validation, not control outcomes, selects candidates. A matched Gaussian isolates prior shape from differing fitted moments. Gaussian factor initialization uses eigenfloor1e-8 to avoid trapping a covariance factor in a deficient subspace. Bounded L-BFGS termination does not establish global maximum likelihood.

The initial moment Gaussian was weaker than a conventional likelihood fit: its directed mean task cost .0741395 falls to .0360809 after likelihood refinement on the SAME acquired records, versus mixture .000947949. All9methods/1,728short-task evaluations and failures remain. The mean-only ablation has physically infeasible first probes in128cases and is not the primary competitor. Heldout mean response errors near.083 do not distinguish coherent prior quality; grouped/conditional likelihood does.

The original GH9x9integration changed some Gaussian action choices under refinement and was rejected. Adaptive Genz-Malik then falsely reported small errors on four of900candidate values: two mixture discrepancies about+.00194528/-.00182176 and two Gaussian discrepancies-.000102805. Independent tighter tensor Gauss-Kronrod21 and scrambled Sobol corroborate the failures. Analytic Gaussian tail bounds remain valid but do not bound cubature error. All18directed choices and all18random-radius choices happened to remain unchanged under complete rescoring.

Every pilot/refinement/adaptive source, design, raw dataset, result and failed audit is preserved separately. Corrected immutable `_gk21.py` batch agrees with all900independently reconstructed scores within9.97e-12; complete controls/priors/fits are exactly unchanged from the adaptive parent. It acquires no fresh evidence. Three pulses with one intermediate read are NOT sustained feedback or a separate learning milestone.

## E40. Persistent Cartesian construction pilot [TESTED, ENGINEERING]

`cartesian_feedback_v16.py` reuses that one acquired corpus and propagates full8D beliefs through six current-goal-only phases without motion/belief resets. Each phase has three pulses: read after first, commit lasttwo, read after third for the NEXTgoal. Two goal sequences, two hidden preparations, two paired fresh noise repeats and eight methods give64episodes/384endpoints. Initial p0/rest are supplied exactly only once; ordinary input thereafter is the actual noisy camera/action history.

Mixture components receive declared plug-in coefficient curvature covariance I_output tensor(R*weightedGram^-1). All17existing fits are reselected using their actual predictive law on validation groups. This is conditional curvature, not full Bayesian assignment/model-error uncertainty. Mixture/generalGaussian/matchedGaussian each use directed/cautious actions, with mixture/Gaussian random-direction references. The extra pilot learned-variation direction menu is preserved but removed from the fresh comparison below.

Pilot mean phasecost/precise endpoints of48: mixture directed .000376472/48, ordinary task .417843872/27, random .084058946/44; generalGaussian directed .034535144/41, task .566171917/19, random .074462837/41; matchedGaussian directed .033992108/42 and task .475099162/21. These are one learned corpus/four independent noise paths, not64independent bodies. All1152physical intervals,384full filter/continuation phases,17prior reconstructions,24nominal nonlinear feasibility trajectories and counters replay independently; maximum Hamiltonian state error8.5e-15 and witness normalized residual difference1.2e-12.

Separate actual-integral checks cover ALL132initial candidates plus360later topthree scores. No checked action ordering reverses, but Genz-Malik false error estimates recur, up to.000102805. Tight split-domain checks confirm them. Retain numerical failures rather than replacing their flags with a blanket audit pass. Production therefore changes numerical integration and requires fresh acquired corpora.

## E41. Fresh acquired-response reuse over uninterrupted goals [TESTED]

Frozen `cartesian_feedback_v16_study_design.md` precedes all fresh replicates100..107. Each independently acquires256fit/64validation/64test groups, refits the same17mixture/fiveGaussian candidates and deploys all10methods on the same physical/sensing/loss contract. No redundant short-task deployment is run on these corpora. Selected mixture component counts are {{SELECTIONS}}; this is validation behavior, not discovery of a physical number of modes. Eighteen of136EMcandidate fits reach the150iteration limit; rep107's DEPLOYED K3candidate is among them. All other deployed K2candidates converge. These unfinished fits are retained and independently reproduced, not discarded or retuned. All16GaussianMLruns report ftol-success, not global/strict-gradient optimality.

The trace-matched isotropic diagnostic retains the general Gaussian's learned mean and total coefficient variance, replacing Gamma by trace(Gamma)/4*I4. It keeps the same full joint filter, memory, observations and continuation solver. It changes anisotropy, marginal variances and covariance spectrum as well as correlations; it is NOT a pure coherence ablation or an optimized from-scratch baseline. The serious primary Gaussian retains its full learned covariance and conventional updates.

All priors share16uniform directions at radii.05/.1, zero and each prior's own cautious first action:34candidates, common geometric grid plus model-specific cautious action. There is no extra learned-eigenvector action for the structured prior. Random direction is uniform over the same16, with radius chosen by model direction-average. Every method has equal sensor opportunities/clock/bounds, not necessarily equal actual energy. Effort is scored and separately retained.

First-action expectations use gk21 on four fixed standard-normal quadrants, totalatol1e-8/rtol1e-7, then reintegrate the lowestthree at1e-10/1e-9 before selection. Coarse/strict values/errors, points, subdivisions and analytic discarded-tail envelopes remain. Numerical errors are estimated, not certified; outer candidate search is finite. Independent differently partitioned/rule tests check actual fresh integrals, with exact scope recorded in the separate numerical report. These do not prove global optimality or uniform accuracy on every possible belief.

Same two sequences of offsets*L, L=.001meter:

~~~text
A:(1,0),(1,1),(0,1),(-1,1),(-1,0),(0,0)
B:(0,1),(1,1),(1,0),(0,0),(-1,0),(-1,1)
~~~

Each adjacent goal moves one coordinate step. Each phase costs ||p3-p0-goal||²/L²+.25||v3/(L/T)||²+.01sum_all3||u||². Precision requires percoordinate .0003m AND.003m/sec at the ENDPOINT, not a sustained final window. Only the current goal is disclosed. Exact nonlinear nominal zero-first-action witnesses establish their own trajectories' feasibility, not recovery from all states reached by another method. All actions, beliefs, states, camera readings and failed goals remain.

Each table row has384clustered endpoints, from eight separately acquired corpora, two fixed preparations, two sequences and two paired sensor repeats. Means are formed per corpus BEFORE descriptive SE across8corpora. Only32independent paired noise paths underlie the640method episodes. These are not a sample of640bodies or general robotic tasks. First/later goals are separated to expose initial learning cost and subsequent adaptation; full sequence/phase profiles and paired differences are saved.

{{TABLE}}

The data support the comparisons in this declared contract. Mixture directed achieves63/64first and320/320later endpoint checks; the general Gaussian52/64first and311/320later, with later cost dropping to.01097051 from first.09381124. The isotropic diagnostic28/64first and222/320later. The full Gaussian therefore already acquires and reuses useful structure, and improves markedly after initial evidence. These results do not establish universal superiority of mixtures or explicit modes: ordinary task actions can themselves gather useful evidence, and distributional shape is only one possible way to preserve useful conditional consequences.

One real rare case dominates the mixture's marginal cost SE: rep105/sequenceA/repeat0/hiddenpreparation1/firstgoal. The first noisy read favors the wrong fitted component with posteriorweight.8760; terminal radial error is-.00175235m, phasecost3.07187341, while it ends nearly stopped. The terminal camera read reverses the component weighting to.999999617on the other component, and all five subsequent phases are precise. This failed first decision is retained; no probe amplitude, model or noise was tuned to remove it. The paired generalGaussian-minus-mixture meancost difference is.01634871+/- .00195621SE across8corpora, despite the mixture's marginal .00801190SE, because both methods share the difficult sensor paths. The isotropic-minus-generalGaussian difference is.15257009+/- .00932290SE. These are descriptive fixed-study comparisons, not universal risk or population-normality guarantees.

Shape and interaction choice can matter with finite noise/budget, but improved precision is not a universal need for discrete physical identification. Retain the covariance-isotropy result with its spectrum/marginal-variance caveat. The full per-corpus and per-sequence profiles, costs, effort and precision are in `cartesian_feedback_v16_aggregate_results.json`; no failure is removed for a cleaner summary.

### Independent verification and resources

Every fresh acquisition group,17EM/fiveGaussian fit candidates per corpus, heldout predictive likelihood, actual filter/continuation/action/score and physical trajectory is independently reconstructed. The auditor derives continuation expectations through full raw8D second moments, not the implementation's expanded formula; Hamiltonian replay and Radau spots/witnesses provide independent physics. Maximum fresh physical state discrepancy1.18e-14, continuation-command discrepancy7.72e-14; independent20-coordinate GaussianMLfinite-difference gradients differ at most6.25e-9. Aggregate means, paired SEs and counts reconstruct to roundoff.

Independent actual-integral checks use gk15 on NINEboxes with edges[-7,-2,2,7], tighter1e-10/1e-9, independently reconstructed joint conditioning/raw8Dmoment QPs and no canonical controller imports. All1440checked integrals agree within1.76e-9:1184initial/256later, all64initial histories' topthree,128later selected/runner histories at prespecified phases2/5 and fixed branch/repeat strata, plus32complete34candidate initial menus on100..103. No checked ordering/radius choice changes. The initial plan to check2176initial integrals was curtailed after this balanced policy-relevant coverage: unchosen K3hypotheticals were expensive and no uniform certificate was claimed. All completed checks remain. Later random-radius averages and complete104..107initial menus are NOT independently reintegrated; rep107's tinyK3selected/nearby forecasts ARE checked. Partial107fixtures are revalidated against final complete tuples before final hash binding. This is finite numerical confidence, not analytic enclosure or global stochastic-control certification. Exact hashes/scopes remain in the independent numerical report. The ledger builder requires completed physical/statistical/numerical audits before writing this file.

Fresh learning acquisition totals3072groups/6144intervals/614.4simsec/18432scalar camera reads, including noisy initial reads although exact p0 is supplied. Fresh control totals640episodes/3840clustered endpoints/11520intervals/1152simsec/15360scalar reads,32paired noise paths/768distinct scalar noise draws. There are1984computed designs,67456initial candidate integrations,5952tighter candidate reintegrations and1,419,473,144hypothetical integrand rows. Additional privileged nominal witnesses require11088plant-flow calls; independent numerical checking adds71,240,234hypothetical rows. These are separate from ordinary experience; exact preparation remains an unpriced supplied resource.

An inherited reporting bug was caught DURING the firstfour runs: their resource dictionary describes control-phase parent reuse/new0 even though the invocation itself generated a separately saved acquired corpus. Freeze initial source and preserve raw results. Laterfour source changes only origin/cost/timing labels, not numerical behavior. Aggregate combined costs are reconstructed from each actual acquisition array. `wall_seconds` originally excludes acquisition; corrected source also names control and total invocation timing. No raw result is silently rewritten. Both executable source versions and the protocol erratum are retained.

The integrator can evaluate hundreds of millions of hypothetical rows per corpus. Planning computation is not charged as physical delay in this simulation. No real-time implementation, latency-aware physical controller or computational superiority is established. More sophisticated approximate planners might reduce that cost, but refining this optimizer is not the next scientific milestone.

## Existing research and reproducibility

Full primary method/experiment sections were inspected in the independent literature review, not just abstracts: [Klenske/Hennig2016](https://jmlr.org/papers/volume17/15-162/15-162.pdf), [Nagabandi etal.2019](https://arxiv.org/html/1803.11347), [Chaumette/Hutchinson2006](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf), [Przystupa etal.2021](https://arxiv.org/pdf/2106.06083), [PETS2018](https://arxiv.org/pdf/1805.12114). The first establishes dual-control/turn-off precedent; the others challenge adaptation, visual-interface and coherent uncertainty choices. None of their full implementations was reproduced. Torque-to-Cartesian acceleration is not a velocity-command visual Jacobian. No claim against complete modern competitors follows from this instrument. Root additionally inspected the joint-state/parameter definitions and nonparametric formulation of the dual-control paper; our exact within-model filter relies on its unusually simple linear-in-augmented-quantity transition.

~~~bash
cd /home/dtdav/dev/learn
# Preserve published artifacts: choose a new output path for reproduction.
.venv/bin/python research/cartesian_feedback_v16_study_initial.py --replicate 100 --output-prefix /tmp/cartesian_v16_rep100
.venv/bin/python research/cartesian_feedback_v16_study.py --replicate 104 --output-prefix /tmp/cartesian_v16_rep104
.venv/bin/python research/analyze_cartesian_feedback_v16.py
.venv/bin/python research/audit_cartesian_feedback_v16_study.py --replicates 100 101 102 103 104 105 106 107
.venv/bin/python research/audit_cartesian_feedback_v16_aggregate.py
~~~

Use existing owner PC/project.venv; no new packages/environments/services were installed. The immutable batch GH/refinement/Genz/gk21 artifacts, persistent pilot, fresh acquired corpora/results, independent executable audits and complete past negative results are durable research memory. Publication uses the existing SSH origin without force-push; remote commit verification is recorded in the continuation handoff after committing.

## Direction reassessment and best next question [OPEN]

This cycle resolves a real uncertainty: a passive missing distinction can require useful intervention, and acquired population variation can let that new evidence support subsequent decisions. It does NOT discover general laws from unrestricted experience. We supplied the context grouping, initial rest and local response vocabulary. Correct ordinary Bayes solves inference/decision once the joint law is supplied; that is a successful conventional explanation, not a deficiency to manufacture away.

The consequential next question is **which relationships in experience remain interchangeable for useful decisions when the preparation, response vocabulary and persistence boundaries are no longer supplied?** Retained structured estimation, episodic conditional completion, generic causal-history learning and direct decision reuse are serious competing formulations. No latent-state, model or policy architecture is logically required by that question. The next test must show a learned relation transferring to actual new interventions/conditions, or falsify its sufficiency, rather than add another coefficient/optimizer menu.

Unknown initial motion is a decisive diagnostic. In the constant-B surrogate, p1=p0+Tv0+.5T²Bu. Choosing vB-vA=.5T(BA-BB)u makes the first observation identical for distinct B. A probe that identifies a prepared response need not identify it without rest. Extra observations can remove this ambiguity; another cadence/rank theorem alone would repeat identification drift. Similarly, a camera-rotation/chirality construction can expose coherent grouped knowledge in a surrogate, but nonlinear leakage and engineered symmetry make it optional, not automatic escalation.

`cartesian_v16_direction_independent.md` and `after_cartesian_v16_question_independent.md` actively challenge the scope and propose observable falsifiers. Do not conclude universality from this local theorem/task, do not increase difficulty solely to defeat a conventional method, and do not indefinitely postpone a less supplied learning construction by refining integration. Continue the ACTIVE general-learning Goal after publication, preserving honest conventional sufficiency and negative evidence. The ultimate objective has not been achieved.

---

# Complete preceding living ledger, preserved verbatim

'''
    doc=doc.replace('{{TABLE}}','\n'.join(table)).replace('{{SELECTIONS}}',str(selections))
    prior=(ROOT/'first_principles_robot_learning_ledger_v15.md').read_bytes()
    target.write_bytes(doc.encode()+prior)
    assert target.read_bytes().endswith(prior)
    print('saved',target,'prior_sha256',hashlib.sha256(prior).hexdigest())


if __name__=='__main__':main()
