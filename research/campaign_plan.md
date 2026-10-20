# Decision: close the two-candidate attempt; ultimate objective unresolved

2026-10-20. The second frozen screen is complete and independently checked.
Implicit relation conditioning fails its continuation gate on fresh bodies:
OOD MSE at 16/64 support is .710211/.578818 versus CoDA .609899/.306228,
ALPaCA .636735/.464097 and nested explicit .707521/.564881. Query conditioning
is healthy. No numerical rescue, hyperparameter search, control rollout or timing
benchmark is warranted. Read implicit_relation_findings.md and independent
check_implicit_relation_results.md for actual evidence and finite scope.

Both serious formulations in this attempt are now retired. This is the point to
end the attempt under AGENTS.md, not declare the ultimate Goal achieved or paused.
No third architecture is selected and no v18 ledger is justified. The scientific
objective remains active: find an actual learning mechanism with defensible
competitive capability. Two defeated inductive biases are negative evidence,
not a breakthrough, a universal impossibility, or grounds to lower the standard.

The next investigation starts with a fresh direction decision: identify a real,
consequential limitation of the strongest applicable learning account, derive
what experience must change to overcome it, and check serious existing answers
before selecting another executable candidate. Do not infer a lack of task
boundary handling from these stationary experiments or run old v17 diagnostics
by default. This is a handoff of the decision, not an unexecuted experiment menu.

Full earlier plans and decisions follow unchanged.

---

# Current second attempt: implicit relation meta-conditioning

2026-10-20. Joint error transport is retired. The second independently framed
candidate learns feature relations, conditions their coefficients on observed
transitions by ridge regression, then solves a small relation for each query.
Read implicit_relation_plan.md for the frozen proposal, assumptions, budgets and
failure criteria. Original train/validation and two frozen ALPaCA/CoDA fits are
reused. Sixteen fresh held-out bodies have been collected from the declared RNG
before fitting; no test score has been inspected. Candidate and matched explicit
fits each receive two learning rates and 2,000 steps, selected on validation only.

Independent tiny formula/nesting/gradient checks pass. The direction critic
supports exactly one bounded screen, with explicit capacity and stability
caveats. This remains related to established implicit identification and learned
feature adaptation; no novel general principle is claimed. A failure ends this
attempt without rescue tuning; a success earns confirmation and decision tests.
No v18 is warranted by selecting or screening another candidate.

The complete prior branch decision and original proposal follow unchanged.

---

# Branch decision: retire joint error transport after the frozen screen

2026-10-20. The one-seed exploratory screen is complete and independently checked.
At 16/64 support observations on held-out out-of-range bodies, candidate scaled
MSE .750616/.640077 versus CoDA .359611/.306448 fails the declared continuation
criterion. The matched separable update and ALPaCA also beat it. Correct support
helps, but the extra operation earns no practical advantage here. No tuning,
new seeds, control tests or new ledger will rescue this branch. Scope is one
small fully observed noiseless physical family and a finite training budget.

Read error_transport_screen_findings.md for the actual decision and limitations.
The frozen proposal follows intact; its selected-candidate language is historical.
The ultimate Goal remains ACTIVE and unresolved. Next effort must identify an
independent consequential learning limitation and derive a justified mechanism,
not extend this failed candidate or automatically execute the old arm diagnostics.

---

# Active campaign: learn what an experienced error should change elsewhere

2026-10-20. User resumed through the replacement Goal attachment
/home/dtdav/.codex/attachments/80880570-6653-4aa2-81e5-9266b707e227/goal-objective.md.
The copied PAUSED text in that attachment describes the prior state; the opening
request and active Goal explicitly resume the revised mechanism campaign.
Latest verified scientific ledger v17 has been read. No v18 milestone is assumed.

## Mechanism question and early rejection

Can experience learn a reusable rule for changing consequences at other inputs,
instead of the programmer supplying a distance metric or a fixed adaptation
subspace? Similar observed errors may imply different corrections depending on
where they occurred and the presently accumulated evidence. A useful rule must
adapt across held-out physical dynamics, not merely fit another sensor interface.

Rejected as a main candidate before experiment: command-centered regression risk.
It targets the same conditional action-dependent mean as competent ordinary
regression and does not itself supply a transferable learning operation. Unknown
mass changes drift AND gain. No drift-only benchmark will be used to favor it.

Selected candidate HYPOTHESIS: learn an error-conditioned functional correction
operator across multiple training systems. At query z and support input x_i,

    f_next(z) = f(z) + mean_i K_theta(z,x_i,r_i) r_i,
    r_i = y_i - f(x_i).

K is a learned bounded signed output-to-output matrix, dependent on the observed
residual, not a supplied proximity metric. Apply a fixed small number of updates
to support/query predictions. Meta-training minimizes errors at disjoint queries
of the same system after adaptation. Zero evidence returns the learned shared
predictor. Training across systems learns the update, actual new observations
change the adapted function. No physical parameter is an input or target.

This is NOT assumed novel. ALPaCA (2018) learns features/prior for analytic
Bayesian output-layer adaptation; separable linear K reduces to that family.
MetaFun (2020) already learns iterative function updates. CoDA (2022) learns a
low-dimensional parameter-adaptation subspace; GEPS (2024) is a more recent
low-rank dynamics adaptation comparator. Residual-conditioned matrix transport is
only a candidate variation; retire it without substantive advantage. Do not
present familiar meta-learning as our new framework. See source-reading note.

## Supplied and acquired

Supplied: smooth observation encoding, episode grouping of one physical system,
shared network family/normalization, square prediction loss, update count and
optimization. First screen uses observed joint q/v to isolate update learning,
not to solve v17's camera ambiguity or claim less prior information. The evaluator
knows mechanics but the learner receives measured transitions and commands only.
Acquired: base prediction and the relation from an error at one observed input
to corrections at other inputs. Shuffled-system support and zero support test
whether within-system experience actually carries the claimed knowledge.

## First discriminating screen, frozen before sampling

One controlled nonlinear two-link family with independently changing link
lengths, masses, damping and actuator gains/cross-coupling: 32 training, 8
validation, 8 new in-distribution and 8 out-of-range physical bodies. Exact
numerical ranges and RNG roles will be executable in the acquisition source
before sampling, not selected from outcomes. Use 8 continuous 96-step trajectories
per body, dt .05, randomized held-pair commands, random moving starts. Ordinary
inputs encode sin/cos q, v and command (8 coordinates), targets are observed
(v_next-v)/dt (2 coordinates), not true acceleration labels. This initial screen
is noiseless and fully observed; no robustness/partial-observation claim.

Training support is a contiguous prefix/window from one trajectory; queries come
from distinct trajectories of the same body. Validation/test supports are fixed
prefixes of trajectory0; query trajectories1--7 never supply adaptation labels.
Support budgets 0/4/16/64. Train-only normalization. Independent bodies are the
statistical unit; overlapping transitions are not independent robots.

Comparators: pooled nonlinear predictor with validation-tuned ordinary fine-tuning;
learned-feature Bayesian adaptation (ALPaCA family, with explicit reproduction
scope); low-rank whole-network context adaptation (CoDA family, checked against
original method). The latter is the required serious dynamics-meta-learning
comparison, not a homemade frozen neural straw model. MetaFun overlap must be
acknowledged; if results justify expansion, compare a matched functional-update
baseline as well. Use comparable network/tuning resources and disclose unequal
compute. Do not claim full published-method reproduction from an inspired model.

Primary early decision: body-averaged scaled consequence MSE as support grows,
plus raw-coordinate MSE and adaptation time. A useful continuation needs at least
10% mean improvement over the strongest tuned comparator at 16 or64 supports on
the out-of-range screen, without an equally large opposite loss at the other
budget, and benefit from correct over shuffled support. This is a prospective
screening heuristic, not significance or confirmatory proof. No outcome tuning
on test bodies. If it loses, reject/reformulate rather than extend diagnosis.

Prediction gain alone does not satisfy the objective. If the screen survives,
freeze fresh-body confirmatory sampling and evaluate actual goal-directed
closed-loop cost with common sensing/commands/planning, plus a second different
controlled nonlinear family using the same learning operation. Holdout dynamics
is the first transfer claim; robotic-general framework claims require much more.
If the screen fails, no control rollout is owed merely to finish a table.

## Effort and verification

Aim for executable candidate within90 minutes, first paired screening within2h;
initially ~2000 training steps/method, two validation learning rates, one discovery
seed, then fresh seeds ONLY if evidence warrants. Owner GPU RTX3060Ti8GB and12CPU
cores; use existing Torch/NumPy/SciPy/project.venv. Record walltime/parametercounts,
keep data/model checkpoints/results and failures. No package installs or services.

Small fixtures: base unit-body physics matches validated parent, finite integration
refinement on a few representative flows, no support/query leakage, train-only
scaling, update gradients/shape/zero-support and ALPaCA solve checks. Independent
critic challenges premise/fairness; verify consequential outcomes proportionately.
Do not replay every historical step, build new hash infrastructure, or publish
another ledger just for this plan. At the first result, continue/pivot/stop must
name what capability/learning rule the evidence actually supports.

## Pre-fit critic correction

Actual MetaFun method reading confirms direct prior art, not just a distant
analogy: local evidence update + learned similarity aggregation + iteration is
already its central operation. Before ANY fit, add a matched separable
MetaFun-family comparator mean_i k(q,s_i)*[U(s_i,r_i)-U(s_i,0)], with nonlinear
cross-output U, learned scalar similarity and the same base/encoder/threeupdates.
The candidate distinction is joint query-support-residual matrix transport.
If this matched operation matches it, retire that variation; do not rename it a
new framework. Bounded entries do not imply contraction; preserve blowups.
This amendment adds one necessary direct comparator, not an architecture search.
