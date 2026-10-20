# AGENTS.md — Build and test first-principles learning mechanisms

## 1. Fixed objective and present state

Discover, derive, construct and rigorously evaluate our own mechanisms for learning
from experience, with the ambition of competing with strong modern methods and
scaling toward unfamiliar nonlinear systems, robotics, articulated bodies and
humanoids. A mechanism must acquire useful capability through experience; do not hand-code
the capability claimed as acquired. Legitimate labels, task inputs, supplied
features and shared physical priors are allowed when appropriate, disclosed and
fairly compared. Every method has assumptions. Test what experience actually adds.

The objective is fixed; the scientific route and architecture remain open.
First-principles derivation and independent scientific judgment take priority over
our previous hypotheses, familiar paradigms and fashionable methods. Our own
mechanism means a concrete, understood learning rule with a defensible purpose,
not renaming an existing algorithm. Genuine originality is desirable but cannot
be promised. If it reduces to an established method, acknowledge that and test
whether the derivation offers a substantive advantage. Retire a rediscovered
candidate when it has none; do not disguise it or repeatedly rescue it to satisfy
an ownership or novelty requirement. Known-method screenings and baselines remain
useful work without becoming our proposed new mechanism.

**Current state:** v17 is the latest verified ledger. There is no demonstrated
breakthrough, reliable general learner or competitive framework. The campaign is
paused at the user's request. This instructions revision does not resume it.
Read `research/campaign_goal.md` and the newest opening of
`research/continuation_handoff.md` for the operational objective and pause state.
Historical instructions to continue immediately do not override a user pause.

## 2. What counts as progress

Primary progress is an executable learning mechanism and credible evidence that
it gains a useful capability, reduces necessary prior knowledge, improves
experience/computation efficiency, transfers better, or matches strong methods
with a consequential simplification. A fundamental theorem can also matter if it
rules out a serious route or changes the construction of a general learner.

A derivation, diagnostic, numerical refinement, simulator extension, audit,
publication or ledger version is not sufficient progress by itself. Ask what
learning-system decision the result changes. If neither plausible outcome would
change the candidate, the comparison or the scientific route, defer the work.

Toy systems are instruments for discriminating hypotheses, not the destination.
Do not repeatedly solve the same learning problem under a new sensor or simulator
and present the interface change as a new learning mechanism. Do not mistake
ordinary fitting, parameter identification or controller engineering for evidence
of a general framework. Judge correctness, novelty and consequence separately.

## 3. Derive before adopting, without prolonged elementary detours

Start with a precise capability the learner lacks and the experience available to
it. Challenge that framing. Identify necessary distinctions, explicit assumptions
and the smallest falsifier. Consider seriously different explanations when they
could change the learning rule, information requirements or predictions.

Do not assume that learning must be founded on states, representations, models,
rewards, policies, prediction, compression, causal inference, symbols or networks.
Use any of these when justified. A change of vocabulary alone is not an alternative
mechanism. Known mathematics and standard software are tools, not foundations that
must be rediscovered. Use a short derivation or counterexample to settle an easy
point and move on. Deep theory is justified by its consequence, not its difficulty.

Study primary research early enough to discover strong existing answers and avoid
straw competitors. Reason independently where useful, then reconcile with the
closest prior methods. Read the relevant actual method/experiment sections and
verify current software/benchmark choices. Do not delay implementation for an
exhaustive literature survey or use novelty as a reason to reject a better method.

## 4. Start each campaign with a short mechanism proposal

Keep one current `research/campaign_plan.md` when a campaign is actually resumed.
It is a compact working document, not a new reporting system. Before expensive
work, state:

- The target capability and the consequential limitation being addressed.
- The proposed learning operation and why experience should change capability.
- What is supplied, what is acquired, and what transfers to an unseen condition.
- The strongest conventional explanation and relevant competitive baselines.
- A discriminating experiment, failure criterion and the decision each outcome
  would change. Include data, compute and elapsed-time budgets.

If the idea is not yet an algorithm, identify the one unresolved question that
prevents construction and bound the work needed to resolve it. Do not substitute
an open-ended list of questions for a mechanism proposal. Choose one main
candidate for the first implementation; do not build an architecture menu.

The programmer-supplied/acquired distinction must be concrete: identify actual
inputs, initialization, update, persistent information and decision use, whatever
the mechanism calls them. Show what changes when experience is removed, shuffled,
restricted or transferred using an ablation appropriate to the scientific claim.
No universal set of internal variables or ablations is prescribed.

## 5. Execution discipline and drift limits

For a campaign lasting hours, use these default decision deadlines. Shorten them
when the question is simple; never spend the whole allowance merely because it
exists. They are effort controls, not promises of discovery.

- **Within 30 minutes:** select a consequential mechanism question and a small
  implementation/comparison plan after initial derivation and prior-art checks.
- **Within 90 minutes:** have an executable candidate or a specific demonstrated
  obstacle to construction. A plan, theorem, simulator or audit alone does not
  satisfy the candidate requirement.
- **By two hours:** obtain a first discriminating candidate-versus-baseline result
  where computationally feasible. A smoke test is not evidence of capability.
  If a long training run is necessary, launch a validated, budgeted run and use
  interim checks; explain the required time before committing to it.
- **Every further hour and at major result boundaries:** assess capability gained,
  evidence against the mechanism, supplied assumptions, comparison strength and
  time allocation. Continue, revise or abandon the branch for a stated reason.

Do not manufacture a weak candidate to meet a deadline. Missing a deadline means
reconsider the branch, not extend it silently. A consequential theoretical route
may need longer: justify the exception prospectively by the construction or
impossibility decision it enables and specify a new bounded endpoint. Do not keep
renewing the same exception without new evidence.

Use rough wall time, not elaborate time-accounting infrastructure. Candidate
construction, decisive evaluation and serious comparisons should dominate effort.
If diagnosis, tooling, audits or documentation have consumed most of a working
hour, reassess before doing more. A safety-, correctness- or claim-critical check
must still be done; narrow the claim or stop the branch if it cannot be validated.

After a failure, permit a targeted diagnostic only when it can choose between
materially different next actions. Try the cheapest adequate check first. Do not
automatically refit every record, replay every trajectory, enumerate hypothetical
commands or raise numerical precision. One substantial, evidence-based reformulation
is reasonable; repeated rescues require a direction review. If two serious
formulations fail without a defensible next mechanism, end that campaign with an
honest handoff rather than consume the remaining budget on low-value tasks.
This ends an unsuccessful attempt, not the ultimate objective.

No routine approval is needed at these checkpoints. Continue autonomously within
the user's authorized campaign and budget; respect pauses and stopping requests.
A release is neither a stopping condition nor an instruction to extend a dead end.

## 6. Competition and generalization must become real

Choose strong, relevant, executable baselines when proposing the candidate, before
claiming success. Include a competent simple conventional method and a relevant
modern method for capability claims, unless one demonstrably covers both roles.
Use maintained implementations or verify faithful reproductions where practical.
Published numbers from incompatible conditions are context, not a direct comparison.

Keep observation access, interaction opportunities, train/test boundaries, task
information and constraints fair. Disclose privileged information, tuning and
pretraining. Report actual interaction/data, planning/training compute, wall time
and tuning cost. Equal hyperparameter counts are not automatically fair; use
competent settings and comparable tuning resources. Do not starve a baseline,
optimize only ours, or compare an entire mechanism against a deliberately crippled
component. A privileged reference is an authority diagnostic, not an ordinary
learning baseline. A homemade neural pilot is not a stand-in for modern methods.

Start with the smallest discriminating task, then test on a meaningfully different
held-out condition required by the claim: dynamics, embodiment, observation,
objective or task structure. Multiple seeds on one body measure repeatability,
not cross-system generality. Specify a transfer test in the plan; do not indefinitely
postpone it behind toy refinements. Escalate complexity because it tests the
mechanism's claimed advantage, not to make a preferred competitor fail.

Early screenings need not reproduce an entire benchmark suite. However, an early
success earns a stronger comparison, not a breakthrough label. A framework claim
requires convincing evidence across relevant systems and strong competitors.
If competitive validation exceeds local compute, establish only the narrower
claim and state the remaining requirement. Never convert that limit into success.

## 7. Evidence and evaluation

Separate exploration, selection and confirmation. Freeze the question, primary
comparison, sampling scope and failure criteria before confirmatory evaluation.
Use held-out experience and appropriate independent units. Do not reuse test
outcomes to select a mechanism and then call them fresh evidence. Report failed
seeds, unstable runs, inconvenient comparisons and development interventions.

Use measurements that test the claim: performance, transfer, experience efficiency,
robustness, prior knowledge or computation as applicable. Prediction error alone
is not decision competence. Relative improvement is not absolute task sufficiency.
Removing future measurement noise does not remove hidden conditional uncertainty.
Good solver termination is not model correctness. Do not repeatedly prove these
elementary distinctions instead of constructing and testing the next mechanism.

Use the following labels when consequential:
**PROVEN UNDER ASSUMPTIONS**, **TESTED**, **EXISTING RESEARCH**, **HYPOTHESIS**,
**OPEN**, **REJECTED / CORRECTED**. State the exact supported scope. An elegant
identity is not automatically original; an interpretation is not an algorithmic
advantage; simulation success is not general intelligence.

## 8. Verification proportional to the claim

Use staged verification:

1. Before expensive runs, check critical data flow, update semantics, leakage,
   units and benchmark correctness with small fixtures and trusted components.
2. Screen whether the candidate is scientifically worthwhile. Preserve screening
   data and label it exploratory; do not publish an unverified success.
3. Independently challenge the consequential mechanism, derivation, baseline
   fairness and decisive result before making a verified claim.
4. Expand numerical checks or replay scope when sensitivity, discrepancies or
   the intended claim warrant it. Reuse validated unchanged components.

Do not repeatedly independently integrate every physical step, reconstruct every
fit or build exhaustive source-hash machinery by default. Existing validated
physics and maintained benchmark tools should free effort for learning research.
A claim requiring extensive verification must receive it or remain unverified.
This rule is proportionality, not permission to ignore correctness or hide error.

On a numerical failure, preserve the failure and inputs, identify whether it
changes the scientific conclusion, and repair proportionately. No silent tolerance
widening. Once resolved to the precision needed for the claim, move on; another
precision ladder is not a research contribution. Do not overwrite released caches
when rerunning an old audit. Use fresh paths or preserve the published checkout.

## 9. Independent agents must improve scientific decisions

Use subagents when useful for a genuinely independent competing derivation,
mechanism, serious prior-art/baseline challenge, counterexample or decisive-result
reproduction. Avoid coordination infrastructure and redundant execution.

For a substantial campaign, assign an independent direction critic early enough
to challenge the central premise, supplied intelligence and baseline strength.
At the first consequential result, have that critic recommend continue, pivot or
stop, explaining opportunity cost. This is a scientific recommendation, not an
extra user-approval gate. A theorem or artifact checker is not a substitute.

Auditors should try to falsify claims and can recommend abandoning the chosen
route. Distinguish independent derivation from reused helpers and repeated runs.
The coordinator remains responsible for integration and must act on criticism or
record a scientific reason for disagreeing. Do not spawn agents just to satisfy
a quota or leave them auditing a branch that should already be abandoned.

## 10. Durable history without publication-driven research

Read the latest verified living ledger and current handoff before a new campaign.
Use older work selectively for dependencies; do not repeat completed investigations
or treat past conclusions as authority. Preserve discoveries, derivations,
experiments, primary references, mistakes, negative results and corrections.

Commit reproducible source, actual results and concise conclusions as progress
warrants. Unfinished work can be checkpointed honestly without calling it verified.
A new sequential ledger is for a substantive scientific change, not each experiment,
audit or unit of elapsed work. Keep the existing convention that every new ledger
incorporates the complete preceding ledger verbatim. Never rewrite released
history to make the program appear more successful.

For a verified milestone, complete the claim-critical audit, save executable
experiments and actual results, write the self-contained ledger, commit and push
over the existing Git SSH remote, then verify the remote contains the commit.
Record findings, limitations and the best next question. No HTTPS, force-push or
discarding unrelated work. Publication failure must be reported and diagnosed.

Continue after publication only when the next investigation passes the consequence
gate and the campaign remains authorized. Do not optimize version counts. A pause,
AI change or execution limit requires an exact current handoff with verified state,
unexecuted proposals, failed routes and the next meaningful decision.

## 11. Existing environment and working documents

Canonical project: `/home/dtdav/dev/learn`. Use its existing Git repository and
`.venv/bin/python` on the owner's PC. Preserve dependencies, configuration,
credentials and unrelated work. Do not create unnecessary projects, environments,
services or research tooling. Inspect changes before editing.

Current operational documents:

- `research/campaign_goal.md`: objective, deliverables and failure boundaries.
- `research/campaign_plan.md`: one compact plan, created only when research resumes.
- `research/continuation_handoff.md`: authoritative newest state at its beginning.
- `research/first_principles_robot_learning_ledger_vN.md`: verified scientific history.
- `research/research_process_lessons_v17.md`: why the previous process drifted.

Do not turn these documents into a parallel bureaucracy. Update them when a
scientific decision or handoff changes; ordinary work belongs in code and results.

## 12. Completion standard

The aim is an actual first-principles learning mechanism with credible competitive
capability and a defensible route toward nonlinear robotic intelligence. A long
run, clean audit, many ledgers, isolated theorem or toy improvement is not completion.

A candidate may fail. Established methods may win. Report that honestly and use it
to change the next scientific decision. Never force novelty, lower the benchmark,
conceal supplied structure or declare a breakthrough to satisfy the objective.

**Build mechanisms, challenge them with strong alternatives, and let consequential
evidence determine what deserves the next hour.**
