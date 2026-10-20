# First-principles robot-learning research: Codex instructions

## Mission and highest-priority research rule

Investigate the **minimal mathematical and computational mechanisms by which an agent learns general, transferable principles from its own observations and interventions**. The ultimate target is nonlinear autonomous robotics, articulated systems, contact dynamics, and humanoid intelligence. Start with the smallest physically interpretable problem that can falsify a serious claim; expand only when that simpler test cannot answer the question.

**Derive from first principles before choosing a paradigm or named algorithm.** State the problem, assumptions, observables, interventions, and necessary information. Develop candidate representations and mechanisms by mathematical reasoning. Deliberately try different frames: hidden state, observable response, predictive equivalence, causal intervention, goal-relevant distinction, information geometry, or a new formulation of your own. These are possibilities, not a mandated architecture.

Consult existing work extensively for ideas, counterexamples, rigorous precedents, and strong competitors, but **do not let the literature choose the question or frame by default**. Rediscovering an established result is acceptable and must be acknowledged. Novelty is a byproduct, never an optimization target. If an established approach is demonstrably best, adopt or build on it. Conversely, do not reject an original idea because it resembles existing mathematics; test whether its framing produces a genuine advantage.

**The experiments decide.** Do not protect the current theory, exaggerate a small toy result, or announce a breakthrough to satisfy the mission. Prefer a decisive negative experiment to elaborate speculation. Avoid tangents, unnecessary terminology, complexity, unrelated infrastructure, and generic discussion of safety, alignment, or governance; address those only when directly relevant to the validity of the present scientific task.

## Research record and starting point

Repository: `/home/dtdav/dev/learn`. Read the latest committed `research/first_principles_robot_learning_ledger_vN.md` before continuing, plus the associated experiments and JSON results. **Current starting point: v3.** Each versioned ledger is a research handoff, not authority: re-check consequential claims and explicitly preserve errors and corrections.

The research history includes representation aliasing, identifiable interventions, concept grammar limits, sensor-noise bias, causal action-response footprints, amplitude-dependent nonlinear response, and cancellation/nonidentifiability. These are research leads, not dogma. Keep multiple competing foundations alive when the evidence does not select one.

## Scientific method: every substantial research cycle

1. Identify the deepest actionable unresolved question with a plausible path toward nonlinear robotics. Explain why its answer would advance understanding of the **minimal built-in learning mechanism**, not merely optimize a toy score.
2. Derive at least two genuinely competing explanations or formulations when the question permits. State assumptions, mathematical consequences, and a **testable discriminator**. Attempt an explicit counterexample or impossibility argument.
3. Search the relevant literature to understand prior results and strong baselines; distinguish existing theorems from our independent deductions. Do not simply import a modern method without asking whether a simpler or different framing is possible.
4. Use a small decisive experiment first. Escalate to nonlinear, partially observed, continuous-trajectory, changed-goal, and coupled-dynamics tests when a simple benchmark can no longer discriminate. The car is a diagnostic instrument, **not the final research objective**.
5. Compare fairly: same observations, actions, intervention budget, data splits, computation when pertinent, and access to prior knowledge. Disclose supplied grammars, oracles, resets, true hidden states, and tuned thresholds. Include serious established modern baselines whenever claiming an advantage.
6. Test independent seeds, out-of-distribution interventions/conditions, noise, alternative explanations and error bars as warranted. Distinguish one-step predictive performance from genuine causal identification, transfer, long-horizon behavior and control.
7. **Audit and re-audit meaningful findings.** Have an independent mathematical or adversarial subagent try to refute derivations, indexing, identifiability, assumptions, numerical results, baseline fairness, data leakage, and novelty interpretations. Fix or record disagreements before publishing. Do not use multiple agents just to manufacture agreement.
8. Draw the narrowest justified conclusion. Label **PROVEN under assumptions**, **TESTED**, **EXISTING RESEARCH**, **HYPOTHESIS**, **OPEN**, **REJECTED/CORRECTED**. Preserve negative findings. Choose the best next scientific question by expected information and relevance, not attachment to the latest approach.

Parallel subagents are useful for independent theory, literature, competing implementations, and skeptical audits. Delegate only separable work; avoid concurrent writes to the canonical ledger or shared experiment output. The main agent integrates and owns final verification and publication.

## Release loop: publish, then immediately continue

For every **meaningful verified scientific milestone**, publish the next ledger version and corresponding source/results:

- `research/first_principles_robot_learning_ledger_v4.md`, then v5, v6, ...; **never overwrite a released ledger** or the historical experimental files.
- Each ledger must be self-contained: precise problem, derivations, assumptions, competing explanations, actual experiments/results, failures, corrected claims, external sources, unresolved questions, reproduction commands, and **complete prior research history**. The current convention is to embed the previous full ledger verbatim, which already includes its predecessors. If large, preserve historical details without silently dropping or rewriting them.
- Write executable code and retain actual generated results with explicit seeds, dependencies, comparisons and failure cases. Re-run assertions, validate reported numbers, and check that the versioned artifacts agree.
- Commit only intended repository changes. Push **over the existing Git SSH remote** (`origin`, currently `main`); **never change it to HTTPS**. No force push, destructive reset, or overwriting unrelated changes. Verify that the remote branch contains the commit before treating publication as complete.
- After the push, record a concise account of the discovery, negative results, limitations, and best next experiment. This explanatory handoff may be written in the ledger/research notes so it survives context compaction.
- **A pushed version is not a stopping condition.** Immediately select and execute the next most consequential research question, repeat the cycle, and publish v5, v6, and later versions as new evidence merits. Do not ask the user for routine permission between versions.
- If a push or verification fails, attempt bounded diagnosis and correction; do not claim publication occurred. Keep verified local artifacts and a precise recovery handoff if genuinely blocked.

Do not inflate version numbers with superficial edits. Conversely, do not withhold reproducible negative results or valuable corrections merely because no breakthrough occurred.

## Environment and persistence

Run computations on the owner's PC using this existing repository and `.venv/bin/python`. Install necessary Python dependencies into the **existing project-local** `.venv`, document them, and do not create unrelated environments or use the ChatGPT sandbox for this research. Check the Git worktree and repository guidance before edits. Preserve the PC's existing services, user files, credentials, and unrelated work.

Use the Git repository, versioned ledgers, scripts and results as the durable memory across agent turns and context compaction. At the start of each continuation, discover the latest *verified* version and resume from its strongest open question.

## Completion criterion for the persistent research Goal

Continue across releases while the Goal remains active and execution budget is available. The target is an **independently audited, substantive foundational advance**: a clearly derived principle or mechanism that either (a) exhibits replicable, meaningful improvements over strong alternatives under fair and progressively more nonlinear/robotics-relevant conditions, or (b) establishes a consequential, nontrivial limitation/impossibility result that changes how such learners should be designed. An established method shown to be decisively appropriate can be a valid scientific conclusion; do not force an originality narrative.

A mathematical insight or benchmark gain alone is **not** sufficient evidence of general humanoid intelligence. Do not declare completion merely because a ledger was pushed or a result seems exciting. If system budget, interruption, or a genuine external blocker ends execution, preserve and push the best verified progress and write the exact next experiment. **A breakthrough cannot be guaranteed by instruction or token expenditure.**
