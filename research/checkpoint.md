# WGR-AK local computation checkpoint
Session: 18 October 2026, Europe/Berlin. Origin: user-supplied 907-line artificial_agency_research_handoff.md; 18 October 2026 handoff read in full. MCP Control manual also applied.

## 1. Scientific state and provenance
Target: warranted goal reconsideration under acquired knowledge, commitments, authority and incomplete information. H1 existing methods suffice; H2 integration gap; H3 missing mechanism. H2 was a working preference, NOT confirmed. This session implemented the FIRST greenhouse pilot; the original handoff correctly indicated none had previously run. Philosophy, agency, consciousness and ultimate values remain outside the measured target.
The research handoff provides relevant prior art including Johnson et al. on Goal Reasoning and Trusted Autonomy (2018), Schut et al. on intention reconsideration (2004), Addison's La VIDA journal article (2025), and AgentAbstain (2026). We did not independently reread/reverify papers during this computational session. None of our work demonstrates novelty.

## 2. Actual computer and repository
MCP Terminal read/write/command execution observed. No sandbox interpreter was used. Local machine: Linux x86-64 kernel 6.17.0-22-generic, Intel i5-12400F, 12 logical processors, 15 GiB RAM (about 7.9 GiB available at initial probe), RTX 3060 Ti 8 GiB, about 518 GiB free on root filesystem. Shell bash, Python 3.12.3.
Repository: /home/dtdav/dev/learn; current branch main, pretask clean working tree, HEAD b9989f5. Existing commit history retained; current tracked tree initially only .gitignore. Existing Python: /home/dtdav/dev/learn/.venv/bin/python; only standard-library modules used for this pilot. No dependencies installed, no machine service altered, no GPU computation.
No AGENTS.md was found at the four checked parent/project paths. No external API credentials, network calls, actuators, cloud inference or paid resources used.

## 3. Files and execution protocol
- research/protocol.md: protocol and gates written before agents were evaluated.
- agency_pilot/model.py: independent mutable world, partial observations, inspection, authority and physical action traces.
- agency_pilot/agents.py: persist, suspend, reactive, rule-integrated belief planner, privileged oracle, and post-hoc negative control.
- agency_pilot/experiment.py: deterministic case replay, scored trace export and summary.
- tests/test_pilot.py: executable consistency tests.
- runs/2026-10-18-wgr-pilot/traces.jsonl and summary.json: retained project results.
- README.md: commands and interpretation limitations.

Replay from the project root:
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m agency_pilot.experiment
Both were executed locally and completed successfully. Tests: 12 passed, no failures (last run).
The post-pilot trace audit caught one environment error: a no-flow irrigation command incorrectly ended the episode. Corrected it to allow subsequent investigation and repair; exposed nonprivileged last_result feedback; added a regression test; reran the entire pilot. Failed initial attempt is not reported as final evidence.

## 4. Frozen evaluation and final observed results
11 pairs (22 cases), two conditions (A complete relevant knowledge, B partial missing/uncorroborated observations), six policies: 264 deterministic episodes. Per policy, 22 evaluated episodes per condition and 11 matched pairs. Fixed record identifiers/seeds 101 through 122; IMPORTANT: no pseudorandom draws consume these seeds in v0.1. They are reproducibility metadata, not evidence from independently sampled stochastic trials. The actual simulator is deterministic and case-authored.
Condition A, successes/22, matched pairs/11:
- always_execute: 11, 1
- always_suspend: 7, 1
- reactive: 22, 11
- belief_planner: 22, 11
- oracle: 22, 11
- posthoc: 11, 1
Condition B, successes/22, matched pairs/11:
- always_execute: 11, 1
- always_suspend: 7, 1
- reactive: 12, 1
- belief_planner: 22, 11
- oracle: 22, 11
- posthoc: 11, 1
The belief planner used 17 inspection steps in B, 0 in A, and never executed a prohibited action. Its mean episode length was 1.818 steps in B versus 1.045 in A. Condition B reactive incurred 54 prohibited execute EVENTS, inflated by repeated attempts up to the six-step budget; this is not 54 independent unsafe cases. Agent scores and trace events, not spoken explanations, determine success.
Example B trajectories: visitor_warning.true/belief_planner: inspect:effect -> suspend; reliability.unverified/belief_planner: inspect:effect -> suspend; valve.blocked/belief_planner: inspect:blocked -> repair_valve -> execute. effect.harmful/posthoc: execute (harm violation) -> suspend, scored failure.

## 5. Interpretation and decisive limitations
This validates only the harness's ability to distinguish rigid persistence, overcaution, uncertain evidence, role-limited authorization, blocked means and post-hoc restraint. It exhibits an engineered integration difference between an optimistic reactive control and a purpose-built planner under partial information. It does NOT distinguish H1 from H2 for real general-purpose agents. The belief planner is tightly fitted to known greenhouse policy variables; oracle/grader uses a parallel representation of the same rules. Success is near-tautological in the deterministic, low-dimensional task and cannot establish robust generalization.
There is no independent human label review, model training, probabilistic calibration, noisy sensor process, true acquired causal structure, development, third-party goal-reasoning/BDI implementation, held-out mechanism transfer, second domain, statistical population sampling or scientific novelty claim. A perfect inspection returns ground truth. Paired labels are internally checked, not externally validated. No significant confidence interval or p value is scientifically justified from 22 handcrafted cases.

## 6. Go/no-go and next justified experimental step
Go: strengthen test validity by independent semantic review of cases and construct a separate held-out Condition C requiring genuine experience-dependent inference with matched memory-on/off baselines and discriminative novel versus superficial features. Add a serious externally grounded BDI/GDA or other established existing-method competitor, avoiding hand-coded oracle-aligned benchmarks. Prespecify mechanisms, training/test separation, action costs and resource budgets. Only then draw conclusions about H1 versus H2.
No-go: GPU/LLM training, an invented new agency architecture, publication-scale claims, or theoretical H3 claims from this preliminary simulator. If a known planning method works under fair held-out generalization, treat H1 as supported in that domain rather than searching for forced novelty.

## 7. Cleanup and handoff
All deliverables reside beneath /home/dtdav/dev/learn. No screenshots/inspection captures were created; no persistent test servers or background worker processes were started. Managed MCP services and Terminal were not restarted/stopped. Work remains uncommitted for owner review; unrelated existing repository data was not changed. Next session should inspect git status, read research/protocol.md, this checkpoint, and stored traces before altering the protocol.