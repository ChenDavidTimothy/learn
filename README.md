# First-principles robot learning

Research ledgers, reproducible learning/control experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Research is paused.** The next campaign is governed by the revised
[research instructions](AGENTS.md) and [campaign objective](research/campaign_goal.md):
construct a concrete first-principles learning mechanism and evaluate it against
strong relevant methods. Read the [current handoff](research/continuation_handoff.md)
and [lessons from v17](research/research_process_lessons_v17.md) before resuming.
The earlier exhaustive diagnostic proposal is optional, not the default agenda.

**Current release:** [Living Research Ledger v17](research/first_principles_robot_learning_ledger_v17.md), self-contained with all prior ledgers preserved verbatim.

V17 constructs learning from continuous moving Cartesian camera/action history without certified initial rest or exact position. Fresh replication corrects the exploratory pilot's apparent reliability: retaining new experience reaches 55/192 sustained precision windows, versus 19/192 with the acquired library frozen and 192/192 for a privileged reference. Retention improves relative cost but does not establish reliable control. All 310 failed windows, numerical corrections and independent audits remain. **No claim of novelty, general intelligence or competitive real robotics.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/cartesian_stream_v17_acquire.py --replicate 200 --output-prefix /tmp/v17_rep200_reproduction
.venv/bin/python research/cartesian_stream_v17_study.py --replicate 200 --prefix /tmp/v17_rep200_reproduction
.venv/bin/python research/analyze_cartesian_stream_v17_study.py --require-audits --output /tmp/v17_aggregate_reproduction.json
.venv/bin/python research/build_ledger_v17.py --check
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v16](research/first_principles_robot_learning_ledger_v16.md), [v15](research/first_principles_robot_learning_ledger_v15.md), [v14](research/first_principles_robot_learning_ledger_v14.md), [v13](research/first_principles_robot_learning_ledger_v13.md), [v12](research/first_principles_robot_learning_ledger_v12.md), [v11](research/first_principles_robot_learning_ledger_v11.md), [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
