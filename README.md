# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v11](research/first_principles_robot_learning_ledger_v11.md), self-contained with all prior ledgers preserved verbatim.

V11 separates learned-model error, live information limits, and a planner/executor constraint mismatch. More training barely helps the selected case; passing the existing cutoff deadline into the planner removes most conditional endpoint bias without changing sensing. Stopping remains imperfect. V10 constructs learned feedback control; all earlier results, failed experiments and corrections remain preserved. **No claim of universal learning or competitive robotic-humanoid performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/precision_information_v11.py > /tmp/precision_information_v11_check.json
cmp research/precision_information_v11_results.json /tmp/precision_information_v11_check.json
.venv/bin/python research/deadline_control_v11.py > /tmp/deadline_control_v11_check.json
cmp research/deadline_control_v11_results.json /tmp/deadline_control_v11_check.json
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
