# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v10](research/first_principles_robot_learning_ledger_v10.md), self-contained with all prior ledgers preserved verbatim.

V10 constructs and audits learning from noisy position through new-goal feedback control. Learned linear models track about as well as more accurate nonlinear models on this four-plant grid; hidden-state uncertainty and stopping precision remain important limits. A failed optimizer pilot and stronger PD baseline corrections are preserved. V9 distinguishes nonlinear physical attribution from predictive learning; earlier ledgers retain all findings and corrections. **No claim of universal learning or competitive robotic-humanoid performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/goal_control_v10.py > /tmp/goal_control_v10_check.json
cmp research/goal_control_v10_results.json /tmp/goal_control_v10_check.json
.venv/bin/python research/goal_control_PD_v10.py > /tmp/goal_control_PD_v10_check.json
cmp research/goal_control_PD_v10_results.json /tmp/goal_control_PD_v10_check.json
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
