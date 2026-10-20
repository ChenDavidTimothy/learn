# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v14](research/first_principles_robot_learning_ledger_v14.md), self-contained with all prior ledgers preserved verbatim.

V14 learns coupled two-joint displacement relations from noisy experience. Broad pose coverage supports new goal decisions; narrow pose coverage extrapolates poorly. The chosen inverse relation outperforms the forward relation within this basis, with all physical, preparation and noise assumptions explicit. Independent audits, information-use repairs and complete prior history are preserved. Next work removes rest preparation and tests relevant history. **No claim of general intelligence or continuous robotic control.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/coupled_learning_v14.py --output-prefix /tmp/coupled_v14
cmp research/coupled_learning_v14_results.json /tmp/coupled_v14_results.json
cmp research/coupled_learning_v14_data.npz /tmp/coupled_v14_data.npz
.venv/bin/python research/audit_coupled_learning_v14.py
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v13](research/first_principles_robot_learning_ledger_v13.md), [v12](research/first_principles_robot_learning_ledger_v12.md), [v11](research/first_principles_robot_learning_ledger_v11.md), [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
