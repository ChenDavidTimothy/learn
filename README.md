# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v12](research/first_principles_robot_learning_ledger_v12.md), self-contained with all prior ledgers preserved verbatim.

V12 compares direct policy search with predictive learning from the exact same collected experience. Conventional OE/EKF/MPC yields lower mean goal cost on four supplied toy plants; repeating CEM inside the learned model does not consistently help the same direct controller. Observation-only fitness and feedback-data likelihood assumptions, independent audits, failed pilots and all historical corrections are retained. **No claim of universal learning or competitive robotics performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/shared_experience_v12.py --output-prefix /tmp/shared_v12
cmp research/shared_experience_v12_results.json /tmp/shared_v12_results.json
cmp research/shared_experience_v12_data.npz /tmp/shared_v12_data.npz
.venv/bin/python research/model_reuse_v12.py --output-prefix /tmp/reuse_v12
.venv/bin/python research/audit_shared_experience_v12.py
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v11](research/first_principles_robot_learning_ledger_v11.md), [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
