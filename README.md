# First-principles robot learning

Research ledgers, reproducible learning/control experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v15](research/first_principles_robot_learning_ledger_v15.md), self-contained with all prior ledgers preserved verbatim.

V15 learns coupled continuous behavior from noisy position/action histories and uses it for switched-goal control without rest resets. Conventional history and filtered-motion methods work; validation-selected history shows no clear advantage over a fixed rich window. Prediction, position cost and precise stopping remain distinct. Eight acquired datasets, independent audits, complete failures and prior history are preserved. **No claim of novelty, general intelligence or competitive real robotics.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/continuous_history_v15.py --replicate 100 --output-prefix /tmp/continuous_v15_rep100
.venv/bin/python research/summarize_continuous_history_v15.py
.venv/bin/python research/audit_continuous_history_v15.py
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v14](research/first_principles_robot_learning_ledger_v14.md), [v13](research/first_principles_robot_learning_ledger_v13.md), [v12](research/first_principles_robot_learning_ledger_v12.md), [v11](research/first_principles_robot_learning_ledger_v11.md), [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
