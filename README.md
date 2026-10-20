# First-principles robot learning

Research ledgers, reproducible learning/control experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v16](research/first_principles_robot_learning_ledger_v16.md), self-contained with all prior ledgers preserved verbatim.

V16 learns local response variation from unlabeled Cartesian experience and reuses it through six successive goals without motion or belief resets. Across eight acquired corpora, a mixture reaches 383/384 precision endpoints, a full conventional Gaussian 363, and its trace-matched isotropic diagnostic 250. The rare mixture failure, unfinished fits, numerical corrections and independent audits remain. Exact initial rest and a local response vocabulary are supplied. **No claim of novelty, general intelligence or competitive real robotics.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/cartesian_feedback_v16_study_initial.py --replicate 100 --output-prefix /tmp/cartesian_v16_rep100
.venv/bin/python research/analyze_cartesian_feedback_v16.py
.venv/bin/python research/audit_cartesian_feedback_v16_study.py --replicates 100 101 102 103 104 105 106 107
.venv/bin/python research/audit_cartesian_feedback_v16_aggregate.py
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v15](research/first_principles_robot_learning_ledger_v15.md), [v14](research/first_principles_robot_learning_ledger_v14.md), [v13](research/first_principles_robot_learning_ledger_v13.md), [v12](research/first_principles_robot_learning_ledger_v12.md), [v11](research/first_principles_robot_learning_ledger_v11.md), [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
