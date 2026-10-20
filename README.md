# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v13](research/first_principles_robot_learning_ledger_v13.md), self-contained with all prior ledgers preserved verbatim.

V13 uses existing learned models to optimize the same small controller family with verified derivatives. Mean tracking cost improves, while precise station keeping falls to1/64 fresh trials. V12's shared-experience comparisons, independent audits, failed checks/pilots and complete prior history remain preserved. The scalar optimizer branch is closed; next work challenges coupled dynamics and supplied function vocabulary. **No claim of universal learning or competitive robotics performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/gradient_reuse_v13.py --output-prefix /tmp/gradient_v13
cmp research/gradient_reuse_v13_results.json /tmp/gradient_v13_results.json
cmp research/gradient_reuse_v13_data.npz /tmp/gradient_v13_data.npz
.venv/bin/python research/audit_gradient_reuse_v13.py
.venv/bin/python research/audit_gradient_pathwise_v13.py
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v12](research/first_principles_robot_learning_ledger_v12.md), [v11](research/first_principles_robot_learning_ledger_v11.md), [v10](research/first_principles_robot_learning_ledger_v10.md), [v9](research/first_principles_robot_learning_ledger_v9.md), [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
