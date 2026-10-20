# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v7](research/first_principles_robot_learning_ledger_v7.md), self-contained with all prior ledgers preserved verbatim.

V7 derives conditional intervention weights that isolate a passive coefficient under unknown nonlinear gain, proves a precise limit of continuous-probe weighting, and tests randomized zero commands as a repair. Stronger conventional IV comparisons prevent a superiority claim. V6 studies context/response ambiguity, v5 hidden nonlinear coupling, and v4 corrects an unstable simulator. **No claim of universal learning or robotic-humanoid performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/feedback_virtual_zero_v7.py > /tmp/feedback_virtual_zero_v7_check.json
cmp research/feedback_virtual_zero_v7_results.json /tmp/feedback_virtual_zero_v7_check.json
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
