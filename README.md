# First-principles robot learning

Research history, reproducible car simulations, corrections, competing explanations, and open mathematical questions are kept in versioned ledgers under [research](research/).

**Current:** [Living Research Ledger v2](research/first_principles_robot_learning_ledger_v2.md) (self-contained, including the complete original v1 history).

**Run v2 on this machine:**
~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/car_learning_benchmark_v2.py > research/car_learning_benchmark_v2_results.json
~~~

Python 3.12.3, NumPy 2.5.3 used for recorded results. The existing local `.venv` is not committed. The v1 benchmark, results, and ledger are preserved unchanged. v2 results and the ledger are committed in the same research revision. New research releases increment the ledger filename rather than overwriting earlier versions.

**Scope:** Synthetic 1D semi-implicit car, output-only observations, generated algebraic features, noisy measurement identification, and action excitation. No claim of a general intelligent learner or validated real-robot transfer.
