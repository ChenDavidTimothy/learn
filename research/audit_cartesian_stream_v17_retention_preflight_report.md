# Independent pre-execution retention-ablation audit

2026-10-20. **PASS** for source/design and no-physics fixture. Exact hashes and
actual check results are in `audit_cartesian_stream_v17_retention_preflight_results.json`;
reproduce with `.venv/bin/python research/audit_cartesian_stream_v17_retention_preflight.py`.
The source hash is `94e3e5d7ac4e415aa6caecbe1c8042d1ef6a14d63eb5e6fa4a4eb46b4ca0f598`.

The distinct `local_frozen_library` tag selects the parent's existing local
preparation/affine-path branch. Its only mathematical intervention is an explicit
no-op at each completed real pair. Actual current camera/action history, refits,
selected planner, goals, paired cases/noise and physical driver stay the same.
Fresh selected acquired libraries are loaded per episode. The wrapper saves
coefficients/queries itself because the parent's exact-`local` tag provenance
branch does not apply, and reports constant library counts/90 no-op notifications
instead of falsely claiming row additions. Partial failures preserve actual
streams and coefficient captures.

Using a real ordinary validation prefix, separate selected-local instances
produce EXACTLY identical query/coefficients, coherent member paths, modeled
loss and command gradients before additions. One completed-row notification
leaves all frozen arrays unchanged at2,880 rows/member; the separate online
instance grows to2,881. Re-preparing the unchanged query reproduces coefficients
exactly and coefficient/query capture counts are correct. The fixture replaces
physical flow with a raising guard; no optimizer, witness or physical episode
was executed.

No blocking implementation mismatch found. The ablation retains learned data,
contextual refitting and feedback; it is not a scratch/no-learning/no-adaptation
baseline. Actual720 decisions, coefficient libraries, paired trajectory streams,
scores and optimizer residuals still require independent artifact audit. A
paired closed-loop difference measures the total effect of appending experience
under this protocol, not an isolated per-row causal parameter gain.
