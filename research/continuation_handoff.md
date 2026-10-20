# Active continuation after v11

The persistent Goal is ACTIVE. Latest verified scientific ledger v11 embeds full
v10/all previous history verbatim. Prior v10 publication was450272cc1f34282b7dd2521db3972280982c2c7e.
Verify v11 Git/SSH origin/main publication status rather than inferring from this
file. Read the new v11 section and independent audits before extending.

## Verified v11 correction and findings

In the slow linear motor case,16x more nested physical training (512→8192) barely
changes control. Reducing position-sensor SD .2→.02 helps but cannot remove the
latest hidden motor innovation. Known-parameter conditional covariance yields an
information floor; exact positions still leave xi68 unseen at action69, and the
future xi69 also matters during16 forcedzero transitions.

Crucial correction: those true information bounds did not explain most original
endpoint error. The executor knew ACTIVE70, but the fixed8-action planner was
never passed remaining availability. It optimized unavailable future controls.
Known-KF mean endpointbias²1.676571 vs variance.343423 at SD.2 (finite24-history
averages). Deadline-aware N=min(8,70-t), with same16plannedzero tail for both
comparators, reducesbias²to.001556 and latepositionRMSE .652→.331. At SD.02,
bias² .603351→.000664 and lateRMSE .482→.231; strict whole-stop counts3/24→6/24.
More/better modellearning was not the needed repair; passing existing task
constraints through the planning interface was. No autonomous failure diagnosis,
new physical information, general learning theorem or optimal stopping policy.

All nested data and336 new episodes/23520 plans independently reconstructed;
all prior v10 pilot/errors and first sensor-focused interpretation preserved.
Sources/results: precision_information_v11, deadline_control_v11; audits:
audit_precision_information_v11, audit_precision_cutoff_v11,
audit_deadline_control_v11, v11_theory_independent.md.

## Immediate next research cycle: question whether model learning is necessary

Do not continue accumulating scalar covariance formulas. A genuinely competing
foundation is direct improvement of goal-directed behavior from experience,
without fitting a physical/predictive law. Conventional policy search is a
legitimate baseline, not novelty. Compare it fairly against the now-correct
learned predictive controller, or redirect to coupled dynamics if a simpler
discriminator shows no consequential uncertainty.

Promising first derivation (not yet independently audited): in the LAGGED model,
v[t+1] depends on commands only through u[t-1] and earlier, hence it is independent
of fresh sensorerrors e[t],e[t+1] under observation-adaptive policies. Therefore
observed stage loss (o[t+1]-g[t])²+.3*(o[t+1]-o[t])²+.03*u[t]² differs in EXPECTATION
from true-position/velocity stage loss by1.6*sensorVariance, constant acrosspolicies
for the same goal schedule. This could train a direct controller using ONLY
actual observations/goals/commands, no privileged true-state reward. In motor-first
timing feedback can invalidate the difference-square correction; derive a
counterexample before generalizing. Noise scales need not be known just to rank
expectations, but finite-data fitness variance and differing goal normalizations
must be handled honestly.

Next concrete work:

1. Independently derive/falsify that observation-loss claim under precise lag,
   feedback, process/sensor-noise and goal-access assumptions. Save any counterexample.
2. Design an actual experience-learning comparison: conventional finite-history
   direct controller/policy search versus standard OE/filter/MPC using the same
   observation channel, action bounds and actual interaction counts. Supply no
   true-state reward to direct learner. Both inductive biases/tuning compute must
   be explicit; no modern-RL superiority claim from a weak handpicked baseline.
3. Reconsider the artificial forcedzero cutoff. Ordinary station-keeping with
   continuing feedback is an equally legitimate reach-and-stop objective; keep
   task/executor/planner action availability consistent. Do not confound improved
   performance with extra intervention opportunities or shutdown-noise changes.
4. Test new target sequences/distances and unfamiliar physical parameters. Include
   strong simple feedback and observable-history alternatives. If established
   simple mechanisms suffice, accept that and escalate coupling only for a
   specific remaining learning uncertainty, not to force a complex method to win.
5. Audit independently; publish meaningful v12 with fullv11, source/results/audits,
   commit/push existingSSH andverify. Continue immediately afterward. The broader
   objective remains unresolved, so keep Goal active across releases.

Owner PC, canonical repo, existing .venv only; no external blocker/approval needed.
