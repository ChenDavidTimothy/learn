# Independent innovation-control result check

2026-10-21. **PASS within this bounded scope.** The independently authored
`check_innovation_control_results.py` reconstructs declared costs and sustained
precision directly from saved q/v states, issued actions and goals. It imports
no environment/scoring helpers and performs no physical integration, neural
forward pass, model fit or PPO replay. The earlier synthetic math/GRU fixture
result `check_innovation_control_results.json` is preserved; this result is
`check_innovation_control_screen_results.json`.

Checked all 26 archives: 16 learned-policy validation checkpoints, two RLS
validation settings and eight fresh test method/split archives. Each archive
contains 32 bodies and 240 transitions: 199,680 scored transitions. Joint errors
are independently wrapped using atan2(sin(error),cos(error)). The per-step cost
uses post-transition state and the goal active when the action was issued.
Precision checks both coordinate errors and speeds at every saved endpoint
61–80, 141–160 and 221–240. Maximum cost discrepancy is 3.55e-15; all precision
flags agree exactly. No recorded discrepancy affects the conclusion.

Hidden body/actuator metadata, initial states and goals agree exactly between
all rivals and ablations within each validation/test split. Such privileged
metadata is used only for this pairing audit. The checker verifies all source
hashes, fit-report hashes, all 16 checkpoint weight hashes/finite states/parameter
counts, validation-only checkpoint and learning-rate selection, RLS forgetting
selection, all 128 recorded updates per fit and declared interaction counts.
It checks training-log arithmetic and finite losses, without reconstructing PPO
updates or independently regenerating actions.

The four fits account for 7,864,320 training interactions; validation adds
138,240 and fresh evaluation 61,440, totaling 8,064,000 transitions or 403,200
simulated seconds. These include repeated use of paired bodies/noise opportunities
and are not counts of independent robots. Training and evaluation wall time
remain separate from simulated interaction time.

All frozen continuation gates fail on both ID and OOD. The moment candidate has
cost 0.860764 and 2/96 precise ID goals, versus RLS 0.266882 and 42/96. OOD gives
0.901111 and 0/96, versus RLS 0.289021 and 33/96. The GRU also has only 2/96 ID
and 0/96 OOD precision. Zeroing the candidate's moment has essentially unchanged
cost (ID 0.860743, OOD 0.901162), rather than the required 10% benefit. All 687
failed test goal windows remain indexed in the machine-readable result; none is
dropped. RLS performs better but also fails the candidate's absolute competence
thresholds; no general task-solving claim follows for it.

The supported decision is **reject/retire this fixed formulation at the stated
budget**. Failure does not prove every correlation-memory or meta-RL mechanism
incapable. No additional fit, physics replay, test-informed feature change or
optimization ladder is recommended by this check. Earlier bounded timing,
likelihood, GRU and instrument fixtures remain separate evidence; this scoring
check does not expand their claims.
