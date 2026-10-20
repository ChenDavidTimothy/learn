# Active research continuation after v8

Latest verified release: v8, commit33be698bb90bf83c63c5d0328c55aa30c78d2ea1, verified on SSH origin/main. The persistent Goal remains ACTIVE. Do not mark complete from release count. Read v8 plus source/results/audit before continuing; all previous ledgers remain unchanged.

## Strongest current finding

The command-position interface cannot distinguish mechanical pole.65/actuatorpole.4 from the reverse allocation, under the declared linear cascade/noise/initialization. This is full adaptive-policy IO equivalence. Physicalrho estimation has minimax MSE>=.015625; a specified motor-replacement response has expected-RMSE lower bound.185837 before additional observations. Standard AR2-IV predicts well but cannot label the physical poles. Zero commands alone do not mean zero physical actuation. Force sensor/physical disconnection supply extra information.

## Investigation already begun (unreleased v9)

First competing formulation: impose velocity-independent linear actuator and nonlinear passive f(v)=rho*v-c*tanh(v),0<c<rho<1. With motor-first timing, elimination produces coefficients [rho+lambda,-rho*lambda,-c,+lambda*c,beta] on [v,vprev,tanhv,tanhvprev,u]. For c!=0 and full regressor rank, nonlinear ratio labelslambda; c0restoresv8ambiguity. Small c and small velocity range cause weak identification. Naive nonlinear IV on tanh(measuredD) is biased by errors-in-variables; do not reuse the linear IV proof.

Adversarial competing formulation: permit actuator velocity coupling (real motors can have back-EMF/feedback). With LAGGED timing vnext=f(v)+m,mnext=h(m,v,u)+xi, the invertible transform mp=m+phi(v),fp=f-phi, hp(mp,v,u)=h(mp-phi(v),v,u)+phi(fp(v)+mp) preserves all observed paths. For phi=-c*tanh(v), original nonlinear passive+linear actuator becomes linear passive+nonlinear velocity-coupled actuator, still TWO states, still input-affine and same additive motor noise. Nonlinearity alone does not label passive force without the modularity restriction.

The independent_theory subagent derived/audited both algebraic statements. These are classical realization/gauge facts, not novelty. Important distinction: motor-first formula versus lagged timing are separate declared models; never mix their indices.

`nonlinear_force_gauge_v9_pilot.py` executes the latter counterexample under nonlinear noisy feedback for8192steps, using no extra sensor. Its saved JSON is a PRELIMINARY illustration with independent artifact audit pending. It is not a released ledger or a completed comparison. Source/results were retained to avoid losing the active investigation.

## Exact next work

1. Independently rerun/audit v9pilot with a fresh adversarial implementation; check same commands/observations and mp=m-c*tanh(v), same noise placement, zero initialization.
2. Derive necessary supplied physical assumptions for distinguishing passive nonlinearity from coupled actuator dynamics. Attempt a counterexample with physically meaningful motor-current or force constraints; distinguish coordinate redefinition from genuinely distinct physical force channels.
3. Within restricted linear-actuator family, test noiseless nonlinear ratio recovery across independent seeds/cstrength/amplitudes; include regressor singular values and weak/non-real cases. Then add position noise and a proper observation-model/state-space comparator. Plain nonlinear IV is not automatically valid.
4. Test ordinary command-sequence transfer separately from component interventions/replacement. Do not claim all prediction/control impossible from force-label ambiguity.
5. If a meaningful verified milestone emerges, publish NEXT self-contained v9 embedding full v8, with full source/results/audits; commit/push SSH origin/main and verify. Continue immediately afterward.

Owner PC, existing .venv only. Worktree was clean after v8; only this handoff and unreleased v9pilot are intended continuation changes. No external blocker or user permission required. Do not overwrite released artifacts or change SSH remote.
