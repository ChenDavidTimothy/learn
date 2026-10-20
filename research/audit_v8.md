# Independent v8 audits, 2026-10-20

Auditors: independent_theory and v5_adversarial_audit. Neither edited canonical artifacts.

## Mathematics

- Full IO-law equivalence proved by eliminating hidden motor force and coupling arbitrary observation-adaptive policy random coins/noises. Not limited to linear policy, Gaussian probes, zero commands or response moments.
- Scope checked: zero or compatible initial conditions, noise entering motor state, independent output sensing. Known mechanical noise, motor-reset preparation at nonzero velocity, component priors or force/current measurements can break equivalence.
- Apparent one-state coefficient is S=rho+lambda. AR1 fitted moments cannot expose omitted lag because selected instruments are orthogonal to Dprev. Unused lag2 residual moment is -rho*lambda*beta=-.312.
- AR2 position-noise residual is MA3; chosen sequential probe IVs remain valid, full relevance rank for beta!=0.
- Calibrated added motor-force channel identifies physicalrho. Force disconnection differs from zero command.
- Algorithm-independent physicalrho MSE bound .015625 (root-MSE .125) independently derived for the two equivalent worlds.
- Component transfer before new observations has tight expected-RMS lower bound half of the two response distance; old motor pole must not be revealed in replacement metadata.

## Final artifact verification

- Final source rerun including component_transfer is BYTE IDENTICAL to canonical JSON (/tmp/actuator_memory_v8_audit_final.json).
- Independent direct AR2 feedback recurrence reconstructs Gaussian/atom measured displacements, commands and force measurements within1.4e-15.
- Separate cascade-convolution impulse response matches forecast.
- All means,MC SE,RMSE,real-root subsets and forecasts independently recomputed.
- Gaussian and atomSD.5 real roots48/48; atomSD1.5 only34/48. Conditional mean pole errors disclose missing14; all48 retained for coefficient/forecast analysis.
- Coupled observable path differences <=3.56e-15; hidden force differs.
- Independent motor-replacement recurrence yields response RMS distance .37167470652234286 and minimax expected-RMS lower bound .18583735326117143 (expected-MSE bound .034535521867117426).
- Forecast uses known zero velocity/motor initial states, clean output,20-step open-loop inputs; not unknown-state filtering or control performance.
- Rechecked final v7 pruned output is byte-identical to its frozen-source audit output.

## Interpretation

This is classical cascade/realization ambiguity plus a concrete counterexample to assuming zero command means zero physical actuation. No novelty or modern-method superiority. Accurate IO prediction remains possible; physical component labels and component-transfer predictions need added information. No unresolved blocking disagreement remains.
