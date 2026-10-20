# V13 bounded diagnostic: capacity versus finite policy search

Prescribed after inspecting v12 outcomes. Exploratory method design informed by
those outcomes; evaluation uses NEW streams. One diagnostic, then leave scalar
controller optimization regardless of result. Do not replace failures by another
optimizer ladder or claim that a local failure proves family incapacity.

Use each of the32 learned nonlinear OE models at the final v12 checkpoint, and
exactly its six-coordinate stationary clipped controller, parameter bounds, goal
tasks, sensor SD .2, process noise zero during virtual training. No new plant data
for fitting. Optimize the model's position/velocity/action loss on a fixed bank of
16 virtual episodes (eight draws per each of the two training goals), resetting
all memories/states per episode. This objective uses predicted model states,
not hidden actual plant states. It is a model-using hybrid. Reusing virtual random
numbers is an optimization device; only fresh validation/test support transfer.

Verified pathwise derivatives through smoothing, the actual issued clipped
commands (derivative0 outside bounds,1 inside), motor/body dynamics and squared
residuals; no straight-through surrogate. Coordinates are logs/logit as in v12.
Three FIXED starts: original physical-CEM winner, virtual-CEM winner, generic
(.08,.6,.6,0,0,0). Bounded least_squares TRF max160 function evaluations/start,
ftol/xtol1e-9,gtol1e-8. Keep every attempt/status, objective, projected gradient,
selected coordinates and number of virtual steps; success is no global guarantee.

Choose among all three optimized candidates AND the two original winners by
mean model loss on a separate fixed16-episode virtual validation bank. Preserve
all validation scores, include unoptimized baselines so numerical fitting cannot
force a bad replacement on that validationbank. Validation is not plant testing;
selection on it is optimistic. No test scoring enters choice.

Disjoint RNG namespace: per v12 physicalseed plus20000000000 for virtual fitting,
plus30000000000 for virtual validation, plus40000000000+task for physical tests.
Fit/validation each use one fresh generator to draw16x87 errors. Fixed starts
share that model's banks. Physical tests run original physical-CEM, virtual-CEM,
gradient-selected and nonlinear MPC on exactly the same new stream, both heldout
goal amplitudes. Current goals only; keep MPC finitehorizon semantics, while all
three direct controllers stationary. Report paired metrics by8seed clusters/plant,
not as new independent model fits; modeltrainingdata are reused from v12.

A constructive lower-cost stationary controller in the SAME family would refute
that the family itself cannot attain such behavior. It would locate at least part
of earlier deficit in search/use of experience. Remaining deficits versus MPC may
still involve class, bounds, state uncertainty or planning. Failure leaves that
uncertainty OPEN. Neither outcome establishes model necessity, general robotics,
or a new learning principle. Preserve results and audit; then turn to a coupled or
unknown-function-vocabulary learning discriminator instead of optimizing this toy
further.
