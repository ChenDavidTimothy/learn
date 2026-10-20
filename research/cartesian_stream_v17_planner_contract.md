# Precise first control pilot contract — before forecast/fork/control outcomes

2026-10-20. Companion to the frozen learning design. Ordinary fitting was launched; no candidate validation score, heldout fork or
control outcome had been read while writing this contract. Its creation was
batched in the same tool invocation that first returned completed fit scores,
so the fit had finished before this file reached disk; its scores did not inform
the choices written here. This freezes the remaining planner, RNG and nominal authority details.
No learning architecture/goal revision follows from test results.

## Common planner

Six two-interval blocks (12future camera samples), three coherent fit-member mean
paths. Freeze each member throughout a hypothetical plan; freeze local fit/query
at the current actual renewal. For every sample, cost squaredCartesian target
error +.25*squaredcamera-difference velocity (firstdifference uses actual current
noisy reading). Add .01*sumu² for EVERYphysicalinterval (two perblock). Average
over members. This is a deterministic approximate planner, not full conditional
Bayesian control or a guarantee of accurate physical risk. No future reading is
assimilated inside a plan. Goal is only the currently announced target.

Initial plan is whichever has lower finite modeled cost: shifted previously
selected plan (zero finalblock) or the allzero plan. Apply SciPy L-BFGS-B with
analytic Torch gradients, bounds[-.5,.5] on12coordinates, maxiter40,maxfun80,
maxls20,ftol1e-9,gtol1e-6. Accept its final finite feasible plan only if cost no
larger than the selected initial competitor within1e-12; otherwise use the
competitor. A termination failure may still yield a lower finite plan; record it
as a failure rather than pretending optimality. Save candidates/selectedcost,
plan, nfev/njev/nit/status/message, fallback and selected projectedgradient
x-clip(x-gradient,-.5,.5). One actual command is held for bothnextintervals;
bothcamera readings then become available, the local library appends exactly
one completed real causal row, and the next renewal replans. No goal resets.

## Independent randomness and goals

Four evaluator initial preparations use the acquisition distribution, with paths
[20261020,17,670,0,context_id,0], context_id0..3. The same preparation is shared
between methods and two noise repeats; no exact preparation value is ordinary.
Warmup command role[20261020,17,670,0,context_id,2] draws four uniform[-.5,.5]^2
commands heldtwo after eightzero intervals; samewarmup across paired repeats and
methods. The16interval prefix is charged1.6sec. Each noise path is
[20261020,17,670,0,context_id,1,noise_repeat], repeat0/1, and draws197x2independent
N(0,.0002²) values. This covers16warmup+180controlintervals+initialread. Pair the
same path across methods; opportunity counts and distinct draws are separate.

At warmup end defineg0=.95*ordinarynoisyreading. Announceg0, R(.04)g0, R(-.04)g0,
each6seconds/60intervals/30decisions. Do not reveal latergoals to decisions. Body,
history and retained local library continue. Local library starts from the same
selected acquired bootstrap copies separately per methodepisode, not previous
testepisodes. No online neuralweight updates. Preserve all actual states, issued
commands, noisyreads, currentgoals, local fit summaries/coefficients and plans.

## Privileged nominal authority witness

Before ordinary control, execute the same initialpreparation, warmup and
sensor-generatedgoals using an EXTERNAL exactstate bounded reference, not a
learned competitor. At each .2renewal, compute a jointtarget for the current
Cartesiangoal using unitlinkinversekinematics, clippedgoalradius to[0,2] solely
for this evaluator computation. If clippingchangesgoal, record that fact and
score against the ORIGINAL goal. Choose the inversebranch/2piroundings with
smallest jointdistance to currentq (evaluator only). Apply computedtorque
u=clip(M(q)*[-(q-qtarget)-2*v]+h(q,v),-.5,.5), heldtwo. Gains1/2arefixed, not
tuned on success. This gives an actual diagnostic of nominal bounded authority;
failure of this particular reference is not proof of infeasibility. No witness
action, inverse, q/v/M/h or outcome enters fitting/selection/ordinary decisions.
Save all failures and do not changegoals to rescue a method.

## Scoring and costs

At actualsample endpoints compute integratedtrueCartesianerror+.25*trueCartesian
speed²+.01*issuedtorque² timesdt. Report goalperiod cost, warmup effort/motion
cost (no goal was announced), and combinedchargedtime/effort. Sustainedprecision:
eachcoordinate endpoint error AND trueCartesianvelocity <=.03 at ALL10samples in
the last1secondofeachgoal. Saveperphase profiles/effort/failedwindows.

Ordinary pilot:16method episodes,3goals=48phasechecks,1,440decisions,3,136flows/
313.6simsec including256warmupintervals. Thereare3,152camera-read opportunities/
6,304scalarnoise opportunities, but eightdistinctpairedpaths use3,152actual
scalarRNGdraws. This is fourphysicalpreparations and oneacquiredcorpus, not16
independentbodies. Eightnominalwitnessepisodes add1,568privilegedflows/156.8sec,
withpairedalreadygeneratedsensorpaths; noextraordinarytrainingexperience.
Controlplanningwalltime/calls are separate; physicalexecutiondoesnotwaitoutthat
latency in this simulation. No realtime or modernbenchmarkclaim.
