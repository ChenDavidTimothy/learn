# Moving Cartesian control pilot: local sufficiency and unresolved attribution

2026-10-20. **TESTED / INDEPENDENTLY RECONSTRUCTED**, one acquired corpus and four
fresh physical preparations with two paired camera paths each. No v17ledger or
general-learning breakthrough is claimed. Sources/designs/outcomes unchanged
from the frozen first pilot. The active research Goal remains unresolved.

## Actual result

| Mechanism | Mean18sec task cost | Sustained precise goals /24 | Precise goals by phase |
| --- | ---: | ---: | --- |
| Privileged exact-state nominal reference | .01558579 | 24 | 8,8,8 |
| Frozen nonlinear history predictor + common planner | 29.15144107 | 0 | 0,0,0 |
| Retained local predictor + common planner | .78768159 | 22 | 6,8,8 |

Cost integrates true endpoint error²+.25Cartesian speed²+.01torque² over all
three6secgoals. Warmup motion/effort and all physicalseconds are separately
retained; no unannouncedgoal is scored during warmup. Precision requires BOTH
coordinatewise positionerror<=.03m AND trueCartesianvelocity<=.03m/s at EVERY
camera sample in the final1second. It is not an endpoint-only test. Reference
uses exactstate/unitbodyinversekinematics and computedtorque, so it is an
authority diagnostic, not an equally informed learned baseline or an optimized
oracle. No referencegoal needed clipping; two witnessdecisions saturated.

Local mean phasecosts are.75935617,.02126210,.00706331. Two first-goalfailures
are the two camera paths at the SAME physical context2: total18sec costs
3.10711604/3.07658966, followed by precise second AND third goals without
physical, library or planner resets. The other six totalcosts range.01063–.03550.
Keep both failures. Eight paths are clustered within fourpreparations, not eight
independent bodies or an independent-corpus performance estimate.

## What this resolves

A simple conventional local camera/action relation can support useful sustained
feedback under unprepared moving conditions, without joint encoders, known rest,
truecurrentvelocity, an episodebranch label or a supplied fixed-B propagation law.
Each realcommand is heldtwointervals; all camera/action history is ordinary.
The local library starts from the same2,880bootstrapped trainingrows per member,
refits by current context and appends90completed realrows per18seccontrol episode.
History/library/body/plan persist through goal changes. Its generic finite-memory,
linear-extrapolation and local-affine assumptions remain supplied priors.

This partly extends v15's history/control construction to a noninjective sensor.
It does not prove the chosen summary is sufficient generally, identify a physical
latent state, demonstrate transfer to another body, or compete with a complete
modern robot-learning benchmark. Ordinary retained estimation remains a successful
explanation. No special mode architecture/probe has been established necessary.

The weak offline command-contrast findings remain valid. Good frequent feedback
may need less than accurate prediction of every individual-world command fork,
and actual new experience can revise a local relation. Conversely the positive
control score does not retroactively make those initial contrasts accurate.
The present comparison does NOT isolate acquired-library reuse from onlineappend,
feedback, locality, fit class or optimizer behavior. That motivates the next
minimal paired frozen-library ablation, not another neural architecture menu.

## Numerical and mechanistic audit

All1,440ordinarydecisions and2,160online localmemberfits reconstruct independently
through NumPy networkrollouts and augmentedleast-squares localfits, including
every candidate/selected modeledcost, selectedgradient/projectedgradient, first
issuedcommand, currentgoal, shiftedplan, retainedlibrary and exactappendcount.
Maxcostdifference6.68e-13; gradient/PG2.50e-13; localcoefficient1.60e-11. All
21,265 modeled value/gradient calls reconcile. No optimizerfailureflag/fallback
occurred, but SUCCESS IS NOT A STATIONARITY CERTIFICATE.

Nonlinear projectedgradient median1.76e-6,99thpercentile2.10e-4,max.147447;
five decisions>.001,two>.01. The maximum is case7/renewal63/time142 and stops on
relativefunctionreduction. Local median5.90e-7,max6.02e-5. Don't attribute the
nonlinear physical failures solely to a missing representation or call its
nonconvex plans globally solved. Under frozen localcoefficients, every future
feature/output is affine in the plan. Its quadratic loss is strictlyconvex from
positive effort; that mathematical convenience is part of this complete
mechanism, not evidence its model is physically correct.

Independent Hamiltonian checks cover ALL4,704plantintervals and ALL24fullrolling
trajectories (8references+16ordinary), with maxstate1.28e-10/Cartesian8.55e-11
for wholetrajectories.124Radauintervalchecks max5.13e-12. All720reference
IK/computedtorque decisions, RNGroles, pairednoises/warmup/goals, scores/sustained
windows and counters reconstruct. No canonical source was used as the physical
reference. See separate controlordinary and controlphysics reports/executables.

Ordinary16episodes cost3,136intervals/313.6simsec including25.6secwarmup,3,152
camera-read opportunities and6,304scalarnoise opportunities; eightpairedpaths
contain3,152distinctscalarRNGdraws. References add1,568privilegedintervals/
156.8sec. Root invocation walltime181.30sec, reported separately from simulated
execution. Planninglatency is not physicallycharged; no realtime claim.

## Next falsifier and publication boundary

Run onlyeight additional frozen-acquired-library episodes from the same paired
preparations/noises/goals. Disable realrowappend; retain currentcontextrefitting,
feedback and the commonplanner. If this matches, append necessity is unsupported;
if it improves, accept possible harm without test-tuning retention. This isolates
the totalclosed-loopeffect of appending, not an individual coefficient effect.
Then freeze meaningful fresh independent acquired corpora/controlcontexts for
both local variants. The unchanged nonlinear pilot is a limited reference, not
the sole basis for claiming superiority. Do not publish a newledger merely for
this one-corpus success or another loss identity. Preserve verified progress
and continue the broader investigation.
