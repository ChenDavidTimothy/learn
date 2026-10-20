# First moving-history predictive pilot: weak acquired action effects

2026-10-20. **TESTED, INDEPENDENTLY RECONSTRUCTED for forks:** the frozen first
corpus and selected conventional predictors produce the results below. This is
an engineering negative, not a v17 learning release. Actual control has not yet
run. The general-learning Goal remains active. Do not change this pilot's models,
goals, candidate selection or planner to rescue the heldout outcome.

## What was learned and selected

Twenty-four continuous training trajectories supply2,880 overlapping renewal
rows. Four nonlinear candidates xthree whole-trajectory bootstrap members and
four retained-local candidates use ONLY noisy Cartesian histories/commands.
No exact preparation, joint state or fixed response law reaches ordinary fitting.
Eight complete validation trajectories select by six-block member forecast MSE.
All training losses, candidate validation profiles and parameters are retained.

Selection: nonlinear width128/Adamweight_decay.01; local512neighbors/ridge.001.
Their member-averaged validation coordinate MSEs are.00732309/.00702159m² over
1.2sec open-loop paths. On eight actual test trajectories, ensemble-mean
one-block(twofuturecamera) coordinate MSEs are4.40934e-5/3.33068e-5m²; six-block
mean-path MSEs.00602836/.00623331m². These absolute errors mix moving drift,
observation noise, action response, finite-history ambiguity and fitting bias.
They do not establish accurate intervention knowledge or calibrated uncertainty.
Independent complete fit-record verification now PASSES: allcandidatevalidation
and selectedtestorigins, train-onlyscales, bootstrapIDs and savedparameters
reconstruct (NNmax1.17e-15, local1.96e-13). All12initialseededlosses check;
intermediateAdamstates were not independentlyreplayed. See independentfitreport.

## Decisive external contrast diagnostic

At32 frozen heldout prefixes (fourtimes in each of eighttesttrajectories), the
evaluator independently forks the same actual hidden state under five commands:
zero and ±.4along eachtorqueaxis, heldtwo .1secintervals. Ordinary predictors see
only their existing noisy prefix and the proposed command. Forkmeans/states are
never training/selection inputs. The320privilegedflows/32seconds and160state
preparations are external diagnostic resources, not ordinary acquired evidence.

Across the four nonzero commands and bothfuture times, always predicting ZERO
consequence contrast gives coordinate MSE1.18886287268e-5m². Nonlinear selected
mean contrast MSE is1.15283992646e-5, ratio.96969966255; local1.13699778698e-5,
ratio.95637420691. These are descriptive values, not statistically established
population improvement. Prefixes/times share eight physicaltesttrajectories.

True average contrast norms at .1/.2sec are.00151169/.00605644m. Nonlinear
forecasts give.00034004/.00099987m; local.00063559/.00251732m. Both substantially
understate the typical command-induced change. A small improvement over zero
contrast is not zero learning, but does not support a claim that these selected
predictors have acquired strong individual-context action consequences.

Independent audit reconstructs forecasts through NumPy tanh matrices and a
separate augmented-SVD weighted local regression, not canonical prediction calls.
Maximum forecast differences4.44e-16(nonlin)/2.38e-14(local). All320physicalflows
replay in independent Hamiltonian coordinates;66Radauintervalchecks agree within
3.41e-13state. Full artifact hashes, scales/bootstrap/selection metadata, every
contrast and summary/resource count pass. This weakness is not an observed
dataflow or fork-simulation bug. See audit_cartesian_stream_v17_forks*.

## What it means, and what it does not

The moving instrument proved that passive motion can distinguish two known
alternatives. It never proved that a generic finite-window learner would acquire
their useful response from this finite random corpus. The new negative supplies
that missing check. The actual corpus crosses folded/extended poses; treating
one episode as one persistent elbow is false. Local/finite-memory smoothness and
neural regularization remain supplied approximations.

Several explanations remain: conditional averaging of insufficient histories,
sparse contextual intervention support, response shrinkage while fitting larger
drift, finite model/optimization bias, or a mixture. Strong alternative methods
or different honest acquisition may suffice. This pilot does not establish a
fundamental architecture limitation, mandatory latent modes, mandatory probing,
failure of conventional system identification or superiority of a new method.
Three fit-bootstrap members do not automatically preserve intrinsic ambiguity.

The frozen actual control pilot is still needed: weak individual predictions
might nevertheless support adequate frequent feedback. Conversely, no claim of
useful acquired mechanism follows from forecast tables if decisions fail. Keep
the already specified modest goals, bounded optimizer and exact-state nominal
authority diagnostic. Preserve failed cases and independently audit them.

## Prospective alternative, not a test-tuned rescue

An elementary competing formulation follows from command randomization. If the
conditional mean has the restricted form E[Y|history,u]=d(history)+G(history)u,
newcommands have known zero mean/covariance and are independent of fullhistory,
then G=E[(Y-dhat(history))u^T|history]Cov(u)^-1 for any fixed/pastmeasurable dhat.
This separates an action-dependent projection from moving drift. It is a
varying-coefficient/instrumental-moment possibility, not proof of a new mechanism;
finite-duration physics need not be exactly affine in u and finite histories
need not transfer across changed policies. Cross-fitting/ordinary validation
and actual variance/support would matter. The independent theory auditor is
challenging this framing before any new learner/protocol. No forktruth can tune
the existing or future method. Do not start another architecture ladder merely
because this first conventional construction is imperfect.
