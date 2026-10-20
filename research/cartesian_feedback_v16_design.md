# Persistent Cartesian learning/control construction pilot

Frozen before feedback outcomes. The prior batch experiment is NOT sustained
feedback. This extension tests whether acquired response uncertainty can be used
and retained through actual Cartesian goal changes without joint encoders or resets.
Latest verified ledger remainsv15; no milestone for an aliasing-only calculation.

Use exactly the384group acquired corpus/selected conventional priors from the
corrected adaptive batch run. No new acquisition, branch labels, trueB or joint
state enters ordinary fitting/action. SameT=.1, SD=.0002, bound±.5, actual continuous
nonlinear arm, initial hidden elbow/rest and openly supplied Cartesianp0. Local
constant-response model remains an explicitly supplied approximation, not learned
universal physics. No supervision for physical latent configurations.

Model family: normalized joint uncertain quantity z=[s,w,vec_row(B)], where
s=(p-p0)/T² and w=v_cartesian/T. Known issuedu gives linear transition
s'=s+w+.5Bu,w'=w+Bu,B'=B. Observe s through normalized camera noiseR. Under the
constant-B assumption a full joint Gaussian filter is exact; mixtures update each
Gaussian component and component weights. Preserve all state/B correlations.
Physical nonlinear dynamics need not satisfy this local model indefinitely.
No additional process/drift noise is inserted in this first bounded local test.

For the learned mixture, initialize within-component coefficient covariance as
I_output tensor (R*Gk^-1), where Gk is soft-assignment-weighted calibration input
Gram. This is a local plug-in curvature approximation, not exact empirical-Bayes
uncertainty with unknown assignments/model error. GeneralGaussian uses the bounded
likelihood-refined covariance. MatchedGaussian uses COMPLETE initial mixture
first/two moments, including within-component covariance. Initials/w exactly0.
Because adding curvature uncertainty changes the predictive law, re-evaluate ALL17
already fitted mixture candidates on the SAME validation groups using that actual
Gaussian-component predictive likelihood, and select there before feedback. This
does not use feedback outcomes, true labels or additional fitting/physical data.

Every goal lasts3pulses. Camera is read after firstpulse; the last2commands are then
committed. A terminal camera read updates the NEXT batch's belief after both
commands have executed. The next goal is disclosed only then. Thus2camera reads
per3pulse goal, no hidden future goal and no physical/controller/parameter reset.
This is sustained two-stage-per-goal feedback, not arbitrary per-pulse feedback.
The last2commands minimize expected terminal position/velocity/effort loss under
the correlated posterior, with exact boxQP/global face-enumeration checks.
Constants containing state uncertainty are retained when comparing firstactions.

Compare learned mixture/generalGaussian/matchedGaussian each with informed first
action versus ordinary cautious3pulse firstaction and posterior update; add generic
random direction for mixture/generalGaussian. All8methods share acquired data,
clocks, sensor opportunities, objective and command bounds. First-action candidates
are0, cautious action, and radii.05/.1 in8uniform torque directions; include plus/
minus the leading learned coefficient-variation eigenvector at both radii. No
physical radial/tangential direction is supplied. Random direction draws from8
uniform directions, radius chosen by direction-averaged expected continuation
cost. Charge allactual effort/time; do not claim identicalenergy.

Anticipated first-observation laws are integrated adaptively in2Dstandardnormal
coordinates, Genz-Malik, atol1e-6,rtol1e-5, max2000subdivisions, domain[-7,7]^2.
Record estimated errors, analytic Gaussian-tail upperbounds using feasiblezero
continuation, and convergence. This corrects frozen coarseGH9failures; independent
checks remain required. First-action search is a bounded candidate heuristic,
not a globally optimal dual controller or a reproduction of a published algorithm.

Two six-goal sequences, offsets multiplied byL=.001meter:
A:(1,0),(1,1),(0,1),(-1,1),(-1,0),(0,0).
B:(0,1),(1,1),(1,0),(0,0),(-1,0),(-1,1).
Onlythecurrentgoal is given. Adjacentchanges are one coordinate step; no impossible
instantaneous elbow switch. The second sequence first permits a useful motion
without resolving every hidden distinction. Prior/posterior knowledge is retained
when later goals need it. Every phase scores normalized terminalposition²+.25*
normalizedCartesianvelocity²+.01*all3command², as in the batch experiment; record
all intermediate motions separately. Precision .0003m/.003m/sec percoordinate at
each phaseendpoint, not sustainedwithinphase stationkeeping.

Engineering pilot uses one acquiredrep0, both hiddenbranches, bothsequences and
2freshnoise repeats, all8methods. SeedSequence [20261020,16,640,rep,4,sequence*100+repeat]
paired acrossmethods/branches;12independentcamera2Dnoises persequence/repeat.
Theseare4pairedsensorpaths, not64independentlearnedbodies. Preserve parenthashes,
allstates/actions/observations/beliefs/solverand integrationrecords. Save partial
completed episodes before expensive work so numericalfailures cannot disappear.

Privileged exact nonlinear witnesses start each branch and follow both sequences
with knownphysics, zero firstaction and optimized remainingcommands. They certify
those nominal trajectories' feasibility only, not arbitrary failed recovery states.
Any local postprobe stopping-authority diagnostics use truth externally and never
select ordinary actions. No precision success or branch classification alone proves
general learning. If ordinaryGaussian adaptation catches up afterinitial evidence,
retain it: a useful prior shape may matter transiently without being universally
necessary. No architectureornewscientificprinciple is imposed by this experiment.
