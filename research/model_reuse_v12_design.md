# V12 prescribed diagnostic: reuse consequences within the SAME controller family

Written before inspecting corrected shared-comparison outcomes. This is a mechanism
diagnostic, not an independently tuned performance contest. The question is whether
an apparent benefit of OE/EKF/MPC requires the richer MPCcontroller, or whether
predictive experience reuse can improve the SAME sixparameter stationarycontroller.

For each of the32 corrected shared-data runs, use the nonlinear OE model fitted at
33024 physicalsteps. Search the identical sixparameter controller with the identical
8generation/24candidate/2task CEM protocol (33024 VIRTUAL steps), on that fitted
model with sensorSD.2, zero train motor noise. Fresh disjoint virtualCEM/episode
seeds base+10000000000, using the same disjointrole offsets. No additional physical
training; current goals only, same supplied bounds/resets. Final winner is evaluated
on the actual plant under the EXACT shareddata heldout streams. Holdout metrics
may not select gains, virtualnoise, modelclass, seed or CEMsettings. This adds5504
physical teststeps and64 resets, not trainingdata, across32runs.

Report both original directcost and model-reused directcost paired bytrainingseed,
plus MPCdiagnostic. Same family and stationarymetadata isolates family/timefeatures
from the path to newparameterchoices, but not compute: modellearner uses32k extra
simulatedsteps and a supplied physicalfunctionfamily. It can use its own predictor
as a simulator; directCEM lacks that simulator. No optimality/globalmethodclaim.
Modelbias/selectionnoise and limitedvirtualsearch may still defeatreuse.

Also compute first70step costs from all existingtesttrajectories. MPC uses itsfull
16action horizon through t70 inclusive, so its shortened endhorizon cannot affect
that prefixcausally. This removes one possible clockexplanation of prefixgaps,
without changing trajectories or retrospectively choosing a newprimarymetric.
