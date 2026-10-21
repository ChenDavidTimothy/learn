# New campaign: does acquired executable structure improve later learning?

2026-10-20. First proposal after closing the two failed post-v17 formulations.
No v18 is assumed. This is a scoped construction hypothesis, not a claim that
symbolic features, modular meta-learning or Bayesian adaptation are new.

## First-principles question and proposed operation

An encounter can change coefficients within an existing family, or change the
computations available to future learning. Neither is intrinsically superior.
Hypothesis: acquiring a small executable feature fragment from physical experience
can improve short-support learning on later bodies more than competent continued
neural learning with the same experiences. Neural features can also acquire
computations: no expressivity impossibility is asserted against them.

Construct a feature DAG from standardized observed inputs and constant 1. Generic
operations are multiplication, sin, cos, abs, and r(z)=1/(1+z*z). Propose bounded
fragments, including two-operation lookahead so a nonpredictive intermediate can
participate in a useful computation (e.g. an even intermediate in an odd effect).
Do not supply compound physical terms, physical parameters or equation labels.
Reject nonfinite/huge training features, normalize using training observations
only, semantically deduplicate on training observations, and retain selected
computations as persistent knowledge. These grammar and search restrictions are
substantial supplied priors, not learned universal arithmetic.

Select a new fragment by its mean disjoint-query squared prediction error AFTER
ridge coefficient fitting on a short support prefix/window. Use the exact Schur
complement rank-one update to score additions; it is an efficiency device, not
a new learning principle. Repeatedly searched episodes are training data.
After feature acquisition, learn a Gaussian coefficient prior/precision with the
same predictive-NLL ALPaCA core as the conventional feature learner. New-system
evidence updates coefficients; experience across systems changes available
computations and their prior. Record every retained operation and selection loss.

## Acquisition and later test

Reuse original 32 training and eight validation bodies/instrument, old train-only
input/output normalization, and q/v/action interface. Pretrain a 32-feature DAG
(nine initial atoms, 23 additions) and its Gaussian prior. Fresh acquisition has
eight new bodies, four new validation bodies. Their command law adds body-specific
cubic response, and damping adds quadratic velocity drag. This is a deliberately
specified structural change; multiplication/abs align well with it. A win is
conditional on that grammar and family. Finite observed delta-v/dt remains the
target, not oracle acceleration or an exact formula.

Only the simulator receives coefficients: alpha per actuator uniform [.5,1.5],
beta per joint uniform [.1,.3]; effective issued input is u+alpha*u^3 and effective
damping d+beta*abs(v). Hidden old physical parameters follow original ID ranges.
Use unchanged dt .05, eight trajectories/body, 96 intervals, actions held two
steps and RK4 eight substeps. Spot-check the changed integrator against DOP853;
do not launch a historical full-physics audit.

After acquisition, test 24 entirely fresh bodies: eight new-law ID, eight
new-law OOD using original out-of-range physical ranges, eight original-law ID
for retention. Freeze seed roles [20261020,18,40,split,role] before sampling:
split0 acquisition,1 new validation,2 new ID,3 new OOD,4 old ID;
role0 physical parameters,1 initial states,2 issued commands,3 new coefficients.
Original test bodies from either retired screen are not reused for a new claim.
New parameters remain evaluator-only. Training uses grouped bodies and exact
observations; this is not unknown-boundary or perception learning.

## Competitors and capacity

- Meta-grown program: append 16 fragments to the acquired 32-feature DAG.
- Fit-grown program: same initial DAG and 16 additions, grammar/search limits,
  but select by ordinary fitted group error on the SAME support/query union.
  This controls the claimed value of support-to-query structure selection.
- Frozen old DAG: continue learning its Gaussian prior on acquisition plus replay
  without adding features. This controls the value of the added computations.
- Neural ALPaCA with 64 learned features: pretrain on old data and continue ALL
  neural/prior parameters on new data plus replay. Capacity exceeds the program
  coefficient dimension; frozen neural features are not the main competitor.
- CoDA: original selected whole-network contextual model, extend its train-task
  contexts for eight acquisition bodies and continue ALL shared/context parameters
  with replay. Adaptation learning rates remain validation-selected.

Use the existing faithful core scopes and disclose their limits; this is not a
full published benchmark reproduction or coverage of the entire modern field.
No comparison solely against a handicapped frozen learner earns continuation.

## Frozen budgets and selection

One initialization seed, two Adam LRs .001/.003. Pretrain old DAG prior and
neural ALPaCA64 for 2,000 steps/rate; choose old validation checkpoints only.
CoDA pretraining reuses both earlier fits and their validated selection. Continue
all five methods for 8,000 steps/rate, checkpoint every 2,000. Batch eight bodies,
half acquisition/half old replay; support 0/4/16/64, 32 disjoint-trajectory queries.
Continue from the selected pretraining state, initialize appended prior dimensions
with unit precision/zero mean, and initialize new CoDA contexts without parameters.

Graph search uses 64 training episodes per addition, support 16 or 64, 32 disjoint
queries. Half old/half acquisition episodes during growth; old only in pretraining.
Search ridge .1, fixed. Up to 512 proposed fragments per addition, maximum 15
expanded operation nodes, maximum 120 wall seconds per search phase; save any
budget termination, never silently rerun. A graph short of its feature count is
retained as the actual outcome. Final learned coefficient precision is unrestricted
SPD within the ALPaCA core. All physical experience, search, fit, tuning, parameters
and wall costs are recorded. No speed claim from a single timing.

Validation combines old and new body means with equal group weight and support
4/16/64. CoDA adaptation LR grid .01/.03/.1, 100 steps, unchanged. Program/ALPaCA
condition analytically. Select checkpoint/global LR using validation ONLY, then
freeze all choices before test. Forecast tests use trajectory0 prefix and all
trajectories1–7, budgets0/4/16/64, plus wrong-body support.

## Falsifier and continuation

Continue only if at 16 or 64 support the meta-grown program reduces new-law ID
mean error by at least 10% versus BOTH continued neural competitors and frozen
old DAG, without >5% harm at the other budget, new-law OOD or old-law retention
versus the strongest neural comparator. Correct support must help versus wrong.
Claiming advantage of its selection criterion additionally requires beating
fit-grown features by at least 10% at the claimed budget. Otherwise explicitly
separate any expansion benefit from criterion benefit. No nonfinite test body
is dropped. No test-informed grammar, physics or search-budget changes.

A surviving exploratory screen earns fresh training-seed/acquisition replication,
common-access actual control, and a structurally different family before any
framework claim. Failure retires this formulation; no favorable new grammar or
feature menu is a repair. Independent direction criticism already supports this
one bounded screen with the above conventional continued-learning competitors.

Prefit implementation specifics: initial atoms are constant 1 and the eight
old-train-standardized observations. Proposal cap divides into up to 256 shallow
and 256 two-operation fragments sampled by fixed RNG. Normalization and numerical
semantic pruning use a fixed 2,048-observation training-only probe. Reject raw
probe magnitude above 1e6, standard deviation <= .05, or residual energy outside
the existing full linear feature span <= 1e-5. This finite semantic pruning is
not a symbolic identity proof; it also excludes alternate parameterizations of
the same span. Retain only a training-episode improvement greater than
max(1e-9,1e-5*currentloss). Existing nodes and normalization stay frozen when adding
new fragments. Search uses float64; learned priors/neural comparisons use float32.
Graph seeds [20261020,18,42,phase_index] (old/meta/fit indices 0/1/2); fit init seeds
[20261020,18,42,old_or_continue_index,method_index], with method order in driver.
Training-episode seeds are [20261020,18,41,old_or_continue_index]. These are fixed
before physical sampling, graph search and learner fitting.

Prefit instrument correction: the original eight-substep spot check failed its
unchanged 1e-7 flow criterion in two of six new-law cases (maximum 4.95e-7).
No new data had been sampled and no learner fitted. Preserve the original source
and failure in structure_acquisition_data_8substeps_*; increase RK4 to 16
substeps, keep the criterion and physical family unchanged, then repeat only
these six spots. This supersedes the eight-substep sentence above. The old DAG
search used unchanged old data and had already completed; it is not rerun.
