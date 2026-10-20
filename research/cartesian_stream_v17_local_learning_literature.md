# Established local learning: reconciliation during fixed replication

2026-10-20. Read after deriving and constructing the pilot, while the frozen
eight-corpus replication runs unchanged. This note changes no protocol or model.

Primary source: [Atkeson, Moore and Schaal, *Locally Weighted Learning for
Control*](https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1997_2/moore_andrew_1997_2.pdf),
CMU-hosted manuscript dated June1996 (the1997 paper cited in the earlier
independent review). Root read section1, section2's inverse/forward/control
discussion, section3.1 through3.5, and section4 through the dimensionality
discussion; the complete remaining paper/proofs were not reviewed here.

The paper stores experiences, constructs local regression models for queries,
and uses their forward maps/derivatives for control. It discusses local data
coverage, multivalued inverse mappings, limits of one-step correction, predictive
regulation, exploration and growing lookup costs. Its control examples make
additional task/state assumptions. This is direct precedent for our conventional
local mechanism, not a method it should be presented as superseding.

Own application and limitation: our camera history differs from an assumed
observable task state. We supply the lag window, slopes, metric, local affine
class, regularization menu and planner. The data acquire useful local
coefficients inside that vocabulary; they do not discover every notion of
relevance or a universal state. The paper's local-model rationale does not prove
this particular finite camera window is sufficient, nor that a nearby record
has the same hidden response. Those are matters for our actual tests.

Our independent proof that fixed local coefficients yield affine plan rollouts
and a strictly convex quadratic is a property of this deliberately restricted
surrogate. It does not prove the true physical objective convex or make the
nonlinear predictor's optimization equally reliable. A common optimizer budget
does not remove the distinction between its two objective surfaces. Preserve
the neural pilot's large projected gradients and avoid family-level superiority.

The observed retention question is also conventional: adding experience changes
the local data support and hence later decisions. Fresh replication can establish
its practical effect here; a positive result would not make memory-based local
learning newly invented. Conversely stored episodes, local regression and
conditional prediction are compatible descriptions of the same computation.
Do not manufacture a conceptual competition by merely renaming those objects.

This review strengthens the simplest existing explanation. The scientific value
of the present investigation must be its verified information/experience boundary
and useful decisions with fewer supplied physical certifications, including
negative evidence and remaining assumptions. It cannot rest on novelty of a
weighted local model or an isolated simulation success. The ultimate nonlinear
robotics/general-learning objective remains open.
