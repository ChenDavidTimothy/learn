# Moving Cartesian question: primary context and limits

2026-10-20. Research continuation after verified v16, not a new release.
Independent derivation/instrument preceded this literature reconciliation.

## Classical observability: distinguish the questions

Primary source: [Hermann and Krener, 1977, author's PDF](https://www.math.ucdavis.edu/~krener/1-25/10.IEEETAC77.pdf), printed pp732–733, sectionIII definitions and theorem3.1, visually read from the original scan. The complete paper/proof was not reviewed here. A search-indexed university mirror failed404; the author's copy was obtained successfully.

The paper separates indistinguishability over admissible inputs from what one input reveals, and distinguishes local and global observability. Its rank condition implies local weak observability. Our application must respect those qualifications.

Own elementary check: in one nonsingular elbow chart, the derivative of(q,v)->(p(q),J(q)v) is block triangular with two diagonal Jblocks, determinant det(J)^2 nonzero. Local reconstruction is therefore possible even at rest. This does not settle the other elbow's identical stationary ZERO-INPUT record. Nor does that passive alias imply indistinguishability under every available command. The new acceleration calculation concerns two separated inverse-kinematic alternatives and a particular information contract, not a novel rank theorem.

## History learning is a serious conventional alternative

Primary source: [Dalla Libera etal., 2020](https://arxiv.org/html/2002.10621), sectionsII–III andIV including training/causal-filter/open-loop-control comparisons, read in full here; the v15 lineage already records a broader review. Their model uses position history with Gaussian-process transition learning and offers both nonparametric and physically informed kernels. It is precedent for learning useful history relations without derivative targets, not evidence that a Cartesian lag window is always sufficient.

Own inference: the paper's measured generalized coordinates differ from the present noninjective endpoint. Its physical kernel still supplies structural information. We should compare an appropriately informed generic history learner, not grant joint state to it or call all uncertainty a missing derivative. The stationary Cartesian counterexample does not refute the paper under its own sensing assumptions. No full implementation was reproduced or outperformed here.

## Two corrections to the prospective framing

Independent polar mechanics agrees with the root joint-space calculation. Moving passive separation is quadratic in speed. For unforced conservative motion, rescaling initial speed rescales time along the geodesic: waiting can supply evidence, and its time/motion must be counted. A finite noisy short-window limitation is not categorical lack of information.

The artifact auditor gives a different counterexample: multiply BOTH point masses by k>0. Inertia M and Coriolis h both scale by k, so qddot=-M^-1h and every zero-torque trajectory are unchanged for matched q/v. The torque contribution M^-1u scales1/k. Passive identification of elbow configuration inside a known body family therefore does not learn absolute intervention gain. This follows directly from mechanics, independent of the classical local-rank result. Whether gain uncertainty matters to a specific task still needs fair decisions against robust feedback; full parameter identification is not automatically necessary.

These facts constrain the next acquisition/goal contract. They do not select a mandatory architecture or warrant a v17 ledger by themselves. The next substantive work must acquire a transferable conditional relationship from actual moving experience and use it in charged decisions with fewer supplied physical assumptions.
