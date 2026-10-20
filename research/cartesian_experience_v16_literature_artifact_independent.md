# Independent literature and design audit: acquiring reusable Cartesian responses

2026-10-20. Design critique, not a numerical result, implementation reproduction, or release recommendation. Read against the verified v15 ledger and Cartesian question. This supplements `cartesian_ambiguity_v16_design_artifact_independent.md`; it does not replace its finite-duration, task-direction, and preparation caveats. No canonical experiment source has been edited.

## What question would this construction resolve?

The consequential question is whether population experience preserves relationships among alternative responses that help a learner use one new observation for subsequent decisions. Merely showing that two hidden configurations have the same resting Cartesian output does not resolve it. Nor does supplying their two response matrices and implementing a posterior update. The matrices, variation, and useful dependence across actions must be acquired from unlabeled physical experience.

The smallest serious comparison is an offline-learned response mean and full coefficient covariance, an ordinary linear-Gaussian posterior update, and the same bounded quadratic continuation optimizer used by richer priors. Mean-only planning is an informative ablation. It is not the strongest conventional alternative. A coherent mixture and a Gaussian matched to its first two coefficient moments help distinguish prior shape from estimation differences. A separately fitted general Gaussian and fitted mixture also differ in fitting constraints, regularization, and selection, so their difference cannot be attributed solely to non-Gaussianity.

This is a constrained local construction. Exact Cartesian starting position, initial rest, episode grouping, actuator timing, and an approximately constant linear torque-response vocabulary are supplied structure. If the conventional Gaussian mechanism is sufficient, that is an acceptable answer. Neither a discrete elbow classifier nor an explicitly named belief architecture is logically required.

## Correction before data: one-action marginals do not identify a coherent prior

Let theta be row-major vec(B), and let H(u)=I_2 tensor u^T, so that B u=H(u) theta. Define

    A = [[0,1],[-1,0]],
    K = [[0_2,A],[A^T,0_2]],
    Gamma_plus = I_4 + epsilon K,
    Gamma_minus = I_4 - epsilon K,       0 < epsilon < 1.

K is symmetric, K^2=I_4, and both Gamma matrices are positive definite. For every u,

    H(u) Gamma_plus H(u)^T
      = H(u) Gamma_minus H(u)^T
      = ||u||^2 I_2,

because u^T A u=0. Zero-mean Gaussian coefficient priors with those covariances therefore give exactly the same single-action marginal response distribution, including after adding the same independent sensor noise. More single-pulse episodes with unrelated hidden contexts do not resolve this ambiguity.

Their cross-action laws differ. In particular,

    Cov(B e1, B e2) = +/- epsilon [[0,1],[-1,0]].

An observation following e1 consequently has different implications for the response to e2. Single-action covariance fitting cannot recover a general coherent four-coefficient prior. Special elbow families may remove the ambiguity through additional restrictions; those restrictions must not silently become a claim about general covariance discovery.

Both independent auditors raised this before Cartesian acquisition. The coordinator accepted the correction and changed acquisition to two consecutive actions in the same hidden context, without resetting between them. Under the locally constant B approximation,

    p1-p0 = T^2 B (0.5 u0),
    p2-p0 = T^2 B (1.5 u0 + 0.5 u1).

The two effective input columns have determinant 0.25 det(u0,u1). Thus their group design is full rank whenever the commands are linearly independent. Fitting whole groups exposes the cross-action dependence that the original marginal design omitted. Known exact p0 avoids unnecessary baseline measurement noise; independent endpoint sensor errors give a simple known group covariance. Nearly collinear commands still give poor conditioning. Whole-group likelihood/moment fitting is preferable to inverting each noisy episode and treating the resulting B estimate as exact. Full nonlinear motion introduces approximation error even with exact numerical integration.

## Full-text primary research inspected

The method and relevant experimental sections below were inspected in the full papers, beyond their abstracts. No published implementation was reproduced. These are precedents and tools, not evidence that the proposed learner matches their robotic capability.

**Dual control.** [Klenske and Hennig, 2016, sections 3–6](https://jmlr.org/papers/volume17/15-162/15-162.pdf) explicitly analyze the turn-off phenomenon: cautious myopic control can suppress informative action. Their longer-horizon treatment accounts for observations changing subsequent decisions. They develop approximations rather than a general exact Bayes-optimal solver, and examine task-relevant exploration. The proposed experiment therefore revisits an established question. A Gaussian prior is a serious comparator, but its adequacy for opposing coherent modes must be tested. Their framework also emphasizes joint uncertainty in physical quantities and model coefficients; a noisy post-probe observation cannot safely be treated as a separately known physical state.

**Reuse through online adaptation.** [Nagabandi et al., ICLR 2019, sections 4–5](https://arxiv.org/html/1803.11347) learns dynamics priors that adapt from recent experience through gradient or recurrent mechanisms. Its comparisons include a pooled model and ordinary test-time adaptation, with common model/control choices. This motivates the same-data prior plus ordinary online update baseline here. Their state-based dynamics and much broader robotic experiments are not equivalent to Cartesian ambiguity, and their neural architecture need not be copied for four local coefficients. Their work is precedent for experience reuse, not evidence of superiority for the present construction.

**Visual-servo interface.** [Chaumette and Hutchinson, 2006, sections II–IV](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf) develops feature feedback through interaction matrices mapping camera velocity to feature velocity, with conditioning and stability qualifications. [Przystupa et al., 2021, algorithm 1 and experiments](https://arxiv.org/pdf/2106.06083) compares learned Jacobians with local linear and Broyden approaches; its controller receives joint angles and issues joint velocities. Those inputs differ materially from hidden joint configurations and torque commands. Neither paper justifies an ignorant delta-position/torque Broyden baseline. The transferable idea is local response estimation with appropriate dynamics and retained information, not granting unavailable joint state or confusing a torque-to-acceleration map with a kinematic Jacobian.

**Modern probabilistic model-based control.** [Chua et al., PETS, 2018, sections 4–5 and algorithm 1](https://arxiv.org/pdf/1805.12114) combines probabilistic dynamics ensembles with trajectory-sampling MPC. Its TS-infinity variant retains one model through a sampled trajectory, illustrating why temporally coherent uncertainty matters. The paper explicitly leaves directed exploration for future work; it is not an implementation of the present probe-value calculation. Its standard state-transition interface does not automatically resolve hidden Cartesian configurations. A small coherent coefficient prior is an appropriate minimal baseline for this instrument, but no superiority to PETS is established without running an applicable implementation.

## Minimum fair decision comparison

All priors must be learned from the same grouped acquisition episodes, without true q, true B, branch labels, episode parity, or hidden action-response axes. Episode boundaries and known preparation are legitimate common information. Validation should score heldout groups, including calibrated uncertainty; mean prediction error alone cannot select among coherent variation models. Retain all fit candidates, PSD projections/floors, negative raw covariance eigenvalues, likelihood failures, and conditioning information. Selecting one/two/three mixture components avoids silently supplying the number of physical alternatives, although the finite mixture menu remains a structural prior.

Compare purposeful first actions with ordinary task actions that also update the same prior, plus paired equal-amplitude or equal-effort random first actions. Every strategy gets the same observations and continuation solver. A task action may already gather enough evidence; a separate probe must earn its movement, time, effort, and risk. Maximizing response variance or entropy is not automatically minimizing task loss. Any finite first-action candidate search should be described as such rather than globally optimal active learning.

The proposed first deployment is a two-stage contract: choose u0, observe p1, then commit to u1 and u2. It contains three physical pulses but only one intermediate adaptation. It is not full three-stage feedback or general dual control. Under a constant B, fixed u0, and quadratic endpoint/stopping/effort loss, the remaining four-dimensional bounded optimization is convex and depends only on posterior coefficient first and second moments. This mathematical simplification does not make those moments sufficient for choosing u0 under arbitrary priors: the observation-dependent posterior law matters.

Account coherently for post-probe motion. Starting at rest, the approximation gives v1=T B u0 and p1=p0+0.5 T^2 B u0. Both are functions of the same uncertain B. Conditioning on p1 updates their joint uncertainty; inserting the noisy reading as an exact initial position while separately updating B would ignore correlations or use the same information inconsistently. Final formulas should retain p3-p0=T^2 B(2.5u0+1.5u1+0.5u2) and v3=T B(u0+u1+u2), with the physical evaluator checking the approximation under continuous mechanics. True-state endpoint scores are acceptable evaluator diagnostics, not ordinary information for selecting fits or actions.

Use common independent heldout contexts/noise for paired decisions, count acquisition preparation and every physical probe/test, disclose offline fitting and hypothetical planning separately, and preserve high-loss cases. Test nonlinear residuals and actual feasible tasks; a local leading-response oracle is not an exact finite-time task guarantee. Ordinary estimators should retain experience across goals unless a real episode reset occurs.

## Drift judgment and continuation criterion

This construction can answer whether coherent response variation learned from unlabeled experience helps new bounded decisions, and when a Gaussian update is sufficient. It does not establish general visual robotics or a new learning principle. A single fixed Cartesian preparation with two hidden possibilities and three pulses can collapse into hierarchical linear regression with known preparation. That is a useful diagnostic if stated plainly, but repeating optimizer menus until one wins would not address the ultimate objective.

Proceed as a construction pilot, not an aliasing-only milestone. If useful, the next consequential test should preserve acquired knowledge in sustained Cartesian feedback with ordinary goals, moving histories, and no goal-change resets. If conventional uncertainty-aware adaptation performs comparably, accept that evidence and investigate its actual boundary rather than increasing difficulty merely to force a failure. No external benchmark superiority or broad mechanism claim follows from this design review.
