# Independent critique: Cartesian sensing, ambiguous configurations and reuse

2026-10-20. Design reasoning only; no numerical experiment or new learner is claimed. The payload pilot did not establish a need for adaptation. Removing joint encoders creates a different, physically legitimate information problem, but it must earn a learning/control result rather than another aliasing-only release.

## What is actually indistinguishable?

For unit links, `p(q)=(cos q1+cos(q1+q2), sin q1+sin(q1+q2))`. Let `qA=(-phi/2,phi)` and `qB=(phi/2,-phi)`, with zero velocities. Both have `p=(2c,0)`, where `c=cos(phi/2)` and `s=sin(phi/2)`. Under zero torque every passive position history has the same distribution. No longer passive delay, numerical derivative or estimator can identify the branch from those observations alone. This statement concerns passive indistinguishability, not impossibility of feedback control once actions are applied.

For nonsingular `sin(phi)!=0`, the task Jacobians are

\[
J_A=\begin{pmatrix}0&-s\\2c&c\end{pmatrix},\qquad
J_B=\operatorname{diag}(-1,1)J_A.
\]

The inertia is identical in the two configurations. Consequently the leading rest-pulse torque-to-Cartesian map `B=J M^-1` satisfies `B_B=diag(-1,1)B_A`, with `delta p=.5*T^2*B*u+O(T^4)`. Each map is full rank. A horizontal desired acceleration needs opposite leading torques; a vertical desired acceleration needs the same leading torque. Thus ambiguity is task dependent: not every target requires identifying a branch.

For pure horizontal acceleration `a`, direct calculation gives `u_A=(a/(2s),-s*a)`, `u_B=-u_A`. This gives a useful leading feasibility bound, but exact finite-duration commands must still be witnessed under the actual nonlinear dynamics. Use identical Cartesian goals for both branches, inside the nonsingular workspace, rather than generating goals from hidden branch-specific random torques. Avoid near-extended/folded configurations where poor conditioning would confound missing information with difficult control.

Generic actions distinguish the branches, but actions in `ker(row_x B_A)` are blind at leading order. Equal-prior one-step mean response has `B_average` with zero horizontal row: it is rank one even though both realizations are controllable. This exposes a real danger in averaging unresolved possibilities. **It is only a leading affine statement.** Exact finite-time mean Cartesian response can have horizontal fourth-order effects; common rotation changes horizontal position quadratically. Neither exact horizontal uncontrollability nor “every mean-based controller stays at zero” follows. A full-history conditional mean can also become informative after an action.

## What must be learned, rather than supplied?

The learner should receive Cartesian observations, issued torques, episode/reset boundaries and current Cartesian goals, with noise/timing/bounds explicitly declared. Joint angles, branch labels, true B/J, chosen component identity and future noise belong only to the environment/evaluator. Randomize hidden configurations independently of episode order; do not let parity, filenames or fixed test ordering supply the branch. Retain true coordinates for auditing, not training selection.

Experience must acquire reusable alternative responses, not merely classify two matrices supplied by the researcher. At a single repeated starting position, an unlabelled episode-level mixture of local response regressions or simple episodic retrieval is a serious conventional construction. The number of components is itself a prior if fixed at two. Validation over one/two/more alternatives, or a generic recurrent history learner, would challenge that choice. Learned labels need no physical elbow interpretation.

The alternatives must be coherent over an episode. A model that resamples an independent response matrix each step has the wrong joint law and cannot represent why one observation informs later actions. Check heldout sequence predictions as well as one-step marginal variance. Generalization beyond an exact two-pose lookup needs new orientations or elbow magnitudes; otherwise describe the result as reuse in a small latent-context instrument, not learned general kinematics.

Elbow branches do not spontaneously interchange along an interior nonsingular Cartesian path. Switching them at the same stationary endpoint requires an external hidden-configuration reset or a physical path through a singular boundary. Count and disclose this preparation. Do not call an impossible instantaneous branch switch ordinary unannounced continuous mechanics. After one identification, ordinary estimates/cache memories should persist through goals unless a real episode reset occurs.

## Fair competitors and information cost

The useful claim is that prior experience reduces the additional interaction needed for a specified task. Compare methods with the same offline experience, not a learned library with extensive calibration against a deliberately memoryless estimator started from nothing. Serious alternatives include a nonlinear position/action delay learner, a local MIMO recursive estimator with an offline-learned prior and retained memory, simple episodic retrieval, and a learned mixture or recurrent summary. An aggregate conditional-mean predictor is a diagnostic ablation, not the sole strong competitor.

Classical uncalibrated visual servoing is relevant but its input semantics matter. A kinematic Jacobian maps joint displacement/velocity to task velocity; here commands are torques and the plant is second order. Blindly applying Broyden to `delta p/torque` while ignoring motion/past torque creates a weak mismatched baseline. Use a correct sampled response/dynamic local model, causal history and regularization, and retain its information across ordinary goal changes.

Do not assume a separately labelled “probe” is necessary. An ordinary task-directed action may both make progress and identify the hidden configuration. Compare explicit informative actions with that feedback strategy and equal-duration/energy random actions, including every trajectory's probe motion, time and sensing in the task score. No hidden reset to rest after probing: a real torque pulse ends moving. If resets are allowed as a capability, give them to all methods and count them. Subsequent control must handle the actual post-interaction motion.

Finite sensing is essential. With known common initial Cartesian position, endpoint covariance `sigma^2 I`, `k=.5*T^2`, and two known leading means, equal-prior classification error is `Phi(-k*abs(row_x B_A*u)/sigma)`. This is a diagnostic Gaussian formula, not a learned-method guarantee or finite-T physical identity. If initial position is only measured, baseline uncertainty/correlation must be included; do not repeat the redundant noisy-baseline error from v14. Noiseless infinitely small probes can make identification artificially free.

## Established research and the falsifiable next step

[Hosoda/Asada1996](https://www.jstage.jst.go.jp/article/jrsj1983/14/2/14_2_313/_article/-char/en) describes online Jacobian estimation for uncalibrated visual servoing. The accessed publisher abstract supports this precedent; its complete method is not reproduced here, and torque/velocity interfaces must be reconciled before using it as a comparator.

[Klenske/Hennig2016](https://jmlr.org/papers/volume17/15-162/15-162.pdf), full paper, explicitly treats the dual-control “turn-off” phenomenon: myopic uncertainty handling can suppress the actions needed to learn. Their examples make nonzero information-gathering useful over a longer horizon. The current physical rank-collapse framing is a concrete instrument for that established issue, not a new foundational discovery. Their Gaussian approximations are not automatically appropriate for two coherent opposing modes.

[Nagabandi et al.](https://arxiv.org/html/1803.11347), full paper, uses meta-learned dynamics priors with recent experience for rapid online adaptation, providing a relevant modern reuse competitor and precedent. [PlaNet](https://arxiv.org/abs/1811.04551), accessed paper abstract, demonstrates learned latent dynamics/planning from pixels; it is a reminder that recurrent latent approaches are alternatives, not that a small Cartesian experiment reproduces or outperforms visual robotic learning. Any later superiority claim would need actual applicable implementations and matched information/data budgets.

The decisive experiment asks whether learned prior alternatives improve finite-noise Cartesian goal behavior per total experience, and whether informed interaction improves on ordinary feedback and equal-cost random interaction. Retain the possibility that simple local adaptation or a conventional history predictor is sufficient. A correct aliasing calculation, a branch-classification score or a supplied two-model Bayesian update alone is not the requested general-learning milestone. Explicit beliefs, named states and active optimality remain candidate mechanisms, not mandatory foundations.
