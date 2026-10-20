# Independent Cartesian-branch information and construction note

Date: 2026-10-20. First-principles proposal, no canonical experiment implemented. This changes sensing rather than increasing payload until a baseline fails. It is not a milestone by itself.

## Physical aliases and their consequences

For the same two unit links, end-effector position is

    p(q)=[cos(q1)+cos(q1+q2), sin(q1)+sin(q1+q2)].

Let qA=(-phi/2,phi), qB=(phi/2,-phi), with0<phi<pi. Both give p0=(2*cos(phi/2),0). With zero velocity/torque, both produce exactly the same stationary Cartesian history for any duration. Under a common independent camera-noise law their observation distributions are identical; a previously acquired dataset cannot identify the present branch from these observations alone.

The joint inertia is identical because it depends on cos(phi). With s=sin(phi/2), c=cos(phi/2), direct differentiation gives

    JA=[[0,-s],[2*c,c]], JB=[[0,s],[2*c,c]].

At rest under constant torque, smoothness/time reversal imply

    p(T)-p0=.5*T^2*Bbranch*u+O(T^4), Bbranch=Jbranch*M^-1.

The Cartesian transformation adds only fourth-order corrections because the joint displacement is second order. Remainders are uniform on a suitable compact action/pose region. At phi=pi/2,

    BA=[[1/(2*sqrt(2)),-3/(2*sqrt(2))],
        [1/(2*sqrt(2)), 1/(2*sqrt(2))]],
    BB=[[-1/(2*sqrt(2)),3/(2*sqrt(2))],
        [1/(2*sqrt(2)),1/(2*sqrt(2))]].

Their radial rows are opposite, tangential rows equal. A common first action cannot produce the same nonzero leading radial acceleration in both branches. Both individual maps are nonsingular (detJ=plus/minus sin(phi), detM>0), so sufficiently small Cartesian goals have individual local inverse commands; action bounds are met when the requested leading acceleration is sufficiently small. Exact finite-time witnesses must still be checked before evaluating failed controllers.

This is a task-specific distinction: tangential goals can share the leading action relation, while radial goals require resolving sign. It does not prove that every long common open-loop sequence fails, that a history feedback controller cannot work, or that every finite goal/deadline is reachable. Finite-time radial means can have fourth-order common curvature effects; the exact mean radial response is not identically zero for arbitrary finite pulses.

## Learning alternatives without being handed their labels

Under an equal initial branch prior and randomized first interventions independent of that hidden branch, write the leading normalized response as

    z=[sigma*(r*u), t*u]+eta, sigma=plus/minus1,
    r=[1,-3]/(2*sqrt(2)), t=[1,1]/(2*sqrt(2)).

An ideal conditional mean has radial row0, although each real branch has radial authority. A planner that substitutes that mean can therefore erase a controllable direction; this is a consequence of averaging alternatives, not proof of a particular optimizer's actual finite-time behavior.

There is information beyond the mean in UNLABELED interventions. With independent centered response noise of known variance Rxx,

    E[z_x^2|u]=(r*u)^2+Rxx=u^T*(r^T*r)*u+Rxx.

Diverse issued actions can identify the rank-one quadratic r^T*r up to sampling/approximation errors; its factor is r up to sign. Tangential first moments identify t. One need not mandate an explicit mixture architecture to retain this reusable variation. A generic conditional second-moment learner, latent mixture, episodic alternatives or suitable continuous parameter prior could all use it. The learner must not receive the true radial axis, r/t, branch labels, joint states or named two-mode formula as if discovered. The displayed symmetry is an evaluator derivation, not the ordinary model's input.

These response identities assume the baseline position is exact or response noise is independently correctly modeled. Differencing two noisy camera readings shares the initial error with observed context; v14's measurement correction cautions apply again. A raw-observation sequential likelihood is another option. Actual finite-pulse dynamics introduce approximation error that known-noise subtraction does not remove. This derivation is not a reason to publish another isolated covariance result.

## Can learned variation reduce new probe interactions?

If experience has narrowed the local alternatives to BA and BB, any action satisfying(BA-BB)*u nonzero gives distinguishing evidence; in the noiseless leading model one response suffices to resolve the sign. Relearning a completely unrestricted2x2 B requires rank2 action excitation to identify the full matrix. However, a single goal need not require identifying every matrix coefficient, so this is not a universal two-probe lower bound for useful action. Subsequent goal pursuit can itself be informative.

A strong same-data alternative is a continuously uncertain local-Jacobian model with a learned mean/covariance prior. For this symmetric example B=Bbar+zeta*D, even a continuous Gaussian prior on zeta can preserve the same learned direction D and update it from one probe. Do not attribute all reuse benefits to discreteness or compare a learned multimode prior only against a baseline forced to discard the acquired data. A from-scratch local estimator is a useful ablation, not the sole strong competitor.

At finite noise the relevant test is total goal-control cost and precision versus actual action/time/sensing budget, including informative motions and wrong-direction initial excursions. Discriminating the hidden label is only a diagnostic. Choosing the most separating action ignores its control cost, so a generic probe or ordinary task-driven local update may perform as well. Nothing here requires separate probing, an explicit belief variable or a world-model architecture.

## Minimal substantive construction and failure criteria

Keep the physical body. Supply only noisy Cartesian positions, issued torques, clock/bounds and Cartesian goals. Acquire genuine mixed-branch experience without hidden labels or encoders. Preserve trajectory grouping when it represents a persistent hidden context, but do not silently turn evaluator branch IDs into learning supervision. Initial rest/position preparation, if supplied, remains an acquisition resource.

Evaluate a small prespecified set of new radial/tangential goals after indistinguishable stationary histories, allowing sustained feedback and no branch announcement. No future physical reset is granted during each task. Fit all ordinary mechanisms from the same acquired data. Compare a serious generic history predictor/controller and local online Jacobian/visual-feedback baseline, including its learned-prior version; add a multiple-consequence mechanism only when it is genuinely distinguishable. True B/q/v belong solely to feasibility/oracle diagnostics. Review established visual-servoing/partial-observation learning literature before freezing an algorithm comparison.

Useful outcomes include ordinary history/local updating being sufficient, or reusable learned variation improving total decisions with fewer interactions. A conditional-mean failure alone is insufficient if the baseline is trapped by an avoidable local optimizer or wrong finite basis. A branch-classification theorem or rest-pulse plot alone is not a release. The consequential question is whether experience preserves distinctions that change attainable decisions when observations genuinely alias physical possibilities.
