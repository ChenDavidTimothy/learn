# Separating useful action dependence from moving drift — prospective derivation

2026-10-20. Root reasoning after the independently verified weak-contrast pilot,
BEFORE actual control outcomes. This is a proposed next diagnostic/formulation,
not a selected learner, test-tuned rescue, v17 milestone or novelty claim. The
existing charged pilot remains unchanged. A separate independent theory critique
must challenge these identities and their relevance.

## What uncertainty this could resolve

The first corpus has new commands randomized at known renewal boundaries.
This is real experimental information, beyond passively observing motion.
Can that randomization give an ordinary-data criterion for action knowledge,
without privileged repeated-state forks and without requiring the learner to
identify a complete hidden physical state? The observed negative does not yet
show whether finite history lacks useful information or fitting/selection mostly
allocates capacity to moving drift. A more elaborate network is not the only
possible response. A simpler sufficient statistical formulation would matter.

## Own linear-projection derivation [UNDER EXPLICIT ASSUMPTIONS]

Let Z be ordinary context available before a renewed command U, and let Y be a
vector of later camera consequences (or corrections to a Z-measurable baseline).
U is drawn independently of full ordinary/hidden history, with known mean0 and
positive-definite covarianceSigma. Assume finite second moments. For the collector,
the conditional best linear action projection is

    G*(Z) = E[Y U^T | Z] Sigma^-1.

No affine physics is needed to DEFINE this projection. If the conditional mean
actually equals d(Z)+G(Z)U, it recoversG. Subtracting any fixed/past-measurable
dhat(Z) does not change its population moment because E[U|Z]=0. It can reduce
variance when appropriate, but can also increase it if poor; it is not a free
precision improvement. A model trained on the same outcome being scored need not
be fixed/past-measurable; use independent validation or whole-trajectory crossfit.

For fixed candidates, the observable risk

    Q(Ghat)=E[tr(Ghat Sigma Ghat^T)-2(Y-dhat)^T Ghat U]

equals E[tr((Ghat-G*)Sigma(Ghat-G*)^T)] minus the common constant
E[tr(G*SigmaG*^T)]. Complete squares after using the conditional moment. Candidate-
specific dhat is allowed if measurability/independence holds. Comparing squared
pseudo-label errors across candidates with different dhat would introduce extra
candidate-dependent constants; this linear score avoids that mistake.

Neither G* nor a lowQ is automatically an individual world's response matrix or
the true finite-command effect. With symmetric scalar U, Y=U² has G*=0 despite
strong action dependence; Y=U³ has G*=E[U⁴]/E[U²] despite derivative0at0. Thus
action-affinity would be a substantive restriction. Finite.2sec mechanics need
not satisfy it exactly. The supplied uniform[-.5,.5]^2 law has Sigma=I2/12.

## A general centered-action formulation [OWN DERIVATION]

Write mu*(Z,u)=E[Y|Z,U=u], mu_bar*(Z)=E_U mu*(Z,U), and
g*(Z,u)=mu*(Z,u)-mu_bar*(Z). For ANY candidate ghat exactly centered under the
known command randomizer, E_U[ghat(Z,U)|Z]=0, define

    R(ghat)=E[||ghat(Z,U)||²-2(Y-dhat(Z))^T ghat(Z,U)].

Condition onZ: the cross term with any action-independent dhat or mu_bar* vanishes.
The remaining cross term is E[g*^T ghat]. Therefore

    R(ghat) = E||ghat-g*||² - E||g*||².

This does not assume action-affine truth or a physical state architecture. One
construction centers any existing predictor mhat(Z,u) by subtracting its integral
over known randomU. Centering error is consequential: if mean_ugh atZ ise(Z), the
unwanted cross term is -2(mu_bar*-dhat)^T e(Z). Numerical integration errors and
large moving drift can therefore interact; don't call approximate centering an
exact orthogonal score. Independent integration checks would be required.

Even exact R is a finite-data noisy criterion. Large unremoved drift can dominate
its variance; many correlated windows are not independent trajectories. Small
score differences require honest cluster-level uncertainty. Optimizing this
criterion cannot invent unsupported information or make a finite context
sufficient. It estimates average consequences under the acquisition mixture,
not an invariant individual-world mechanism under arbitrary future policies.

## Why ordinary full prediction loss is not fundamentally wrong

Under the action-affine conditional mean and independent zero-mean commands,
ordinary prediction risk decomposes as

    E||Y-dhat-GhatU||² = irreducible error
        +E||d-dhat||²+E tr((G-Ghat)Sigma(G-Ghat)^T).

Thus population squared-error prediction DOES reward correct action effects.
The present negative does not refute prediction as a principle. Finite model
classes, regularization, optimization and limited validation can trade drift
against smaller action effects. A counterexample: restricted candidateA has
perfect drift but zero action gain; candidateB has correct gain but drift errora.
Prediction selectsA if a² exceeds the lost action variance, whereas an action-
only criterion can preferB. This demonstrates a possible tradeoff, not the actual
cause of this pilot. A full accurate predictive model would need no such rescue.

Different tasks may require both drift AND response. Better centered-action
prediction without good drift or charged decisions is not a useful controller.
Nor does this loss select the right intervention opportunities or retain remote
hidden alternatives. A future experiment must distinguish these questions.

## Primary literature reconciliation after deriving the identities

[Nie and Wager, arXiv1712.04912v4](https://arxiv.org/html/1712.04912v4), sections2–3
read here, develops residualized treatment-effect learning, cross-fitting and
validation on an effect-specific loss. It connects to Robinson's transformation
and semiparametric orthogonality. Their binary-treatment, independent-example
formulation and error bounds are not directly guarantees for our continuous
commands, correlated trajectory windows, finite sensor histories or closed-loop
policy shift. No complete implementation or theorem proof was reproduced here.
Robinson1988 is recognized through that primary discussion; its full original
paper has NOT been read here, so don't claim it has. This is established
statistical territory, not a newly discovered universal learning mechanism.

## Next decision, still OPEN

First finish/audit the frozen charged pilot. Then determine whether action-risk
diagnostics can discriminate a useful failure explanation using ordinary data
alone. Any exploratory rep0 analysis is labeledposthoc; fresh corpora/protocols
must test claims. Do not tune new models to the retained testforks or select which
contexts count after seeing failures. If conventional estimation suffices, accept
it. If the score is too noisy or insufficient context is the main issue, preserve
that negative and change the question. Another loss theorem alone is no release.
