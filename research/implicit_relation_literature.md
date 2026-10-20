# Independent derivation and closest existing explanations

2026-10-20. Exploratory mechanism, not an originality claim. First ask whether a
simple constraint can be reused when its solved forward function varies
nonlinearly with context. The proposed learned relation gives a linear support
coefficient problem and a rational query function. It is a supplied family;
experience learns its features and conditions its coefficients. The scalar
example y=g*u/m does not establish an advantage because k=g/m suffices for an
ordinary explicit model. A more relevant conceptual example is (1+c*x)y=1,
whose solved functions do not generally lie in one fixed finite additive basis.
This representational observation supplies neither an optimization guarantee nor
a practical win over nonlinear context-conditioned networks such as CoDA.

[SINDy-PI, Kaheman et al. 2020](https://arxiv.org/html/2004.02322): read sections
1–3.2, including implicit libraries, trivial-zero degeneracy, fixing one term,
and prediction-based model selection. Implicit and rational identification are
established. SINDy-PI uses a prescribed candidate library and sparse fitting; our
screen learns cross-body feature relations and evaluates coefficients through
disjoint-query meta-loss. These differences are operational, not evidence of
originality or superiority. Both inference weighting and errors-in-variables
remain issues; a small equation residual is not a forecast guarantee.

ALPaCA and CoDA are executable strong competitors; their primary-source mapping,
losses and reproduction limits are in error_transport_literature.md and
error_transport_baselines.md. ALPaCA already learns a transferable feature/prior
for rapid Bayesian adaptation. CoDA can express nonlinear function families by
adapting a low-dimensional full-network context. Neither is assumed restricted
to reconstructing physical parameters. The nested explicit comparator is useful
but has fewer parameters than the implicit candidate.

Other formulations considered before this selection: unknown-boundary adaptation
already has MOCA, FOML and LEEDS approaches; latent modular composition has serious
prior work (Schug et al., 2024). Their existence does not settle the general
learning problem, but no distinctive mechanism was derived here that justified
implementing a generic reset or module-selection wrapper. This second screen
tests a materially different conditioning operation, not a rescue of transport.

Independent algebra and scientific-direction criticism are recorded separately.
No claim that all implicit/meta-learning literature has been exhausted.
