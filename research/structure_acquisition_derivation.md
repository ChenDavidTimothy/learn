# What the structure-acquisition rule actually learns

2026-10-20. **HYPOTHESIS:** a compact computation acquired from one collection
of experiences can make subsequent systems easier to learn. Neural parameter
learning can already acquire computations; this hypothesis must beat that account
empirically, not assume it incapable. The executable candidate specializes the
question to a growing feature program and Bayesian coefficient conditioning.

## From observations to a candidate operation

Let Phi_s contain the outputs of current executable features on observed support
inputs, Y_s their measured consequences, and Phi_q the same features at different
query inputs. During graph search, use the supplied sum-ridge objective

    min_W ||Phi_s W - Y_s||_F^2 + lambda ||W||_F^2,
    G = Phi_s^T Phi_s + lambda I,
    W = G^-1 Phi_s^T Y_s, lambda = .1.

The feature graph is the reusable acquired object. Within a system, W changes
through support observations. Across systems, the search can add a computation z
to that graph, so later systems have a different set of available consequences.
This is a design choice, not a necessary definition of intelligent learning.

**PROVEN UNDER ASSUMPTIONS:** for fixed features/normalization and lambda > 0,
the exact prediction after adding one candidate column can be computed without
refitting every augmented system. Let z_s,z_q be that candidate, and set

    h = Phi_s^T z_s,
    v = G^-1 h,
    d = z_s^T z_s + lambda - h^T v,
    a = (z_s^T Y_s - h^T W) / d.

Block elimination of the augmented normal equations gives old coefficients
W-v a, new coefficient a, and query prediction

    Phi_q W + (z_q - Phi_q v) a.

The Schur complement d is positive; equivalently
d=lambda+||z_s-Phi_s v||^2+lambda||v||^2 >= lambda. This derivation assumes the
same penalty for the new feature and existing columns (including the constant).
Changing a sum to a mean without adjusting lambda would change the learner.
Independent augmented SVD fixtures check the implemented batched formula; no
novelty is claimed for block elimination or ridge regression.

The proposed selector averages actual disjoint-query errors after support fitting
over training-system episodes. The ordinary-fit comparator fits and evaluates on
the same union of support/query observations. Both repeatedly reuse training data;
neither selection score is independent confirmation. They use separate fixed
proposal RNG streams, so a one-run contrast is exploratory, not a clean causal
estimate of the criterion alone. Stronger attribution needs paired replication.

## What is supplied and what is acquired

Supplied: exact observation/action interface and sin/cos angle encoding; grouped
physical systems; generic operators and allowed fragment proposals; search limits;
ridge search objective and Bayesian linear coefficient form; replay split and
meta-training procedure. In particular, multiplication/abs suit the selected new
actuation/drag family. No universal-program or representation-free claim follows.

Acquired: specific composed operations, their training-derived normalization,
and a Gaussian prior over their coefficients. New-system observations acquire
posterior coefficients using the learned feature programs. Final prior fitting
uses predictive Gaussian NLL, rather than the fixed-ridge search surrogate.
Testing this surrogate's usefulness is part of the experiment.

Adding duplicate features can reduce ridge regularization without adding a new
function span. The finite training-probe projection removes approximate members
of the existing span, including such duplicates. This is not proof of functional
independence outside those observations; it is an explicit supplied search bias.
The frozen-graph comparator tests expansion's practical value, the fit-grown
comparator probes the selection rule, and full neural continuation with replay
tests the stronger conventional explanation.

The hypothesis survives only through later held-out learning performance under
the declared guards. A readable expression, reduced acquisition fit error,
positive Schur complement or successful program execution is insufficient.
