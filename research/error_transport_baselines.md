# Scoped baseline implementations for the error-transport screen

2026-10-20. Implemented `error_transport_baselines.py` after reading the actual
primary method sections below; no physical data collection or campaign training
was performed by this subtask. These are understood comparison implementations,
not new candidate mechanisms or complete published benchmark reproductions.

## Interface

Both Torch modules accept standardized float32 tensors `xs[B,S,8]`,
`ys[B,S,2]`, `xq[B,Q,8]`. `.predict(xs,ys,xq)` returns `[B,Q,2]`.
`.training_loss(xs,ys,xq,yq,task_ids)` returns a scalar. Constructors expose
width/configuration for the root's validation-only settings. Train-only
normalization, body/split grouping and fair sampling belong to the driver.

`ALPaCA` (`ALPaCABaseline` alias) has hidden width64,32 features, learned prior
mean K0 and full SPD prior precision `L L^T`; L has softplus diagonal plus1e-4.
Its two-output observation covariance is fixed diagonal, configurable through
positive `noise_variance` (default .01 in standardized units). It is a surrogate
likelihood scale, not a claim the noiseless physical transition has noise .01.
`.posterior` returns coefficients/precision Cholesky; `.predictive` returns mean
and marginal diagonal variance. Zero support returns the learned prior prediction.
Training minimizes the Gaussian marginal predictive NLL, including variance,
normalizer and output-noise scale, with gradients through features, K0 and prior.
Shared task IDs are deliberately unused; evidence selects the posterior.

`CoDA` (`CoDABaseline` alias) has a complete8→64→64→2 tanh MLP represented by
4,866 base parameters, shared decoder W[4866,8] and32 learned training-task
embeddings. EVERY weight and bias changes via `theta_e=theta0+W*xi_e`.
Training predicts both support and query observed transitions with the same
task-ID context and adds `regularization * mean_task(sum_parameter((W xi)^2))`
(default1e-4). This is a summed squared parameter norm, not a mean over4,866
coordinates; regularization settings must respect that scale. Task IDs refer
only to training environments, never test identities or physical labels.

`.adapt(xs,ys,xq,steps=100,lr=.03)` fits only a NEW zero-initialized8D context
using support error plus that same locality regularizer. Base/decoder are
detached; global values and existing gradients remain untouched. It works inside
an outer `torch.no_grad()` by locally enabling context gradients. Query labels
are not an argument. Empty support or zero adaptation steps returns basecontext0.
`.adapt_context` exposes the fitted context; each call starts fresh, so the driver
must charge its actual100 Adam steps at every support budget/evaluation call.
No train embedding is supplied as the starting test context.

The core performs no hidden clipping, support selection, early stopping, label
inference, normalization or task/parameter filtering. Nonfinite training or
adaptation outcomes must be retained/reported by the driver. No finite result is
a guarantee of calibration, optimization success or competence.

## Primary-method fidelity and changed scope

Read [Harrison et al., ALPaCA, Sections3–4, equations5–12](https://arxiv.org/html/1807.08912).
The learned feature/prior, exact conjugate posterior and predictive-covariance
NLL follow those equations. This implementation uses a full Cholesky precision
and fixed diagonal output noise; feature architecture, standardized transition
targets, batch sampling and optimization settings are specific to our screen.
It does not reproduce the original robotic datasets, noise calibration, online
acquisition protocol or reported scores.

Read [Kirchmeyer et al., CoDA, Sections3.1–3.3, equations6–8 and Sections5.2–5.4](https://arxiv.org/html/2202.01889).
The faithful core is joint training of base/linear whole-network decoder/per-task
contexts, squared parameter-locality regularization, and new-context-only
adaptation with the shared model frozen. The original experiments fit integrated
continuous trajectories, whereas this screen fits observed finite transitions.
The context dimension8, architecture, initialization, batch/normalization and
regularization/Adam settings are disclosed screen choices. This is a CoDA-l2
core under a changed measured-transition objective, not full trajectory CoDA,
CoDA-l1 sparsity or GEPS reproduction. These limits must accompany any comparison.

## Proportional implementation fixtures

Run `.venv/bin/python research/check_error_transport_baselines.py`.
Actual JSON is `check_error_transport_baselines_results.json`, seed83172.
ALPaCA batch conditioning matches independently iterated Sherman–Morrison
conditioning within2.39e-7 for coefficients and3.58e-7 for covariance in float32.
Prior prediction error with empty support is7.45e-9. Predictive variances/NLL and
all trainable-parameter gradients are finite.

CoDA functional batching matches separately unpacked per-task MLP operations
exactly. A float64 hypernetwork directional derivative agrees with central
differences to2.00e-11. Joint training gives finite nonzero gradients to base,
decoder and selected embeddings. Three synthetic context-only Adam steps leave
global state and its preexisting gradients exactly unchanged; batched versus
separate-task context fitting agrees within2.54e-7. Empty support equals base
prediction exactly. These are author implementation fixtures, not independent
consequential-result reproduction or meta-learning performance evidence.

Default trainable counts: ALPaCA7,904, CoDA44,050 including38,928 decoder and256
training-context parameters. Equal width does not equal capacity/compute.
Report fitting, tuning and adaptation time/counts; settings are selected using
validation bodies alone. Do not starve or tune these rivals based on test outcomes.
