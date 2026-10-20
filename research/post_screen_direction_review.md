# Direction review after two defeated constructions

2026-10-20. The verified negative checkpoint f56ea3c14043ff5b10a76bd4d0525efb288b5188
was pushed over the existing SSH remote; `git ls-remote origin refs/heads/main`
returned that exact commit. The ultimate Goal remains active and unachieved.
The two-candidate attempt is closed under the effort boundary in AGENTS.md.
This review begins the next investigation; it is not a third mechanism proposal.

## First-principles reframing

The two screens supplied the experiences, the grouping of experiences by body,
and a batch of related training bodies. They tested how experience conditions a
prediction rule. A deployed learner additionally has to decide what experience
to obtain, which past knowledge is applicable, and what reusable knowledge to
change after an encounter. Those are separate choices; a new feature head does
not automatically address them. Neither screen demonstrates a fundamental
failure of established approaches at those broader choices.

Two tempting next steps are therefore insufficient without a sharper derivation:
"choose informative actions" and "keep learning between tasks." These identify
capabilities, not yet a mechanism or an established gap. More information about
an entire model need not change a useful decision, and retaining data need not
provide beneficial transfer. These are starting distinctions, not new theorems.

## Serious existing accounts checked before another implementation

[FLEX, Blanke and Lelarge, ICML 2023](https://proceedings.mlr.press/v202/blanke23a/blanke23a.pdf):
read sections 3.3–5, especially the one-step information objective and nonlinear
parameter linearization. It already selects actions through D-optimal design,
with a computational approximation for nonlinear models and online model updates.
Fisher/log-determinant exploration around a learned model is consequently a
serious existing baseline, not a sufficient new proposal. The local approximation
and information objective have stated scope; their existence does not establish
decision-optimal exploration or settle open-ended learning.

[OMLA, Zhu et al., 2025](https://arxiv.org/html/2503.18684): read method IV and
experimental setup V-A, rather than infer the mechanism from its title. It uses
task-specific LoRA adapters, online meta-learning from retained task datasets,
then fine-tuning on a new task. Its setting supplies task descriptions and
demonstration datasets. Accumulating data and learning adaptation priors online
are already addressed; attaching either to our current predictor would need
an independently derived distinction and a consequential test. This is not a
head-to-head reproduction or a judgment of all continual learning.

Other search hits, including OCCAM and 2026 papers, were located but their actual
methods were not yet read. Do not cite them as evaluated competitors or infer
state-of-the-art performance from abstracts. ALPaCA/CoDA were strong executable
competitors for the completed screens, not proof that the full modern field was
covered.

## Exact next decision

Before new fitting, use one bounded first-principles/prior-art pass to identify
a concrete capability limitation of the strongest applicable existing account,
the available experiences, and the proposed operation that would overcome it.
Within the campaign's initial 30-minute planning allowance, either produce that
proposal with a falsifier and serious competitor or record why no defensible
candidate has yet been derived. Do not turn that into a repeated survey, label
ordinary replay or information gain as a framework, or select an architecture
because its source is easy to modify.

No new experiment, architecture, benchmark or v18 is selected by this review.
No source or result from either retired candidate changes. Do not reopen them
because this initial framing has not yet supplied a successful replacement.
