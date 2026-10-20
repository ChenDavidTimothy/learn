# Primary research reconciliation for the coupled relation test

Consulted after independent design/derivation and while independently auditing
actual outcomes, 2026-10-20. No claim of a new inverse-model idea or new kernel
algorithm. These sources are context/competitors, not authority for the research
foundation or evidence for our experiment's numerical claims.

- Nguyen-Tuong and Peters (2011), *Model Learning for Robot Control: A Survey*:
  https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Nguyen_CP_2011.pdf
  Author-hosted manuscript, sections2.1/2.2. Discusses forward, inverse, mixed and
  multi-step relations, direct/indirect learning, and regression's failure when an
  inverse is multivalued. The present rest-pulse inverse is locally well defined;
  that is not a general robot-control guarantee. Their reported real-robot examples
  are much richer than our instrument; no performance comparison follows.
- Jordan and Rumelhart (1992), *Forward Models: Supervised Learning with a Distal
  Teacher*, DOI10.1207/s15516709cog1603_1:
  https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1603_1
  The original work explicitly studies learning when an unknown system intervenes
  between actions and desired consequences. It precludes novelty claims about
  acquiring a forward model to help action learning. Publisher abstract accessed;
  do not pretend that an inaccessible full-text copy was read successfully.
- Romeres et al. (2018), *Derivative-free online learning of inverse dynamics models*:
  https://arxiv.org/abs/1809.05074 and https://arxiv.org/pdf/1809.05074
  Primary paper tests inverse dynamics on real iCub-arm data and learns features of
  position histories rather than relying solely on numerical velocity/acceleration
  estimates. It is a particularly relevant established alternative for the NEXT
  continuous-motion problem. Do not call our rest-pulse regression derivative-free
  continuous inverse-dynamics control or claim to reproduce/outperform that paper.
- Tedrake, *Underactuated Robotics*, multibody derivation:
  https://underactuated.mit.edu/multibody.html
  Source for comparison of independently derived point-mass Lagrangian equations.

The robust conceptual reconciliation is conventional: choosing which unknown
relation to approximate changes function complexity, noise placement, and decision
conditioning. A learned inverse relation is still a model of consequences, and
using its pseudoinverse in a bounded solver is not a purely model-free learner.
Equal kernel feature counts do not make forward and inverse physical-function
classes equivalent. In our leading mechanics, M is affine in cos(q2), while M^-1
is rational in cos(q2); this can favor one representation in a finite basis. The
ordinary learners are given generic periodic RBFs and torque-affinity/rest context,
not the true M formula or which encoder matters. All these supplied priors must
remain explicit. No conclusion of universal model necessity is supported.

Next scientific direction after this ONE compact calibration test should remove
rest preparation and examine continuous coupled histories, relevant missing
context, or model revision, with derivative-free history methods and strong
applicable baselines. Do not accumulate more calibration releases.
