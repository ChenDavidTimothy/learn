# Posthoc modeled optimization certificate: fixed design

2026-10-20. This diagnostic is specified after the fresh physical outcomes are
known. It changes no controller, fit, goal, action, optimizer setting, dataset or
scientific comparison. It uses all 11,520 stored ordinary decisions from fixed
fresh corpora 200–207, both retained and frozen variants, every case and phase.
No outcome-based subset or new solve is permitted.

The question is whether incomplete optimization of the actual fixed-query local
planning objective could account for substantial modeled objective loss. A bound
on modeled objective loss does not bound physical tracking loss, task failure,
future histories, or errors in learned predictions.

## Exact mathematical statement to check

At a decision, retain the actual causal observation and command history, current
goal, fit coefficients and initial query. The coefficients and query are frozen
across all six hypothetical command blocks. Every predicted camera coordinate is
therefore affine in the 12 future action coordinates by induction through linear
history construction, affine normalization, affine correction and linear carry.
The finite-difference velocity coordinates are affine too. The actual objective
is a mean of squared affine position errors plus one quarter squared affine
velocities, plus 0.02 times squared action norm. It is differentiable convex
quadratic, with Hessian at least 0.04 times the identity. No clipping, query
refitting or neighbor reselection occurs inside the hypothetical rollout.

For differentiable convex f, a feasible selected plan x and exact gradient g,

    0 <= f(x) - min_{y in [-0.5,0.5]^12} f(y)
       <= g dot x + 0.5 ||g||_1
        = sum_i |g_i| (0.5 + sign(g_i) x_i).

This follows from the supporting hyperplane inequality and minimizing its linear
part over the box. It requires neither a new optimizer nor a physical rollout.
The last expression avoids cancellation between large dot-product and norm
terms. The nonnegative objective also supplies the optional trivial bound f(x);
report the raw first-order bound rather than silently replacing it.

## Provenance and computations

Bind the diagnostic to final completed study results, arrays, fit metadata,
canonical source/dependencies, the physical audit aggregate and the final
numerical fitting/decision audit. Source/hash binding to that last audit will be
completed after its auditor declares it final; do not silently bind a stale cache
or call a pending numerical audit passed.

For each stored selected plan and selected gradient, verify its shape, finiteness,
box feasibility, decision clock and equality of its first command to the issued
physical command. The source explicitly recomputes the selected gradient at the
selected plan after acceptance or fallback; it is not a gradient from a different
initial or optimizer point. Independently reconstruct all selected modeled costs
and gradients using saved local coefficients, causal ordinary history, recorded
query and training scales through the existing independent NumPy rollout and
analytic derivative helpers. This reuses their earlier independent derivation;
it does not refit any model or rerun an optimizer. Independently reconstructed
coefficient provenance belongs to the final fitting/decision auditor.

Save every decision's location, selected cost, raw and stable bound, relative
bound (bound divided by selected objective when positive), lower bound on modeled
optimum max(0, f(x) minus bound), gradient replay errors and numerical discrepancy
between equivalent bound formulas. Relative-to-selected-objective is not relative
to optimum. Zero objective has no defined ratio; record it explicitly, without a
chosen denominator floor. Summarize counts, means, quantiles and worst locations
for every corpus, variant and phase, and globally. All decisions remain recorded,
including any large bound or numerical failure. No value selects a new plan.

The convex inequality is an exact-arithmetic theorem. Floating-point costs and
gradients are numerically checked values, not outward-rounded interval proofs.
Report replay discrepancies, rounding/cancellation issues and all failures rather
than asserting a formal real-arithmetic certificate for unverified machine
numbers. This diagnostic adds no physical interaction, model fit or optimization
solve. Runtime reflects arithmetic verification only.

## Necessary counterexamples and interpretation limits

For nonconvex f(x)=-x^2 on [-0.5,0.5], x=0 has gradient zero but gap 0.25; convexity
is essential. A stale gradient from the optimum of (x-0.5)^2, reused at x=-0.5,
would falsely report zero gap instead of one; exact selected-point provenance is
essential. A perfectly optimized learned loss (u-0.5)^2 gives zero modeled gap
while a physical loss (u+0.5)^2 is one at that command; model correctness is a
separate requirement. A small absolute bound need not be small relative to a
near-zero objective. First-order bounds can be loose even for convex quadratics,
and an outward normal gradient at a correct box optimum need not itself be small.
Thus solver status, raw gradient size and actual control success are not the
same evidence as this fixed modeled objective bound.
