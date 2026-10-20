# Pre-optimization derivative check correction

The first source execution stopped before fitting because its fixed central
finite-difference step1e-6 crossed a hard-clipping boundary in the bv coordinate.
The discrepancy was.331221, while the other five coordinates agreed to1.3e-8.
This is not a legitimate smooth derivative comparison at that step: one issued
command changes saturation active set between plus/minus perturbations.

Before any optimization results, instrumented the check to retain all discrepancies
and saturation active-set counts at1e-6,1e-7,1e-8. At the two smaller fixed steps
there are NO saturation active-set changes and all derivatives agree: max1.09e-8
at1e-7 and1.66e-7 at1e-8. Both are required to pass. No controller, derivative
formula, optimization budget or evaluation stream was changed. Full values are in
generated checks metadata. This records a corrected verification method, not a
hidden performance-tuning change or an excuse to ignore clipping.
