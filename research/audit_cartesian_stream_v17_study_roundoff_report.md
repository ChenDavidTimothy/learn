# Preserved coefficient replay failure and numerical resolution

2026-10-20. This is an AUDIT-criterion correction, not a controller/model/test
repair. Every canonical study source, observation, action, prediction and outcome
remains unchanged. Poor physical robustness is not removed or explained away by
this numerical finding.

## Original failure and durable evidence

The original independent auditor used coefficient `rtol=2e-8, atol=2e-10`.
It passed200–205 and207 but stopped at206/frozen episode7, renewal88, member1:
maximum independent-SVD versus saved-normal coefficient difference
2.1455866250352784e-10, narrowly exceeding the absolute criterion for a small
coordinate. The original source is preserved byte-for-byte as
`audit_cartesian_stream_v17_study_coefficient_tolerance_pilot.py`, SHA256
`b5ee98183d55d304b3141c7949cea74c3ecfb9e2e95b4f2b3972c19f767ffce6`.
The exact failed tuple/tolerance/assertion is retained in
`audit_cartesian_stream_v17_study_coefficient_tolerance_failure.json`.

Reproduce the independent diagnostic with
`.venv/bin/python research/audit_cartesian_stream_v17_study_roundoff.py`.
Its actual results and complete weighted design/targets, saved/reference
coefficients, ordinary prefix, selected plan, predictions and gradients are in
`audit_cartesian_stream_v17_study_roundoff_results.json` and `_inputs.npz`.
All parent sources/results/data hashes are bound. Existing project mpmath was
used; no dependency/environment change occurred.

## Independent resolution

Reconstructing the source's exact floating-point normal equations reproduces the
saved coefficient array exactly. A third pivoted-QR solve and an80-decimal-digit
weighted ridge least-squares calculation independently agree with SVD to within
4.45e-13. The saved normal solve differs from the high-precision optimum by
2.1456439402989247e-10. The augmented design condition number is1637.7081 and
the normal-matrix condition number is2,682,087.86. Thus the coordinate discrepancy
is real floating-point normal-equation roundoff, not proof that the independent
solution or source semantics differ.

The80-digit calculation forms the declared weighted objective directly from the
binary64 design, weights, targets and ridge. Its exact quadratic objective gap
for the saved coefficients is approximately5.08263035e-22. The weighted augmented
fitted-value difference norm is2.25447e-11. Subtracting rounded double objective
values would obscure this tiny gap; the diagnostic instead evaluates the
positive quadratic difference at high precision. The independent target
arithmetic itself differs by at most2.85e-14 in normalized coordinates; canonical-
target SVD/QR separately establish the same conclusion.

Downstream calculations were checked SEPARATELY on the actual prepared shifted,
zero, initial and selected plans. Largest path difference is6.81e-11, largest
cost difference6.37e-9 at cost about1136 (relative5.7e-12), and largest gradient
difference2.67e-11. These satisfy the ORIGINAL combined absolute/relative
downstream criteria; none was relaxed.

## Explicit secondary criterion

Final ordinary auditor source SHA256 is
`507d3bd427a661a57df07314f10f083bc55d35435225253a250d962b083989d7`.
It retains the original coefficient test FIRST. Only a failed original test may
invoke a secondary certificate; every invocation and its full matrices/solutions
are logged. There is no quiet global widening.

The secondary path requires exact source normal-equation replay, agreement of a
third pivoted QR with the independent SVD under the original coordinate test,
and source backward-error/conditioning accounting. For the computed normal
matrix N and approximate inverse Z, it bounds the inverse norm using
`||N^-1||inf <= ||Z||inf/(1-||I-ZN||inf)`, with a conservative floating residual
evaluation allowance. The difference satisfies
`N*(C_saved-C_reference)=r_saved-r_reference`, yielding a residual/inverse
coordinate-error bound. The preserved case has source backward error1.14e-18,
inverse residual upper bound1.05e-7 and a conservative coordinate-error bound
8.23e-7. This last bound is intentionally much looser than the actual2.15e-10
discrepancy; it is not a new claimed parameter-accuracy measurement.

The standard floating-error accounting is not interval arithmetic or a general
hardware-independent exact proof. The independent80-digit fixture, exact source
replay, third solver and separately unchanged downstream checks supply additional
evidence in the actual case. The auditor validates the diagnostic's hashes and
preserves its original source/failure. All eight corpus audits are rebound and
replayed under the final source before aggregate hash gates may pass.

No study action, target, fitting menu, optimizer, sample, success window or failure
was changed. The diagnostic resolves implementation replay at declared numerical
precision; it does not establish good model predictions, robust physical control,
global optimizer convergence or general learning capability.
