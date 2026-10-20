# V16 preliminary instrument: does a payload change require new learning?

Frozen before these outcomes. This is a bounded diagnostic, not a new milestone.
Do not presume adaptation, a detector or active probing is needed: robust fixed
feedback or the frozen learned history relation may already solve the task.

Use the same ideal horizontal two-link point-mass mechanics, now m1=1 and distal
point mass m2 variable. From kinetic energy with unit lengths:

    M=[[1+2*m2+2*m2*cos(q2), m2*(1+cos(q2))],
       [m2*(1+cos(q2)), m2]]
    h=m2*sin(q2)*[-(2*v1*v2+v2²), v1²].

Unit m2 exactly recovers v14/v15. At the first goal switch (6sec after8sample
warmup), either keep m2=1 or change to2, without a reset or controller announcement.
Coordinates/velocity are continuous: interpret the external attachment as matched-
velocity payload loading, with external energy/momentum accounting at the event.
Do not claim energy conservation across this imposed parameter jump. Between jumps
the same torque-work identity holds. At exact rest and zero torque, BOTH plants
produce exactly constant positions: an earlier alarm is information-theoretically
unavailable and unnecessary until actions make the difference consequential.

One fixed parent acquired-data model: v15 replicate100 richH4, chosen by protocol,
not best among eight. It is frozen throughout. Four starting contexts and same
three goals/6sec dwell as fresh v15. Fresh role-tagged noise SeedSequence
[20261020,16,620,0,1,task], SD.001, paired across payload conditions/controller
variants. Every ordinary action sees current goal/noisypositions/issuedcommands
only, no m2/change label/truev/futuregoal. Parent acquisition is reused, not recounted
as new data. Model/planner formulas are the already audited v15 formulas.

Compare frozen parent MPC against the full fixed nine-setting tanh-PD diagnostic
bank from v15: windows3/5/9 x gains2/4/8, all outcomes retained, no winner selected.
PD sees causal fitted current position/slope only. This bank is not globally
optimized and its failure cannot prove learning is necessary; success may expose
an unnecessary adaptation challenge. No new online learner/probes are evaluated
until this basic change/feasibility instrument is inspected independently.

Retain all trajectories, noise roles, actions/plans and position/velocity/precision
metrics with pre/post-change periods separate. Count all simulator intervals and
reads. Test several arbitrary q/v/u cases for unit-parent equality and torque-work
conservation, and independently audit variable-mass mechanics/representative paths.
Continuous rest-to-rest references after loading can still meet torque bounds:
||M||inf<=13, ||h||inf<=6||v||inf² for m2=2; the prior6sec/.35rad quintic bound
is13*.056133+6*.109375²<.802. This does not certify sampled-command endpoints or
every encountered nonzero-velocity recovery. Preserve all failures and be ready to
reject the premise that this change demands an adaptive representation.
