# Independent payload-instrument physics, information and direction audit

Date: 2026-10-20. Independently derived mechanics/information statements before reading outcomes; then checked the retained pilot's limited numerical interpretation. No canonical source or settings were edited. This is not an adaptive-learning or v16 milestone.

## Mechanics derived from the body

With unit lengths, first mass1 at the first link endpoint, second point mass m at the distal endpoint, and relative second angle, distal speed squared is

    v1^2+(v1+v2)^2+2*cos(q2)*v1*(v1+v2).

Kinetic energy is half the first mass speed squared plus half m times this expression. Collecting quadratic terms gives exactly

    M=[[1+2*m+2*m*c,m*(1+c)],[m*(1+c),m]], c=cos(q2).
    detM=m*(1+m*sin(q2)^2)>0 for m>0.

Euler-Lagrange gives h=m*sin(q2)*[-(2*v1*v2+v2^2),v1^2] and M*vdot+h=u. The source has the correct coefficients/signs and recovers the prior unit-mass mechanics. For constant mass, d(.5*v^T*M*v)/dt=v^T*u; energy changes by commanded torque work between events. These analytic statements support, but are not a substitute for, the separate artifact auditor's numerical replay.

For added distal mass dm, the inertia increment is dm*Jdist^T*Jdist, positive semidefinite. Its determinant after dividing out dm^2 is sin(q2)^2, so it is positive definite away from straight/folded configurations. Holding q and v continuous while changing mass gives

    momentum jump dp=(Mnew-Mold)*v,
    kinetic-energy jump dE=.5*v^T*(Mnew-Mold)*v
                           =.5*dm*||Jdist*v||^2.

This is a legitimate imposed matched-velocity attachment: the arriving payload/external apparatus supplies momentum/energy. It is not isolated-system momentum conservation or a passive unexplained mass jump. Incoming Cartesian payload velocity can match the attachment point so no velocity-reset impact is required. The code implements this interpretation consistently and does not claim energy conservation across the event.

## What can observations distinguish?

At exact rest and zero torque, every positive payload mass produces the same constant position forever. With identical sensor law, the observation distributions are identical. No change detector can announce a difference from those data alone. More passive samples cannot remedy that absence of information.

This does NOT mean every zero-torque trajectory is uninformative. At nonzero motion, M^-1*h generally depends on payload, so free evolution can reveal the change. At rest with constant probing torque, endpoint difference begins as .5*T^2*(Mnew^-1-Mold^-1)*u+O(T^4). Away from straight/folded configurations every nonzero u changes the leading acceleration under the two masses. At singular attachment geometry there can be uninformative action directions. Finite sensor noise can make small changes statistically hard despite noiseless distinguishability.

No ordinary action reads payload, change condition, true motion or future goal/noise. The parameter switch occurs after action selection at the first goal switch; the physical state remains continuous. Before that event, paired no-change/load trajectories and sensor streams coincide. Only four sensor paths are shared across all conditions/controllers, giving paired comparisons rather than80 independent noise experiments. The fixed parent model is replicate100 richH4 by protocol, not selected as best after seeing these outcomes.

The coincident goal/load switch is acceptable for this frozen-controller instrument: neither controller uses a goal-index-to-load rule. It would be inadequate for a claim of autonomously detecting unknown change times if an adaptive learner were designed knowing every second goal announces a load. A serious adaptive protocol must decouple change timing from disclosed task metadata or explicitly grant that prior to all competitors.

## Why load changes need not require identification

For either fixed positive mass and a constant goal g, consider ideal TRUE-velocity feedback

    u_j=-.5*tanh(k*(q_j-g_j))-.5*tanh(k*v_j), k>0.

It always respects |u_j|<1. Define V as kinetic energy plus sum_j log(cosh(k*(q_j-g_j)))/(2*k). Then

    Vdot=-.5*sum_j v_j*tanh(k*v_j)<=0.

For a fixed mass the kinetic metric is uniformly positive definite and the potential is radially unbounded in unwrapped position error. The largest invariant set with Vdot=0 is v=0,q=g, giving asymptotic regulation by the usual invariance argument. The action contains no payload estimate or inertia formula. At an imposed finite matched-velocity mass increase, V can jump upward but remains finite; afterwards the fixed-mass argument resumes.

Thus mass identification/adaptation is not mathematically necessary for eventual regulation in this ideal fully actuated gravity-free problem. This proof does not transfer directly to sampled/noisy polynomial velocity estimates, six-second deadlines, time-varying goals, global convergence rates or efficient action. The actual nine-setting PD bank is a finite diagnostic, not an optimization over every robust controller, so its failure cannot establish a need for learning. Conversely a frozen learned model's success is evidence that this disturbance is tolerated for the chosen task, not proof that payload never matters.

For m=2, ||M||infinity<=13 and ||h||infinity<=6*||v||infinity^2. The prior6sec rest-to-rest quintic with maximum joint displacement .35 has speed<=.109375 and acceleration<=.056133; required continuous torque is bounded by .80149<1. This supports slow post-loading rest-to-rest feasibility. It does not prove a .1sec-ZOH implementation, or recovery from every actual nonzero-velocity state at the event; those remain separate conditions.

## Actual limited outcome and premise rejection

Read retained results only after the independent derivation/source review. Frozen MPC achieves7/8 post-event precision opportunities in BOTH unchanged and loaded conditions. Post-event position MSE increases from .0107440728 to .0124800700, while mean per-goal late instantaneous velocity RMS decreases from .0079659600 to .0068941482. The complete nine PD settings and both conditions remain retained; no winner is promoted.

This rejects the premise that additional learning is required to retain this pilot's already imperfect7/8 precision count. It does not establish noninferiority, a statistical population robustness guarantee, equality of cost, or a ceiling on possible adaptive improvement. There is one independently acquired parent fit, four fixed tasks and four paired sensor paths; eight goal events per condition are dependent outcomes. No online learner, detector, forgetting mechanism or active probe was tested.

The80 physical-control evaluations cost15,040 simulator intervals/1504sec and30,240 scalar sensor readings, with only1512 distinct scalar noise draws reused for pairing. The38 mechanics flow calls add12.8sec. Reused parent acquisition is not new training. These are simulation resources, not hardware evidence or free adaptation data.

## Primary adaptive/dual-control literature reconciliation

Consulted after independently formulating the information/control problem:

- Slotine and Li (1987), *On the Adaptive Control of Robot Manipulators*: https://doi.org/10.1177/027836498700600303 . The publisher abstract describes PD feedback plus dynamics compensation with online unknown manipulator/payload parameters, without acceleration feedback or estimated-inertia inversion. The full publisher PDF was unavailable; do not claim its full proof was reviewed. The authors' publication list independently identifies the original: https://www.mit.edu/~nsl/ . This establishes classical context, not a position-only matched-budget performance comparison.
- Richards, Azizan, Slotine and Pavone (2021), *Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems*: https://arxiv.org/pdf/2103.04490 . It learns adaptive-control features using closed-loop tracking objectives rather than regression alone. Its development uses state/action trajectories and nominal structural dynamics, with fixed disturbances per training trajectory. This seriously challenges any inference from prediction improvement to adaptation performance, but its state/prior access is different from our position-only stream. Do not claim reproduction or superiority.
- Klenske and Hennig (2016), *Dual Control for Approximate Bayesian Reinforcement Learning*: https://arxiv.org/pdf/1510.03591 . The primary work studies non-episodic uncertain dynamics and actions that influence future observations, extending approximate dual control to regression models. Examples distinguish useful/irrelevant exploration and when to wait. The framework relies on specified probabilistic models, features and approximations; it does not imply that exploration is always beneficial or necessary in this payload task.
- Feldbaum (1960), *Dual Control Theory I*, original journal record: https://www.mathnet.ru/php/archive.phtml?jrnid=at&option_lang=eng&paperid=12624&wshow=paper . The journal's English abstract introduces the statistical closed-loop dual-control problem; full Russian proof was not reviewed. Treat this as historical provenance, not verification of a modern algorithm.

## Direction conclusion

Do not increase payload until the chosen frozen baseline fails. The limited negative instrument has served its purpose and supplies no adaptive-learning milestone. Ordinary robustness is a serious alternative explanation, especially for fully actuated equilibrium regulation. A new direction should change a consequential information or temporal-decision requirement, not tune this disturbance into an adaptive-method success story. The separate direction note compares underactuated action sequencing with genuinely noninvertible sensing, including stronger conventional alternatives and their confounds.
