# Independent adversarial theory audit for the proposed v9

Date: 2026-10-20 (Europe/Berlin). Auditor read AGENTS.md, the v8 scientific section, continuation_handoff.md and nonlinear_force_gauge_v9_pilot.py. This audit derives the identities independently; it is not independent numerical reproduction or a literature novelty assessment. No canonical source or historical ledger was edited.

## 1. Exact lagged equivalence and its restrictions

**PROVEN UNDER ASSUMPTIONS.** Let

    v[t+1] = f(v[t]) + m[t]
    m[t+1] = h(m[t], v[t], u[t]) + xi[t].

For any fixed function phi define q = m + phi(v), F = f - phi. The map (v,m) -> (v,q) is invertible without requiring phi to be invertible. Then

    v[t+1] = F(v[t]) + q[t]
    q[t+1] = h(q[t]-phi(v[t]), v[t], u[t])
             + phi(F(v[t])+q[t]) + xi[t].

This is causal: the argument of phi is the deterministic next velocity computed from the current state, not a future noisy measurement. Because xi[t] enters only m[t+1], the transformed innovation is the SAME additive xi[t], retaining its exogeneity under the same filtration. Position accumulation and observation errors can be coupled identically. Any command policy measurable with respect to the observed history and independent policy coins therefore issues the same commands inductively. This proves equality for every such policy, not merely the pilot's feedback policy.

Mapped initial laws are REQUIRED: q0=m0+phi(v0). At v0=m0=0 and phi(0)=0 both systems initialize at zero. Identical arbitrarily prepared nonzero velocities and independently certified zero total actuator force in each world do not in general map; they constitute additional information. Observing or resetting a known physical internal variable may also break the ambiguity.

The unchanged-noise claim is timing-specific. Direct mechanical process noise eta[t] in v[t+1] adds phi(F(v)+q+eta), generally coupling the transformed actuator innovation to eta. Motor-first timing also requires a fresh derivation; do not copy this h' or the additive-noise statement into that convention. The pilot uses lagged timing correctly.

For f(v)=rho*v-c*tanh(v), h=lambda*m+beta*u and phi=-c*tanh(v),

    F(v)=rho*v
    qnext=lambda*q+lambda*c*tanh(v)+beta*u
          -c*tanh(rho*v+q)+xi.

This matches the pilot algebra. It is a two-state, input-affine smooth realization with unchanged motor innovation, and q=m-c*tanh(v). Both drift laws contract toward zero when 0<c<rho<1. A generic nonlinear gauge need not preserve physical constraints such as known force bounds, calibrated current-to-force maps, velocity independence or independently certified zero-force semantics.

## 2. Coordinate freedom versus a physical counterexample

An invertible coordinate change alone does NOT prove two independently grounded physical force channels are different. Without a calibration or intervention giving the term 'motor force' physical meaning, the latent name is underdetermined.

There is, however, a concrete component-boundary interpretation of this example. World A has body drag -c*tanh(v) and a motor with active force/current state m. World B puts the same drag in the removable motor assembly while the body has only linear passive drift. The motor's total delivered force is q=m-c*tanh(v); its underlying active state evolves linearly. Both worlds give exactly the same connected body trajectories for all ordinary commands. They genuinely disagree after physical removal of the motor assembly: the nonlinear drag remains attached to the body only in A. Writing B in coordinates (v,q) yields the displayed two-state dynamics. No noncausal actuator or additional hidden state is necessary.

This interpretation must be declared, not inferred from the abstract gauge. Different component boundaries, real removal mechanisms or residual drag can change the intervention definition. A known total-force sensor distinguishes m from q. A sensor measuring only active current may not locate unmeasured shaft/body drag. Likewise u=0 does not force m=0, and m=0 does not force q=0 when v!=0. Any intervention proof must specify which of these quantities is actually fixed.

## 3. Restricted linear-actuator identification

**PROVEN UNDER ASSUMPTIONS.** Restrict the actuator to be velocity independent and exactly linear, and fix tanh's argument scale. Eliminating m under lagged timing gives

    vnext = (rho+lambda)*v - rho*lambda*vprev
            -c*tanh(v) + lambda*c*tanh(vprev)
            +beta*uprev + xiprev.

Motor-first timing has current u and xi instead. With exact velocity and full regressor rank, the coefficients of [v,vprev,tanh(v),tanh(vprev),uprev] are unique. For c!=0, lambda=-coefficient[tanh(vprev)]/coefficient[tanh(v)], rho=coefficient[v]-lambda and c=-coefficient[tanh(v)]. The remaining linear lag coefficient supplies a consistency restriction. This is identification within the supplied family, not evidence that real actuators are velocity independent.

At c=0 only the unordered poles are determined (unless they coincide). Full rank is not automatic under a policy, and noiseless recovery does not establish statistical usefulness. At velocities |v|<=A<<1, tanh(v)=v-v^3/3+O(A^5), so the nonlinear columns differ from linear columns only at order A^3. Separating rho from c and labeling the poles is then poorly conditioned. Under a simple homoscedastic regression approximation, the label-specific signal is of order c*A^3 and information can be of order N*c^2*A^6/noise_variance; this is an illustrative local scaling, NOT a proven rate for the closed-loop noisy-position experiment. A is actual velocity range, including process excitation, not merely command amplitude. Report singular values, residual nonlinear energy, coefficient errors and failures, rather than a ratio alone.

With noisy position, applying tanh to measured displacement creates nonlinear errors-in-variables bias. Exogenous instruments do not automatically remove this bias: the conditional measurement-error correction depends on the latent velocity. The linear IV proof cannot be imported. A full observation-model likelihood/state estimator, replicated/calibrated sensor correction, or a separately justified nonlinear moment construction is needed. A supplied-family state-space fit is a useful comparator but does not test actuator-family discovery.

## 4. Scope, corrections and research direction

No algebraic error was found in the pilot's gauge. Its output is a numerical illustration, not a general-model discovery algorithm. It does not explicitly accumulate positions, but pathwise velocity equality plus shared initial position and observation errors implies position equality analytically. The two feedback arrays are independently computed from their corresponding velocity histories; they are not forcibly shared. Initial D0 uses the stated special zero reference convention. These details should be made explicit if publishing the pilot as observed-position evidence.

**REJECTED inference:** nonlinear passive curvature universally reveals physical force location from command/position. It does so only after enough physically justified restrictions on alternative actuator models. **OPEN:** which such restrictions can an agent establish using feasible interventions rather than experimenter-supplied labels?

There is substantial drift risk in collecting more variants of the same gauge. The consequential next question is task relative: which distinctions between behaviorally equivalent hypotheses affect an actual transfer/control demand, and what smallest feasible experiment resolves those distinctions? A focused constructive study should learn the common connected behavior from experience, preserve uncertainty about component location, and measure ordinary-command prediction/control separately from component removal/replacement transfer. If an extra intervention is available, compare its information and cost against more ordinary data under the same budget. This would test whether the negative result changes the construction of a learner, rather than indefinitely adding parameter-identification theorems. No general learning breakthrough follows from the present identity.
