# After the tolerated payload: change the uncertainty, not the failure threshold

Date: 2026-10-20. Independent competing framings, not an implementation prescription or claimed result.

The frozen MPC retaining7/8 post-event precision checks under doubled payload is a useful premise rejection. It does not justify making the load larger until adaptation wins. Fully actuated equilibrium regulation allows substantial model error; an energy-shaped fixed feedback law can regulate many positive inertias without identifying them. We should now choose a missing capability whose relevance is independently motivated.

## Underactuation: a real temporal-decision gap

Gravity plus underactuation introduces action sequences that first move away from the goal to transfer energy and later capture the desired configuration. Current v15 uses mild fully actuated goal offsets, so it has not tested that nonlocal temporal coordination. This is a more consequential capability than another calibration improvement.

But a short-horizon local-MPC failure would primarily demonstrate a horizon/optimizer deficiency. Raising task difficulty until that planner fails repeats the scalar optimizer detour. A fair experiment needs strong conventional energy-shaping/capture and trajectory-planning competitors, common observation access, a physically justified fixed task distribution and actual learning comparisons. A generic PD baseline is not adequate. Existing benchmark software also supplies conventions, sometimes true velocities, clipping and termination conditions that must not silently define the scientific question.

The falsifiable question would be whether learned experience can construct useful long action sequences for unseen goals/initial conditions without a supplied energy-transfer law. Direct episode/segment reuse and learned predictive planning are genuinely different formulations; an explicit energy-shaped controller is a privileged structural explanation or declared conventional prior. Underactuation does not itself prove learning is necessary. Choose this route if the intended gap is action sequencing, not because its failure is easier to obtain.

## Sensing: distinguish lost information from a wrong basis

An invertible sensor rotation is a weak next test: squared Euclidean geometry/noise are unchanged, and an isotropic history kernel can already be exactly rotation-equivariant. A hardcoded Fourier basis failing under that rotation would mostly expose its known representation dependence. More general invertible transforms also alter approximation complexity, noise and task metrics. Avoid presenting such predictable failure as discovery of general representation learning.

A stronger minimal change is to observe ONLY Cartesian end-effector positions, as with a camera, rather than both joint encoders. This is a physically natural noninvertible observation, not an arbitrary scrambling designed against a basis. Two elbow configurations can have identical stationary observations and goals while requiring different torque consequences. More offline data does not identify which hidden branch occupies the current stationary history; actions and subsequent observations can.

For unit links, let qplus=(-theta/2,theta) and qminus=(theta/2,-theta), with0<theta<pi. Both have end position(2*cos(theta/2),0). At rest, inertia is identical because it depends on cos(theta). Their endpoint Jacobians are

    Jplus=[[0,-sin(theta/2)],[2*cos(theta/2),cos(theta/2)]],
    Jminus=[[0,+sin(theta/2)],[2*cos(theta/2),cos(theta/2)]].

Thus leading Cartesian pulse responses are .5*T^2*Jplus/minus*M^-1*u. Radial acceleration changes sign between branches; tangential acceleration does not. No shared first pulse can produce the same NONZERO radial leading response in both cases. Each individual map is nonsingular, so sufficiently small radial goals are individually locally feasible. Under an equal branch prior, the mean leading radial action gain is zero, although neither real body lacks radial control authority. Averaging alternatives can erase an actionable possibility.

These are local small-pulse facts, not a global impossibility theorem for a common feedback strategy or a long open-loop sequence. A subsequent nonzero command may simultaneously pursue the goal and reveal the branch; explicit separate probing is not automatically needed. Noise and finite sampling require actual experiments. The distinction is task-relevant: a tangential goal need not require resolving the same ambiguity as a radial goal.

The smallest substantive test would retain one physical body, legitimate indistinguishable stationary camera histories and common mixed-branch position/action acquisition. Evaluate a few new Cartesian goals in sustained feedback, with no branch labels or hidden q/v to ordinary methods. Compare an ordinary generic history predictor/controller against a strong online local-Jacobian/visual-feedback method and, only if justified, a mechanism retaining alternatives or valuing an informative action. Charge all action/sensing time and retain opposite-direction first moves. Do not predeclare a probabilistic world model as mandatory; a direct history policy can encode the same useful distinction.

The scientific question is whether acquired knowledge can preserve and resolve genuinely decision-relevant alternatives that a conditional average loses. This is different from wrong-basis fitting and from tolerating a changed positive inertia. Strong generic approximators and a serious visual-servoing literature review are required before claiming a learning advance; conventional online Jacobian learning may suffice.

## Recommendation

For the most direct first-principles next uncertainty, I favor the noninvertible-sensing formulation: it changes what experience CAN know before action, not merely optimization difficulty, and removes supplied joint-coordinate semantics for a robotics-relevant reason. Underactuated sequencing is an equally legitimate but larger capability test; it should follow a deliberate question about nonlocal actions and strong baseline availability. Neither route warrants a milestone until an actual learner, useful decisions and independent audits exist. Choose one bounded investigation and let conventional sufficiency remain a possible answer.
