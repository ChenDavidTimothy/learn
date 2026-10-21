# Independent saved result check: PASS

The independent theory auditor implemented and executed
`check_amago_control_results.py` once after the completed results were available.
The executable and hash-bound JSON retain the exact checked scope. No environment,
metrics helper, learner or optimizer implementation was imported.

All eight AMAGO validation archives and the secondary fixed-rho RLS archive were
checked: nine batches of 64 bodies and 240 transitions, totaling 138,240 saved
transitions. Cost was reconstructed directly from post-transition joint angles,
velocities, issued actions and the applicable goal, using wrapped angle errors.
Precision was reconstructed from every post-transition sample in each last-20
window, observations 61 through 80, 141 through 160 and 221 through 240. All
saved costs, body means, mean costs and precision flags agree to the stated
numerical check. Failed goal indices remain in the JSON; no bodies were removed.
Initial positions and velocities, goals, evaluator body parameters and input
matrices match exactly across all nine archives.

Literal minimum validation cost across both learning rates and four checkpoints
selects learning rate .0003, epoch96: cost 1.2582925566861174, precision 1/192.
The .0001 fit's best checkpoint is epoch128, cost 1.363133653205706, precision
0/192. The 75% validation gate is false. There are no fresh-confirmation artifacts;
the gate and reviewed decision source prohibit sampling them. This absence is
saved-artifact and source evidence, not exhaustive operating-system tracing.

The secondary paired diagnostic uses rho .98 fixed from the earlier screen,
without new tuning: cost .3001656740886105 and precision 67/192. It is validation
data, not fresh confirmation. It shows substantially better conventional control
on these draws, but does not establish the 75% competence threshold is attainable.

Checks also cover the eight saved weight hashes, report and selection hashes,
current sources against prefit Git commit
14b87ac5fd855ac21c734c07020cc3b3cfc8a6a2, and every vendored official file against
the recorded upstream Git blob. Each fit has all 128 epoch records and exactly
50 recorded official training updates per epoch, with 241 sequence observations,
3,840 unmasked transition entries per replay minibatch, and finite logged values.
These are recorded-log checks, not an independent replay of optimization. The
official pinned core is unmodified; this configurable GRU experiment is not a
reproduction of full Transformer results or a general verdict on AMAGO.

Counts agree: 3,932,160 training interactions, 122,880 AMAGO validation
interactions, and 15,360 secondary diagnostic interactions. The total is
4,070,400, plus 33,600 discarded engineering interactions kept separate:
the two-epoch four-body smoke used 1,920 training and 960 validation interactions;
the full-batch profile used 15,360 training and 15,360 validation interactions.
Tiny mocked adapter checks add no physical transitions.

No independent neural forward passes, optimizer reconstruction or physical
integration were performed. The supported conclusion is failure of this bounded
competence attempt under its frozen settings. No additional audit or automatic
training rescue is warranted by this result.
