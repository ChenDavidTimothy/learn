# Prospective independent critique of saved-prefix consequence diagnostics

2026-10-20. This critiques the proposed fixed diagnostic before implementation or additional physics. No learner, diagnostic, fork or simulation was executed for this report. Original study/protocol/aggregator files remain unchanged.

## Judgment and simplest first step

The proposal is useful for separating errors hidden by an aggregate forecast score. It cannot alone identify their unique cause. Begin with the cheapest complete measurement: reconstruct the saved member one-block predictions at every actual renewal and compare them with the actual next two camera readings and true Cartesian endpoints already stored. There are 11,520 ordinary decision pairs across the fixed study. Use each decision's saved coefficients/query and training-only scales, not a library refitted with later observations.

Call this a prequential one-block diagnostic. Both positions were predicted together before either new reading arrived; do not insert the first actual future reading when evaluating the second prediction. That would change the observation contract. Report absolute errors, first versus second sample, phase/time profiles, fit-member dispersion and relation to task failures, with descriptive clustering by corpus. Future-camera MSE includes measurement noise; true-endpoint MSE removes future sensor noise but retains uncertainty from the noisy conditioning prefix. Their expected per-coordinate difference is the sensor variance under the declared independent noise law, not an equality for each realized pair.

This complete pass can reveal whether failures already occur in immediate predictions on visited states or arise mainly through imagined longer continuations/decision choices. It cannot separate passive drift accuracy from action-response accuracy because only the issued action was realized there.

## Fixed action forks: useful, with a cheaper equivalent

The proposed renewals [0, 15, 30, 60] sample warmup completion, midway through the first goal, and the two goal switches. Across eight corpora, two variants and eight cases these give 512 logical context records. They are prespecified rather than chosen for failures, but not 512 independent or necessarily distinct physical histories. In particular the two variants share the initial state/history/library at renewal 0. Further duplicates are possible. Retain the original count and paired clustering; count actual unique integrations/reuse separately.

The actual issued command's two true successor states already exist in the saved stream. Reuse those outcomes instead of reintegrating a sixth action at every context. Only the five command blocks {0, +/-.4e1, +/-.4e2} need new forks: at most 5,120 intervals/512 simulated seconds for all logical records before duplicate reuse. An actual-command reintegration can be a limited physics consistency check if desired, not new intervention evidence or ordinary learning experience. Do not exclude large/failed/singular contexts after inspecting them.

At each prefix, separate zero-command drift error from zero-centered command contrasts. Since current-context local coefficients are frozen and the current standardized context equals the query, the first-block model is affine in the proposed command. Its baseline and intercept cancel in a contrast. The symmetric action bank therefore also permits simple finite secants and an even-response diagnostic:

    odd_j = [F(X,+a e_j)-F(X,-a e_j)]/(2a),
    even_j = F(X,+a e_j)+F(X,-a e_j)-2F(X,0),  a=.4.

The affine predictor has zero even component. Actual nonzero even terms measure a finite-action restriction of that predictor at this world/scale; odd secants diagnose action directions/gain without conflating them with drift. Keep these as diagnostic decompositions, not a new fit, a tuned action radius, or proof about an unknown conditional mean. Goals and planners are not reselected from these results.

## The crucial conditional-mean limitation

Let X be the actual hidden physical state and H the ordinary noisy prefix. A fork yields a particular world's deterministic consequence F(X,u). A history predictor may target m(H,u)=E[F(X,u)|H], not F(X,u) itself. For a contrast Delta(X,u)=F(X,u)-F(X,0), squared error against that individual-world contrast decomposes, in expectation over the relevant conditional world law, into

    ||predicted_Delta(H,u)-E[Delta(X,u)|H]||^2
      + trace Var(Delta(X,u)|H).

A large fork error can therefore reflect irreducible remaining ambiguity, inadequate representation/fitting, unsupported context, or a wrong conditional uncertainty model. One true fork per world does not identify those pieces. Removing future camera noise does not remove hidden-state uncertainty conditional on H. Per-world secants are valid evaluator potential consequences; they must not be relabeled the conditional mean known to the learner.

The same caution applies to finding similar histories with different fork responses. That can show that the supplied distance/neighborhood merges practically different cases. It is not proof that the full ordinary prefix lacks the information: the histories may differ in a cue the chosen metric discards. Exact history insufficiency requires a separate indistinguishability or statistically justified information argument, not large residuals alone.

## Dispersion, support and policy comparisons

Compute contrast dispersion from coherent within-member differences m_k(H,u)-m_k(H,0). Summing independent marginal variances would discard their shared coefficients/intercept and misstate uncertainty. Three bootstrap members represent limited fit variation, not a calibrated physical posterior. Report empirical error-versus-spread behavior; do not invent a confidence/coverage guarantee after observing outcomes.

A large residual with close library neighbors is evidence against trusting this neighborhood's fit, not a certificate that the context was supported. A large distance with a bad action contrast is compatible with support shift but does not prove it caused failure. Distances, local command-design conditioning and residual/dispersion profiles can make these explanations more or less plausible. None uniquely separates them from history ambiguity.

The two policies visit different later prefixes. Comparing their aggregate prediction errors mixes differences in their predictors, visited states, actions and recovery difficulty. The diagnostic should retain that whole-policy interpretation. If a matched-prefix predictor comparison becomes necessary, a frozen acquired-library shadow can be refitted at a retained policy's saved prefix using the unchanged selected hyperparameters. That is a further declared diagnostic on existing data, not permission to tune the learner or infer an isolated per-row causal effect. It need not be added automatically before the simpler measurements are read.

## What the outcomes would justify next

Accurate zero-command forecasts with poor command contrasts would focus attention on acquiring useful action response, rather than more passive extrapolation. Poor zero-command forecasts on visited states would keep history, local drift and support problems live. Small short-term errors but poor sustained behavior would motivate examining long-horizon approximation, uncertainty or the declared task/validation objective. Error larger than bootstrap spread would show that this spread is not an adequate empirical error bound, without proving that a particular belief architecture is necessary.

The more consequential subsequent experiment could compare population reuse with a justified local/prefix estimate under the same available observations and time. The diagnostic should first identify what such an estimate would need to correct. Do not treat the prefix route, longer histories, a new metric or a probabilistic model as the answer in advance. This is a fixed explanatory investigation of an audited failure, not another learning milestone by itself.
