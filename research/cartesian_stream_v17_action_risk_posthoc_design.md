# Exploratory ordinary-data action-risk diagnostic — frozen before calculation

2026-10-20. AFTER the selected heldoutfork negative; original controlpilot is
running unchanged. This is explicitlyPOSTHOC research onrep0, not a new learner,
candidate selector, confirmatory test or release. No action-risk values have been
calculated yet. Do not change the deployedmodels or their frozen validation
selection from this diagnostic. Freshprotocol/corpora are needed for any new
selection/learning claim. Read rootaction_risk_question.md and independentaction_
moment note first; the simpler independent auxiliary-action identity is used.

Question: do the existing candidates' ordinary one-block prediction risks and
action-centered risks suggest different finite-class tradeoffs? This can test a
possible failure explanation using ordinary data alone; it cannot establish
information sufficiency, individual response accuracy or closed-loop benefit.

Onlytraining/validationordinaryrecords, savedfitmetadata/weights and canonical
ordinarypredictionmath are allowed. No evaluatorstates, forktruth or currentcontroloutcomes enter the diagnostic.
The combined ordinaryNPZ must decompress its storedobservations; testrows are
masked immediately and no testrow contributes toanycalculation. Fit eightcandidates staysunchanged;
allfourNNcandidates(eachthreemembers) andfourlocalcandidates are scored without
retuning. Use960validationrenewals t16,18,...,254 (notonlythe80sixblockorigins).
This differs from the original horizon, so report BOTH ordinaryoneblock risk and
actionrisk ontheseEXACTSAMErows; don't attribute a horizon change to centering.

At each validationtrajectory generate120x8x2physical auxiliarycommandsV uniform
[-.5,.5]^2 fromSeedSequence[20261020,17,680,0,1,trajectory_index]. Sharetheseeight
virtualcommands across ALLcandidates/members. Theyaremodelqueries,no newphysical
camera outcomes, not interventions/resets. SaveallactualV/predictions. At the
sameZ, evaluatecandidate mhat(Z,U_actual) andmhat(Z,V_j), futurepositioncorrections
tomodel'ssamepastmeasurable linearextrapolator. For eachmember, record

    S=.5*mean_coordinate((mhat(U)-mhat(V))²)
         -2*mean_coordinate((Y-dhat(Z))*(mhat(U)-mhat(V))).

Average overauxiliaries,thenmembers,thenrowswithintrajectory,then8trajectories.
Also record the score for the three-membermeanpredictions. The pairwiseidentity
gives an unbiased centeredconditionalmeanrisk up tocommonconstant forfixedmodels
andindependentvalidation; it DOESNOT give postselectionvalidity after usingthese
values tochoose/tunemodels. Scorezero for acommand-independentpredictor is an
exactdescriptivereference, not proofthattrueeffectsarezero. Scoresmaybenegative.

Two PREDECLAREDdriftbaselines: (a)zero (inresidualtargetunits), (b)the meanzero-
commandprediction overALL12NNfits(fourcandidates×threemembers), fittedonlyontrain.
Neither is chosen by action-risk/test outcomes. Bothhave sameexpectation but may
have verydifferentvariance; a badbaseline mayincreasevariance. Showboth. Use
physicalmetercorrectionunits/fourcoordinateaverages consistently. No synthetic
derivative, target B or suppliedbranch enters. Forlocalfits freezecoefficients
once peractualcontext acrossallninecommands; noappendofvalidationoutcomes.

Savepertrajectoryandperrow scores/predictions, clusterSEs over8trajectories,
sharedauxiliaryseedpaths, source/ordinary/fit/weights/designhashes andcounts.
AnytypicalSE is descriptive, notIIDroworindependentbody inference. Original
candidatechoices already used thesevalidationtrajectories; thisposthocdiagnostic
is not independent confirmation. Fixedmathematicalidentities donotpromise low
variance, properpolicytransfer or distributionalknowledge. Comparepairedscore
differences iflaterinterpreting, notmarginalSEs alone. No publishedmilestone for
another lossidentity or table without usefullearneddecisions.
