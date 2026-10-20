# Independent v5 adversarial audits, 2026-10-20

Auditors: independent_theory (independent derivations and counterexamples), v5_adversarial_audit (independent implementation/replication). Neither edited canonical artifacts.

## Independent mathematical audit

- Derived H_u(i)=beta*rho^i,H_b=0 under centered independent white commands.
- Derived K_ij=eta*rho^i*r^(j-i-1) for j>=i+1, otherwise zero.
- Derived beta=H_u0,eta=K01,r=K02/K01,rho=K12/K01 within the supplied family.
- Derived commands-only affine-predictor interaction variance eta²*su²*sb²/[(1-r²)(1-rho²)]. Past-output ARX models are outside this bound.
- Derived shifted-mean repair H_b1=eta*mu_u,H_b2=eta*mu_u*(rho+r) at mu_b=0; higher-order statistics are not universally necessary.
- Verified stationary q,p/displacements have all finite moments; raw integrated position is not stationary and p is not Gaussian.
- Identified latent-scaling gauge: fixed unit gain b→q is supplied. Unknown gain and eta cannot be separately identified.

## Independent numerical audit

- Full script rerun to /tmp/coupled_response_v5_audit.json is byte-identical to saved canonical results.
- Independently recomputed all predictive means/Monte Carlo standard errors/quantiles and mean paired first-order differences.
- Separate impulse construction recovered K01=.7,K02=.35,K12=.315.
- Independent 400,000-step recursion: first-order residual RMS .5790438 versus theory .5792713.
- Independently derived exact finite-L oracle MSE (s is each channel SD):

  beta²*s²*rho^(2L)/(1-rho²) + eta²*s⁴*[r^(2L)/(1-r²)-rho^(2L)/(1-rho²)]/(r²-rho²).

  At L8,s=.8 this is .00007282217130721703, RMSE .00853359. Independent simulation RMSE .00851385; canonical 256-row check .00830627 is consistent.
- Position/action/noise indexing, train-only scaling, independent train/validation/test data, validation tuning, and kernel/ridge solvers passed inspection.

## Mandatory interpretation limits integrated before release

- Blindness is population first-order-summary blindness, not impossibility from full logged data. Higher moments, output variance and estimator fluctuation distributions can carry information.
- The quadratic model receives the correct polynomial family. Generic RBF KRR is one configured flexible baseline; no broad method superiority or autonomous operator-discovery claim.
- Centered-symmetry probes matter. Biased means can make lower-order responses informative.
- This is established Wiener/Volterra/BLA mathematics. Novelty and overall Goal completion are unsupported.

No unresolved implementation or derivation disagreement remains within the reported scope.
