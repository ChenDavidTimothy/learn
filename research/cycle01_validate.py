"""Validation and paired comparisons for Cycle 01. Python standard library only."""
import math
import statistics

from cycle01_sparse_repair import (
    ACTIONS, EPS, HYPOTHESES, K, M, MixtureExact,
    SparseExact, trial
)

def h2(p):
    return -p*math.log2(p)-(1-p)*math.log2(1-p)

def validate_posteriors():
    base = [0]*ACTIONS
    sparse = SparseExact(base)
    assert len(HYPOTHESES) == math.comb(M, K)*2**K
    assert math.isclose(sum(sparse.weights), 1, abs_tol=1e-12)
    assert math.isclose(sum(sparse.marginals()), K, abs_tol=1e-12)
    prior = sparse.marginals()[0]
    sparse.update(0, True)
    analytic = prior*(1-EPS)/(prior*(1-EPS)+(1-prior)*EPS)
    assert math.isclose(sparse.marginals()[0], analytic, abs_tol=1e-12)
    assert math.isclose(sum(sparse.weights), 1, abs_tol=1e-12)
    assert math.isclose(sum(sparse.marginals()), K, abs_tol=1e-12)
    assert all(0 <= p <= 1 for p in sparse.predictions())

    mixture = MixtureExact(base)
    assert math.isclose(sum(mixture.weights), 1, abs_tol=1e-12)
    assert math.isclose(sum(mixture.marginals()), 3, abs_tol=1e-12)
    mixture.update(1, True)
    assert math.isclose(sum(mixture.weights), 1, abs_tol=1e-12)
    assert 2 <= sum(mixture.marginals()) <= 4
    assert all(0 <= p <= 1 for p in mixture.predictions())

def paired(condition):
    pairs = (
        ("structured_active", "structured_random"),
        ("mixture_active", "mixture_random"),
        ("mixture_active", "independent_active"),
        ("structured_active", "mixture_active"),
    )
    runs = [trial(20261010+i, condition) for i in range(80)]
    report = {}
    for method_a, method_b in pairs:
        diffs = [
            x[method_a][64]["all"]-x[method_b][64]["all"] for x in runs
        ]
        mean = statistics.mean(diffs)
        se = statistics.stdev(diffs)/math.sqrt(len(diffs))
        report[f"{method_a} minus {method_b}"] = (mean, se)
    return report

def main():
    validate_posteriors()
    print("POSTERIOR_VALIDATION: PASSED")
    n = len(HYPOTHESES)
    delta = .05
    lower_bound = (
        math.log2(n)-h2(delta)-delta*math.log2(n-1)
    )/(1-h2(EPS))
    print(f"N={n}, h2(epsilon)={h2(EPS):.6f}, "
          f"Fano fixed-budget bound for error <= 0.05: "
          f"T >= {lower_bound:.6f}")
    for condition in ("single", "coupled"):
        print("CONDITION:", condition)
        for label, (mean, se) in paired(condition).items():
            print(f"  {label}: delta={mean:.6f}, SE={se:.6f}, "
                  f"approx95pct=[{mean-1.96*se:.6f},{mean+1.96*se:.6f}]")

if __name__ == "__main__":
    main()
