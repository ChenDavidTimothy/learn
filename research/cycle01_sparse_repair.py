"""Cycle 01: local change adaptation, an established Bayesian baseline, and model mismatch."""
import itertools
import json
import math
import platform
import random
import statistics
import time

M, K, EPS, STEPS, TRIALS = 16, 2, 0.10, 64, 80
ACTIONS = 2 * M
CHECKPOINTS = (0, 8, 16, 32, 64)
HYPOTHESES = [(2*i+a, 2*j+b)
              for i, j in itertools.combinations(range(M), K)
              for a in (0, 1) for b in (0, 1)]
HYPOTHESES_COUPLED = [(2*i, 2*i+1, 2*j, 2*j+1)
                      for i, j in itertools.combinations(range(M), K)]
HYPOTHESES_MIXTURE = HYPOTHESES + HYPOTHESES_COUPLED
ENTROPY_FLOOR = -EPS*math.log(EPS)-(1-EPS)*math.log(1-EPS)
METHODS = ("structured_active", "structured_random",
           "mixture_active", "mixture_random",
           "independent_active", "reset_random", "frozen")

def expected_excess(q, p):
    """Excess predictive cross-entropy (nats) over a noise-aware oracle."""
    p = min(1-1e-14, max(1e-14, p))
    return -q*math.log(p)-(1-q)*math.log(1-p)-ENTROPY_FLOOR

class SparseExact:
    """Exact posterior under exactly K modules each flipping one input entry."""
    def __init__(self, base):
        self.base = base
        self.weights = [1/len(HYPOTHESES)]*len(HYPOTHESES)

    def marginals(self):
        mass = [0.0]*ACTIONS
        for (a, b), w in zip(HYPOTHESES, self.weights):
            mass[a] += w
            mass[b] += w
        return mass

    def update(self, action, observed):
        weighted = []
        for (a, b), w in zip(HYPOTHESES, self.weights):
            bit = self.base[action] ^ int(action == a or action == b)
            py = (1-EPS) if bit else EPS
            weighted.append(w*(py if observed else 1-py))
        normalizer = sum(weighted)
        self.weights = [w/normalizer for w in weighted]

    def predictions(self):
        return noisy_predictions(self.base, self.marginals())

class MixtureExact:
    """Exact Bayes over two possible change regimes, without knowing which."""
    def __init__(self, base):
        self.base = base
        self.weights = ([0.5/len(HYPOTHESES)]*len(HYPOTHESES) +
                        [0.5/len(HYPOTHESES_COUPLED)]*len(HYPOTHESES_COUPLED))

    def marginals(self):
        mass = [0.0]*ACTIONS
        for hypothesis, w in zip(HYPOTHESES_MIXTURE, self.weights):
            for changed_action in hypothesis:
                mass[changed_action] += w
        return mass

    def update(self, action, observed):
        weighted = []
        for changed_actions, w in zip(HYPOTHESES_MIXTURE, self.weights):
            bit = self.base[action] ^ int(action in changed_actions)
            py = (1-EPS) if bit else EPS
            weighted.append(w*(py if observed else 1-py))
        z = sum(weighted)
        self.weights = [w/z for w in weighted]

    def predictions(self):
        return noisy_predictions(self.base, self.marginals())

class IndependentRepair:
    """Heuristic local selective update, ignoring exactly-K constraints."""
    def __init__(self, base):
        self.base = base
        self.flip_probability = [K/ACTIONS]*ACTIONS

    def marginals(self):
        return self.flip_probability

    def update(self, action, observed):
        bit = self.base[action]
        unchanged_y = (1-EPS) if bit else EPS
        flipped_y = EPS if bit else (1-EPS)
        same_l = unchanged_y if observed else 1-unchanged_y
        diff_l = flipped_y if observed else 1-flipped_y
        prior = self.flip_probability[action]
        self.flip_probability[action] = (
            prior*diff_l/(prior*diff_l+(1-prior)*same_l))

    def predictions(self):
        return noisy_predictions(self.base, self.flip_probability)

class ResetPredictor:
    """Diagnostic baseline: discard the old model and relearn each entry."""
    def __init__(self, base):
        self.counts = [0]*ACTIONS
        self.ones = [0]*ACTIONS

    def update(self, action, observed):
        self.counts[action] += 1
        self.ones[action] += int(observed)

    def predictions(self):
        return [(1+s)/(2+n) for s, n in zip(self.ones, self.counts)]

class FrozenPredictor:
    """Diagnostic baseline: never adapt."""
    def __init__(self, base):
        self.base = base

    def update(self, action, observed):
        pass

    def predictions(self):
        return [EPS+(1-2*EPS)*b for b in self.base]

def noisy_predictions(base, flip_probabilities):
    return [EPS+(1-2*EPS)*(1-p if bit else p)
            for bit, p in zip(base, flip_probabilities)]

def select_action(name, model, rng):
    if name.endswith("active"):
        masses = model.marginals()
        uncertainty = [p*(1-p) for p in masses]
        best = max(uncertainty)
        candidates = [i for i, v in enumerate(uncertainty)
                      if abs(v-best) < 1e-14]
        return rng.choice(candidates)
    return rng.randrange(ACTIONS)

def score(model, truth, flips):
    predictions = model.predictions()
    losses = [expected_excess(EPS+(1-2*EPS)*truth[a], predictions[a])
              for a in range(ACTIONS)]
    return {"all": sum(losses)/ACTIONS,
            "changed": sum(losses[a] for a in flips)/len(flips)}

def trial(seed, condition):
    rng = random.Random(seed)
    base = [rng.getrandbits(1) for _ in range(ACTIONS)]
    shifted_modules = rng.sample(range(M), K)
    if condition == "single":
        flips = {2*i+rng.randrange(2) for i in shifted_modules}
    elif condition == "coupled":
        flips = {2*i+j for i in shifted_modules for j in range(2)}
    else:
        raise ValueError(condition)
    truth = [b ^ int(a in flips) for a, b in enumerate(base)]
    noise = [[rng.random() < EPS for _ in range(ACTIONS)]
             for _ in range(STEPS)]
    factories = {
        "structured_active": SparseExact, "structured_random": SparseExact,
        "mixture_active": MixtureExact, "mixture_random": MixtureExact,
        "independent_active": IndependentRepair,
        "reset_random": ResetPredictor, "frozen": FrozenPredictor,
    }
    result = {}
    for midx, name in enumerate(METHODS):
        model = factories[name](base)
        policy_rng = random.Random(seed*100003+midx+771)
        observations = {0: score(model, truth, flips)}
        for t in range(1, STEPS+1):
            action = select_action(name, model, policy_rng)
            observed = truth[action] ^ noise[t-1][action]
            model.update(action, observed)
            if t in CHECKPOINTS:
                observations[t] = score(model, truth, flips)
        result[name] = observations
    return result

def analyze(condition):
    runs = [trial(20261010+seed, condition) for seed in range(TRIALS)]
    summaries = {}
    for method in METHODS:
        summaries[method] = {}
        for t in CHECKPOINTS:
            summaries[method][str(t)] = {
                criterion: statistics.mean(run[method][t][criterion]
                                           for run in runs)
                for criterion in ("all", "changed")
            }
        if method != "structured_random":
            diffs = [run[method][STEPS]["all"] -
                     run["structured_random"][STEPS]["all"]
                     for run in runs]
            summaries[method]["paired_final_difference_vs_structured_random"] = {
                "mean": statistics.mean(diffs),
                "standard_error": statistics.stdev(diffs)/math.sqrt(TRIALS)
            }
    return summaries

def main():
    before = time.perf_counter()
    results = {
        "description": "Expected excess log loss, nats/entry, checkpoint means",
        "design": {"m": M, "k": K, "epsilon": EPS, "trials": TRIALS,
                   "steps": STEPS, "checkpoints": CHECKPOINTS,
                   "hypotheses": len(HYPOTHESES),
                   "mixture_hypotheses": len(HYPOTHESES_MIXTURE),
                   "models_assume_one_changed_entry_per_changed_module": True,
                   "all_models_receive_same_prechange_truth_table": True,
                   "coupled_is_outside_structured_hypothesis_class": True},
        "python": platform.python_version(),
        "cases": {case: analyze(case) for case in ("single", "coupled")},
    }
    results["runtime_seconds"] = time.perf_counter()-before
    output = "/home/dtdav/dev/learn/research/cycle01_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Results:", output)
    print("Trials:", TRIALS, "steps:", STEPS,
          "hypotheses:", len(HYPOTHESES),
          "time:", round(results["runtime_seconds"], 2), "s")
    print("Expected excess log loss, nats per output coordinate")
    for case, methods in results["cases"].items():
        print("\nCONDITION", case, "FINAL STEP", STEPS)
        print(f"{'method':22} {'all':>12} {'changed':>12}"
              f" {'paired_diff':>12} {'SE':>10}")
        for name, entry in methods.items():
            dif = entry.get("paired_final_difference_vs_structured_random",
                            {"mean": 0, "standard_error": 0})
            print(f"{name:22} {entry[str(STEPS)]['all']:12.6f}"
                  f" {entry[str(STEPS)]['changed']:12.6f}"
                  f" {dif['mean']:12.6f} {dif['standard_error']:10.6f}")
        print("structured_active trajectory:",
              [round(methods["structured_active"][str(t)]["all"], 5)
               for t in CHECKPOINTS])
        print("structured_random trajectory:",
              [round(methods["structured_random"][str(t)]["all"], 5)
               for t in CHECKPOINTS])

if __name__ == "__main__":
    main()
