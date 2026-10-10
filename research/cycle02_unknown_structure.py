"""Cycle 02: learn unknown Boolean mechanisms, adapt to unannounced locations.

No model receives the true rule or the identities of shifted outputs.
Observations are full noisy binary output vectors for chosen binary inputs.
For every model, pretraining examples and training budget are matched.
"""
import itertools
import json
import math
import os
import random
import statistics
import time
import numpy as np

D, OUT, CHANGED = 10, 6, 2
NOISE = 0.05
PRE, POST, TRIALS = 96, 48, 24
CHECKPOINTS = (0, 8, 24, 48)
METHODS = ("bayes_active", "bayes_random", "bayes_reset",
           "bayes_noreset", "bayes_oracle")
SEED0 = 3102026

def make_grammar():
    x = np.arange(2**D, dtype=np.uint16)
    bits = ((x[:, None] >> np.arange(D)) & 1).astype(np.uint8)
    labels = []
    values = []
    for s in (1, 2, 3):
        for index in itertools.combinations(range(D), s):
            labels.append(("parity", index, ()))
            values.append(np.bitwise_xor.reduce(bits[:, index], axis=1))
    parity_count = len(labels)
    for s in (1, 2, 3):
        for index in itertools.combinations(range(D), s):
            for mask in itertools.product((0, 1), repeat=s):
                labels.append(("conj", index, mask))
                vals = (bits[:, index] ==
                        np.array(mask, dtype=np.uint8)).all(axis=1)
                values.append(vals.astype(np.uint8))
    matrix = np.array(values, dtype=np.uint8).T.copy()
    prior = np.concatenate([
        np.full(parity_count, 0.5/parity_count),
        np.full(len(labels)-parity_count, 0.5/(len(labels)-parity_count)),
    ])
    return matrix, prior, labels

GRAMMAR, PRIOR, LABELS = make_grammar()
XALL = ((np.arange(2**D)[:, None] >> np.arange(D)) & 1).astype(np.float32)
EPS = 1e-8
ENTROPY_NOISE = (-NOISE*np.log(NOISE) -
                 (1-NOISE)*np.log(1-NOISE))

def rule_world(rng, condition):
    """Learn pre-change rules, then change CHANGED hidden output rules."""
    old, new = [], []
    changed = np.array(rng.sample(range(OUT), CHANGED))
    if condition == "in_class":
        # Half generated from each grammar family, matching the Bayes prior.
        p = len([r for r in LABELS if r[0] == "parity"])
        def sample_rule():
            return rng.randrange(p) if rng.random() < 0.5 else rng.randrange(p, len(LABELS))
        old_ids = [sample_rule() for _ in range(OUT)]
        new_ids = old_ids.copy()
        for j in changed:
            while True:
                candidate = sample_rule()
                if candidate != new_ids[j]:
                    new_ids[j] = candidate
                    break
        old = GRAMMAR[:, old_ids].T.copy()
        new = GRAMMAR[:, new_ids].T.copy()
    elif condition == "novel_parity4":
        # All prechange rules are within grammar; new changed rules are not.
        p = sum(rule[0] == "parity" for rule in LABELS)
        def old_rule():
            return rng.randrange(p) if rng.random() < 0.5 else rng.randrange(p, len(LABELS))
        old_ids = [old_rule() for _ in range(OUT)]
        old = GRAMMAR[:, old_ids].T.copy()
        new = old.copy()
        for j in changed:
            coords = rng.sample(range(D), 4)
            new[j] = np.bitwise_xor.reduce(XALL[:, coords].astype(np.uint8), axis=1)
    elif condition == "dense_threshold":
        # Out-of-class negative control: dense random linear thresholds.
        def create_rule():
            coef = np.array([rng.uniform(-1.0, 1.0) for _ in range(D)])
            intercept = rng.uniform(-0.5, 0.5)
            return (XALL @ coef > np.sum(coef)*0.5 + intercept).astype(np.uint8)
        old = np.array([create_rule() for _ in range(OUT)])
        new = old.copy()
        for j in changed:
            new[j] = create_rule()
    else:
        raise ValueError(condition)
    return old, new, changed

def observe(world, index, rng):
    truth = world[:, index]
    noisy = (rng.random(OUT) < NOISE).astype(np.uint8)
    return np.bitwise_xor(truth, noisy)

def predictive_prob(weights, indices):
    # Returns probabilities of noisy outputs for each index, shape [out, n].
    truth_prob = weights @ GRAMMAR[indices].T
    return NOISE+(1-2*NOISE)*truth_prob

def loss_for(weights, world, eval_indices):
    p = np.clip(predictive_prob(weights, eval_indices), EPS, 1-EPS)
    q = NOISE+(1-2*NOISE)*world[:, eval_indices]
    xent = -(q*np.log(p)+(1-q)*np.log1p(-p))
    return xent.mean(axis=1)-ENTROPY_NOISE

def posterior_update(weights, index, y):
    predicted_rule = GRAMMAR[index][None, :]
    factor = np.where(predicted_rule == y[:, None], 1-NOISE, NOISE)
    new = weights*factor
    denom = new.sum(axis=1, keepdims=True)
    return new/denom

def choose_query(method, weights, rng):
    if method == "bayes_active":
        candidates = rng.integers(0, len(XALL), size=64)
        probs = np.clip(predictive_prob(weights, candidates), EPS, 1-EPS)
        ent = -(probs*np.log(probs)+(1-probs)*np.log1p(-probs))
        # One-step joint information gain = sum of output entropies
        # only if output hypotheses have factorized posteriors.
        return int(candidates[np.argmax(ent.sum(axis=0))])
    return int(rng.integers(0, len(XALL)))

def one_trial(seed, condition, output_datasets=False):
    rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    old, new, changed = rule_world(rng, condition)
    train_x = numpy_rng.integers(0, len(XALL), size=PRE)
    train_y = np.array([observe(old, int(x), numpy_rng)
                        for x in train_x], dtype=np.uint8)
    heldout_x = (np.arange(len(XALL)) if condition == 'novel_parity4'
                 else np.random.default_rng(seed + 998800).choice(
                     len(XALL), size=512, replace=True))
    weights_old = np.tile(PRIOR, (OUT, 1))
    for x, y in zip(train_x, train_y):
        weights_old = posterior_update(weights_old, int(x), y)

    histories = {}
    for method_i, method in enumerate(METHODS):
        rng_method = np.random.default_rng(seed+777001+101*method_i)
        if method == "bayes_reset":
            weights = np.tile(PRIOR, (OUT, 1))
        elif method == "bayes_oracle":
            weights = weights_old.copy()
            weights[changed] = PRIOR
        elif method.startswith("bayes_") and method != "bayes_noreset":
            weights = ((1-CHANGED/OUT)*weights_old +
                       (CHANGED/OUT)*PRIOR[None, :])
        else:
            weights = weights_old.copy()
        records = {}
        for t in range(POST+1):
            if t in CHECKPOINTS:
                by_output = loss_for(weights, new, heldout_x)
                records[t] = {
                    "all": float(np.mean(by_output)),
                    "changed": float(np.mean(by_output[changed])),
                    "retained": float(np.mean(np.delete(by_output, changed))),
                }
            if t == POST:
                break
            ix = choose_query(method, weights, rng_method)
            # query noise uses a seed depending on time, input and trial;
            # matching queries get identical noisy outputs.
            observation_rng = np.random.default_rng(
                seed * 1000003 + (t+1)*10009 + ix*71)
            y = observe(new, ix, observation_rng)
            weights = posterior_update(weights, ix, y)
        histories[method] = records
    if output_datasets:
        return histories, (train_x, train_y, old, new, changed, heldout_x)
    return histories

def summarize(condition):
    results = [one_trial(SEED0+i, condition) for i in range(TRIALS)]
    output = {}
    for name in METHODS:
        output[name] = {}
        for t in CHECKPOINTS:
            output[name][str(t)] = {
                key: float(statistics.mean(run[name][t][key]
                    for run in results))
                for key in ("all", "changed", "retained")
            }
        if name != "bayes_random":
            v = [run[name][POST]["all"] -
                 run["bayes_random"][POST]["all"] for run in results]
            output[name]["paired_final_vs_random"] = {
                "mean": float(statistics.mean(v)),
                "standard_error": float(statistics.stdev(v)/math.sqrt(TRIALS))
            }
    return output

def main():
    assert GRAMMAR.shape == (1024, 1335), GRAMMAR.shape
    assert math.isclose(float(PRIOR.sum()), 1, abs_tol=1e-9)
    start = time.perf_counter()
    results = {
        "protocol": "Cycle02 learned structure with exact grammar posterior",
        "config": {"D": D, "outputs": OUT, "changed": CHANGED,
                   "noise": NOISE, "pre": PRE, "post": POST,
                   "trials": TRIALS, "checkpoints": CHECKPOINTS,
                   "grammar_candidates": int(GRAMMAR.shape[1]),
                   "grammar_class_known_to_bayesian_baselines": True,
                   "change_time_known_to_all": True,
                   "identities_of_changed_outputs_unknown": True},
        "conditions": {condition: summarize(condition)
                       for condition in ("in_class", "dense_threshold", "novel_parity4")}
    }
    results["time_seconds"] = time.perf_counter()-start
    out = "/home/dtdav/dev/learn/research/cycle02_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Result path:", out)
    print("Runtime seconds:", round(results["time_seconds"], 2))
    print("Grammar", GRAMMAR.shape, "pre", PRE, "post", POST,
          "trials", TRIALS)
    for condition, report in results["conditions"].items():
        print("\nCONDITION:", condition)
        print(f"{'method':20} {'t=0':>10} {'t=8':>10}"
              f" {'t=24':>10} {'t=48':>10} {'chg48':>10}"
              f" {'ret48':>10}")
        for name, rec in report.items():
            print(f"{name:20} " +
                  " ".join(f"{rec[str(t)]['all']:10.5f}"
                           for t in CHECKPOINTS) +
                  f" {rec['48']['changed']:10.5f}"
                  f" {rec['48']['retained']:10.5f}")

if __name__ == "__main__":
    main()
