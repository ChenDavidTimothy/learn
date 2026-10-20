#!/usr/bin/env python3
"""Reproducible toy benchmark for first-principles robot learning, v1.

Uses only NumPy. Does NOT claim an agent discovers concepts from raw pixels:
its candidate expression grammar is explicitly supplied. The script verifies
(a) aliasing under position-only state, (b) expressible feature discovery,
(c) identifiability via excitation, (d) judgment-vs-action distinction, and
(e) translation-invariant goal planning.
"""
from __future__ import annotations

import itertools
import json
import numpy as np

RHO = 0.8
GAIN = 1.3
SEED = 20261020


def samples(rng: np.random.Generator, n: int):
    current = rng.uniform(-7, 7, n)
    velocity = rng.uniform(-3, 3, n)
    action = rng.uniform(-2, 2, n)
    previous = current - velocity
    next_velocity = RHO * velocity + GAIN * action
    next_position = current + next_velocity
    return previous, current, action, next_position


def fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def basis(which, previous, current, action):
    one = np.ones_like(current)
    delta = current - previous
    if which == "action_only":
        return np.column_stack([one, action])
    if which == "current_only":
        return np.column_stack([one, current, action])
    if which == "previous_history":
        return np.column_stack([one, current, previous, action])
    if which == "discovered_delta":
        return np.column_stack([one, delta, action])
    raise ValueError(which)


def feature_grammar(previous, current):
    # Each expression uses only measured positions and elementary arithmetic.
    d = current - previous
    return {
        "x_t": current,
        "x_(t-1)": previous,
        "x_t - x_(t-1)": d,
        "x_(t-1) - x_t": -d,
        "x_t + x_(t-1)": current + previous,
        "x_t * x_(t-1)": current * previous,
        "x_t^2": current**2,
        "x_(t-1)^2": previous**2,
        "abs(x_t - x_(t-1))": np.abs(d),
    }


def model_comparison():
    rng = np.random.default_rng(SEED)
    train = samples(rng, 200)
    test = samples(rng, 1000)
    y_train = train[3] - train[1]
    y_test = test[3] - test[1]
    results = {}
    for model in ["action_only", "current_only", "previous_history", "discovered_delta"]:
        coefficients = fit(basis(model, *train[:3]), y_train)
        predicted = basis(model, *test[:3]) @ coefficients
        results[model] = {
            "test_rmse": rmse(y_test, predicted),
            "coefficients": [float(x) for x in coefficients],
        }
    grammar_train = feature_grammar(*train[:2])
    grammar_test = feature_grammar(*test[:2])
    feature_scores = {}
    for name, feature in grammar_train.items():
        design_train = np.column_stack([np.ones_like(feature), train[2], feature])
        coef = fit(design_train, y_train)
        feature_out = grammar_test[name]
        design_test = np.column_stack([np.ones_like(feature_out), test[2], feature_out])
        feature_scores[name] = rmse(y_test, design_test @ coef)
    # A translational shift changes neither displacement nor learned model predictions.
    shift = 1000.0
    coef_delta = fit(basis("discovered_delta", *train[:3]), y_train)
    prediction_orig = basis("discovered_delta", *test[:3]) @ coef_delta
    prediction_shift = basis(
        "discovered_delta", test[0] + shift, test[1] + shift, test[2]
    ) @ coef_delta
    results["translation_test_max_abs_difference"] = float(
        np.max(np.abs(prediction_orig - prediction_shift))
    )
    results["grammar_single_feature_rmse"] = feature_scores
    return results


def identifiability_checks():
    # Two distinct experiments with invertible design, noiseless.
    design = np.array([[1.0, 0.0], [0.0, 1.0]])
    measurements = design @ np.array([RHO, GAIN])
    recovered = np.linalg.solve(design, measurements)
    # Entire trajectory with u=2*v cannot separate rho from gain.
    confounded = np.array([[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0]])
    return {
        "informative_design_rank": int(np.linalg.matrix_rank(design)),
        "estimated_rho_beta": [float(n) for n in recovered],
        "confounded_design_rank": int(np.linalg.matrix_rank(confounded)),
        "confounded_alternative_model": [float(RHO + 2 * GAIN), 0.0],
        "confounded_same_predictions": bool(np.allclose(
            confounded @ [RHO, GAIN],
            confounded @ [RHO + 2 * GAIN, 0],
        )),
    }


def judgment_checks():
    # Three experiments (x, v, a, next_v) under rho=.8 and gain=1.3.
    experiments = [
        (0.0, 0.0, 1.0),  # all 3 candidate rules predict Good
        (20.0, 0.0, 1.0),  # removes forward-but-x<10
        (3.0, 2.0, 0.0),  # removes "positive-command-is-good"
    ]
    survivors = {"command_positive", "displacement_positive", "displacement_positive_if_x_below_10"}
    record = []
    for x, velocity, action in experiments:
        delta = RHO * velocity + GAIN * action
        judged_good = delta > 0
        predictions = {
            "command_positive": action > 0,
            "displacement_positive": delta > 0,
            "displacement_positive_if_x_below_10": delta > 0 and x < 10,
        }
        survivors = {k for k in survivors if predictions[k] == judged_good}
        record.append({"x": x, "v": velocity, "a": action,
                       "displacement": delta, "good": judged_good,
                       "surviving_rules": sorted(survivors)})
    return record


def goal_reachability_checks():
    # Exact two-step reachable goal with unit momentum and discrete motor input.
    def roll(x, velocity, actions):
        for action in actions:
            velocity += action
            x += velocity
        return x, velocity

    result = {}
    for x, target in [(8.0, 9.0), (108.0, 109.0)]:
        successful = []
        for actions in itertools.product((-1, 0, 1), repeat=2):
            final_x, final_velocity = roll(x, 1.0, actions)
            if abs(final_x - target) < 1e-12 and abs(final_velocity) < 1e-12:
                successful.append(list(actions))
        result[f"start_{x:g}_goal_{target:g}"] = successful
    return result


def observational_aliasing():
    # Same current x and action, different velocity -> different future.
    x, action = 5.0, 0.0
    return {
        "current_position": x,
        "action": action,
        "v_positive_next_x": x + RHO * 1.0 + GAIN * action,
        "v_negative_next_x": x + RHO * (-1.0) + GAIN * action,
    }



def nonlinear_feature_checks():
    # A mild velocity-dependent drag law, with the exact term not given to
    # the search routine as the only option (a small expression library is).
    rng = np.random.default_rng(SEED + 1)
    def draw(n, max_velocity):
        v = rng.uniform(-max_velocity, max_velocity, n)
        u = rng.uniform(-1.0, 1.0, n)
        # Displacement d is observable as x_t - x_(t-1).
        y = 0.9 * v - 0.15 * v * np.abs(v) + 1.1 * u
        return v, u, y
    train = draw(400, 1.5)
    test = draw(1000, 2.5)  # harder out-of-range extrapolation
    def feature_names(v, u):
        return {
            "linear": np.column_stack([np.ones_like(v), v, u]),
            "quadratic": np.column_stack([np.ones_like(v), v, v*v, u]),
            "cubic": np.column_stack([np.ones_like(v), v, v**3, u]),
            "velocity_abs_velocity": np.column_stack([np.ones_like(v), v, v*np.abs(v), u]),
        }
    mats_train = feature_names(*train[:2])
    mats_test = feature_names(*test[:2])
    rows = {}
    for name in mats_train:
        coeffs = fit(mats_train[name], train[2])
        rows[name] = {"held_out_rmse": rmse(test[2], mats_test[name] @ coeffs),
                      "coefficients": [float(x) for x in coeffs]}
    return rows


def goal_relevance_check():
    # beta is uncertain. A vague forward-progress goal does not require its
    # identification; a precise target does.
    models = [0.5, 1.0]
    actions = [1, 2]
    good_actions = {str(gain): [a for a in actions if gain*a > 0] for gain in models}
    best_exact_actions = {str(gain): [a for a in actions if gain*a == 1.0] for gain in models}
    return {"possible_beta": models, "actions": actions,
            "good_actions_if_any_positive_movement": good_actions,
            "exact_target_actions_by_model": best_exact_actions,
            "informative_action": 1,
            "resulting_displacement_by_model": [float(g*1) for g in models]}



def observability_checks():
    # state=(x,v), output x, semi-implicit discretization.
    def make(rho):
        A = np.array([[1.0, rho], [0.0, rho]])
        B = np.array([[GAIN], [GAIN]])
        C = np.array([[1.0, 0.0]])
        observable = np.vstack([C, C @ A])
        controllable = np.hstack([B, A @ B])
        return {"observability_rank": int(np.linalg.matrix_rank(observable)),
                "observability_determinant": float(np.linalg.det(observable)),
                "controllability_rank": int(np.linalg.matrix_rank(controllable)),
                "controllability_determinant": float(np.linalg.det(controllable))}
    return {"rho_0_8": make(0.8), "rho_0": make(0.0)}


def main():
    output = {
        "environment": {"rho": RHO, "beta": GAIN, "seed": SEED,
                        "dynamics": "v_next=rho*v+beta*a, x_next=x+v_next"},
        "position_aliasing": observational_aliasing(),
        "regression": model_comparison(),
        "identifiability": identifiability_checks(),
        "judgment": judgment_checks(),
        "goal_reachability": goal_reachability_checks(),
        "nonlinear_feature_search": nonlinear_feature_checks(),
        "goal_relevance": goal_relevance_check(),
        "observability": observability_checks(),
    }
    # Regression/identification assertions: failures raise errors.
    assert output["regression"]["current_only"]["test_rmse"] > 1.0
    assert output["regression"]["previous_history"]["test_rmse"] < 1e-10
    assert output["regression"]["discovered_delta"]["test_rmse"] < 1e-10
    assert output["regression"]["translation_test_max_abs_difference"] < 1e-10
    assert output["identifiability"]["informative_design_rank"] == 2
    assert output["identifiability"]["confounded_design_rank"] == 1
    assert output["identifiability"]["confounded_same_predictions"]
    assert output["judgment"][-1]["surviving_rules"] == ["displacement_positive"]
    assert output["goal_reachability"]["start_8_goal_9"] == [[0, -1]]
    assert output["goal_reachability"]["start_108_goal_109"] == [[0, -1]]
    assert output["nonlinear_feature_search"]["linear"]["held_out_rmse"] > 0.1
    assert output["nonlinear_feature_search"]["velocity_abs_velocity"]["held_out_rmse"] < 1e-10
    assert output["observability"]["rho_0_8"]["observability_rank"] == 2
    assert output["observability"]["rho_0"]["observability_rank"] == 1
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
