"""Small actual-control instrument and conventional online RLS reference.

The learner receives exact q/v, all three future goals and the objective clock.
Body parameters and signed actuator maps are evaluator-only. Mechanics reuse error_transport_data
unchanged. This is a conventional adaptive-control reference, not a reproduction
of a modern method or a proposed new learning mechanism.

Run this module directly for six bounded DOP853 checks and torque-map fixtures;
the checks do not train a model or execute a control study.
"""
from __future__ import annotations

import json

import numpy as np

from error_transport_data import DT, rhs, step as physical_step

SPLITS = ('train', 'validation', 'test_id', 'test_ood')
GOAL_STEPS = 80
EPISODE_STEPS = 3 * GOAL_STEPS


def wrap(angle):
    """Joint displacement in [-pi, pi). State itself remains unwrapped."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class InnovationControlEnv:
    """Synchronous batch with fixed hidden body and actuator map per episode.

    reset(seed,n,split) and step(u) return state arrays [n,4]. step additionally
    returns cost[n]. Cost and precision use the POST-transition state and the
    goal that was current when the action was issued. No state resets at goals.
    """

    def __init__(self):
        self.n = 0
        self.t = 0

    def reset(self, seed, n, split='train'):
        if split not in SPLITS or not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError('positive batch size and a declared split required')
        self.n, self.t, self.split = int(n), 0, split
        split_id = SPLITS.index(split)
        def rng(role):
            return np.random.default_rng(np.random.SeedSequence([int(seed), 73, split_id, role]))
        body_rng = rng(0)
        if split == 'test_ood':
            high = body_rng.integers(0, 2, (n, 4)).astype(bool)
            masses_lengths = np.where(high, body_rng.uniform(1.3, 1.5, (n, 4)),
                                     body_rng.uniform(.6, .75, (n, 4)))
            damping = body_rng.uniform(.03, .3, (n, 2))
        else:
            masses_lengths = body_rng.uniform(.8, 1.2, (n, 4))
            damping = body_rng.uniform(.08, .2, (n, 2))
        # Existing mechanics sees unit motor gains and zero cross coupling;
        # the complete unknown actuator map is applied to issued commands first.
        self._parameters = np.concatenate((masses_lengths, damping, np.ones((n, 2)), np.zeros((n, 1))), axis=1)
        actuator_rng = rng(1)
        gains = actuator_rng.uniform(.6 if split == 'test_ood' else .8,
                                     1.5 if split == 'test_ood' else 1.2, (n, 2))
        signs = 2 * actuator_rng.integers(0, 2, (n, 2)) - 1
        swapped = actuator_rng.integers(0, 2, n)
        rows = np.where(swapped[:, None], np.array([1, 0]), np.array([0, 1]))
        self._input_matrix = np.zeros((n, 2, 2))
        self._input_matrix[np.arange(n)[:, None], rows, np.arange(2)[None]] = signs * gains
        initial_rng = rng(2)
        self.state = np.concatenate((initial_rng.uniform(-.4, .4, (n, 2)),
                                     initial_rng.uniform(-.2, .2, (n, 2))), axis=1)
        self.goals = rng(3).uniform(-.8, .8, (n, 3, 2))
        self._precise = np.ones((n, 3), dtype=bool)
        self._tail_observations = np.zeros(3, dtype=int)
        return self.state.copy()

    @property
    def current_goal(self):
        if not self.n:
            raise RuntimeError('reset first')
        return self.goals[:, min(self.t // GOAL_STEPS, 2)].copy()

    @property
    def time(self):
        return self.t * DT

    @property
    def done(self):
        return self.t >= EPISODE_STEPS

    def observation(self):
        """Full future objective is supplied; hidden body/actuator maps are not."""
        return dict(state=self.state.copy(), goal=self.current_goal, future_goals=self.goals.copy(),
                    goal_index=min(self.t // GOAL_STEPS, 2), time=self.time)

    def step(self, issued_u):
        if not self.n or self.done:
            raise RuntimeError('reset required before stepping a completed/uninitialized batch')
        u = np.asarray(issued_u, dtype=float)
        if u.shape != (self.n, 2) or not np.isfinite(u).all() or np.any(np.abs(u) > 1):
            raise ValueError('issued commands must be finite [n,2] in [-1,1]')
        goal_index = self.t // GOAL_STEPS
        goal = self.current_goal
        physical_u = np.einsum('nij,nj->ni', self._input_matrix, u)
        next_state = physical_step(self.state, physical_u, self._parameters, substeps=8)
        if not np.isfinite(next_state).all():
            raise FloatingPointError('nonfinite physical state; no clipping or body removal')
        error = wrap(next_state[:, :2] - goal)
        cost = np.mean(error**2, axis=1) + .1*np.mean(next_state[:, 2:]**2, axis=1) + .01*np.mean(u**2, axis=1)
        # Post-transition samples 61..80 of each goal give exactly 20 checks.
        if self.t % GOAL_STEPS >= GOAL_STEPS - 20:
            precise_now = np.all(np.abs(error) <= .15, axis=1) & np.all(np.abs(next_state[:, 2:]) <= .15, axis=1)
            self._precise[:, goal_index] &= precise_now
            self._tail_observations[goal_index] += 1
        self.state = next_state
        self.t += 1
        return next_state.copy(), cost

    def precision(self):
        """[n,3] booleans only after every goal has all 20 tail observations."""
        if not self.done or not np.array_equal(self._tail_observations, np.full(3, 20)):
            raise RuntimeError('precision requires a completed episode')
        return self._precise.copy()

    def evaluator_snapshot(self):
        """Privileged audit/scoring data; never a learner input."""
        return dict(parameters=self._parameters.copy(), input_matrix=self._input_matrix.copy(),
                    goals=self.goals.copy(), tail_observations=self._tail_observations.copy())


class RLSAdaptiveController:
    """Conventional 7D affine acceleration RLS plus regularized inverse control.

    Features [1,q1,q2,v1,v2,u1,u2] predict observed (v_next-v)/DT.
    theta/P start at zero/I (ridge=1); no body model, B, or parameter is supplied.
    Call action(state,goal), then update(state,issued_u,next_state) exactly once
    per actual transition. First 32 actions are independent uniform calibration
    commands; their physical cost is part of the episode. rho is validation-only.
    """

    def __init__(self, rho=.98):
        if not 0 < rho <= 1:
            raise ValueError('forgetting factor must be in (0,1]')
        self.rho = float(rho)
        self.n = 0

    def reset(self, n, seed):
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError('positive batch size required')
        self.n = int(n)
        self.theta = np.zeros((n, 7, 2))
        self.precision_inverse = np.broadcast_to(np.eye(7), (n, 7, 7)).copy()
        self.updates = 0
        self._rng = np.random.default_rng(np.random.SeedSequence([int(seed), 74]))
        return self

    def _state(self, state):
        state = np.asarray(state, dtype=float)
        if not self.n or state.shape != (self.n, 4) or not np.isfinite(state).all():
            raise ValueError('reset first and supply a finite state [n,4]')
        return state

    def action(self, state, goal):
        state = self._state(state)
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (self.n, 2) or not np.isfinite(goal).all():
            raise ValueError('finite current goal [n,2] required')
        if self.updates < 32:
            return self._rng.uniform(-1, 1, (self.n, 2))
        desired_acceleration = 4*wrap(goal-state[:, :2]) - 4*state[:, 2:]
        autonomous_features = np.concatenate((np.ones((self.n, 1)), state), axis=1)
        drift = np.einsum('ni,nio->no', autonomous_features, self.theta[:, :5])
        gain = self.theta[:, 5:].transpose(0, 2, 1)
        gain_t = gain.transpose(0, 2, 1)
        gram = gain_t @ gain + .1*np.eye(2)
        target = np.einsum('nij,nj->ni', gain_t, desired_acceleration-drift)
        u = np.linalg.solve(gram, target[..., None])[..., 0]
        if not np.isfinite(u).all():
            raise FloatingPointError('nonfinite RLS command')
        return np.clip(u, -1, 1)

    def update(self, state, issued_u, next_state):
        state, next_state = self._state(state), self._state(next_state)
        u = np.asarray(issued_u, dtype=float)
        if u.shape != (self.n, 2) or not np.isfinite(u).all() or np.any(np.abs(u) > 1):
            raise ValueError('finite bounded issued action [n,2] required')
        x = np.concatenate((np.ones((self.n, 1)), state, u), axis=1)
        target = (next_state[:, 2:]-state[:, 2:])/DT
        px = np.einsum('nij,nj->ni', self.precision_inverse, x)
        denominator = self.rho + np.sum(x*px, axis=1)
        if not np.all(denominator > 0):
            raise FloatingPointError('RLS lost positive information')
        k = px/denominator[:, None]
        residual = target-np.einsum('ni,nio->no', x, self.theta)
        self.theta += k[:, :, None]*residual[:, None, :]
        self.precision_inverse = (self.precision_inverse-k[:, :, None]*px[:, None, :])/self.rho
        self.precision_inverse = .5*(self.precision_inverse+self.precision_inverse.transpose(0, 2, 1))
        if not np.isfinite(self.theta).all() or not np.isfinite(self.precision_inverse).all():
            raise FloatingPointError('nonfinite RLS update')
        self.updates += 1


def checks():
    """Critical bounded checks, with fixed criteria and six independent solvers."""
    from scipy.integrate import solve_ivp
    env = InnovationControlEnv()
    state = env.reset(20261021, 6, 'test_ood')
    truth = env.evaluator_snapshot()
    rng = np.random.default_rng(2026102173)
    u = rng.uniform(-1, 1, (6, 2))
    torque = np.einsum('nij,nj->ni', truth['input_matrix'], u)
    expected = physical_step(state, torque, truth['parameters'], substeps=8)
    actual, _ = env.step(u)
    torque_error = float(np.max(np.abs(actual-expected)))
    assert torque_error == 0., ('issued-to-torque mismatch', torque_error)
    errors = []
    for j in range(6):
        sol = solve_ivp(lambda _, z: rhs(z, torque[j], truth['parameters'][j]), (0, DT), state[j],
                        method='DOP853', rtol=1e-11, atol=1e-13)
        assert sol.success
        errors.append(float(np.max(np.abs(expected[j]-sol.y[:, -1]))))
    assert max(errors) < 1e-7, ('DOP853 spot discrepancy', errors)
    # RLS is ordinary algebra: a single sample must match ridge-1 batch fitting.
    rls = RLSAdaptiveController(rho=.98).reset(6, 18)
    rls.update(state, u, actual)
    x = np.concatenate((np.ones((6, 1)), state, u), axis=1)
    target = (actual[:, 2:]-state[:, 2:])/DT
    batch = np.stack([np.linalg.solve(.98*np.eye(7)+np.outer(a, a), np.outer(a, b)) for a,b in zip(x,target)])
    rls_error = float(np.max(np.abs(rls.theta-batch)))
    assert rls_error < 1e-12, ('RLS single-update mismatch', rls_error)
    return dict(status='PASS BOUNDED INSTRUMENT CHECKS; NO CONTROL STUDY', issued_torque_exact_step_error=torque_error,
                dop853_spot_errors=errors, dop853_max_error=max(errors), dop853_tolerance=1e-7,
                rls_single_update_ridge_error=rls_error, rls_tolerance=1e-12)


if __name__ == '__main__':
    try:
        print(json.dumps(checks(), indent=2, allow_nan=False))
    except BaseException as exc:
        # Preserve the exact failure, source and inputs via the fixed seeded
        # fixture. Never silently change tolerance or retry the instrument.
        import hashlib
        from pathlib import Path
        source = Path(__file__)
        path = source.with_name('innovation_control_env_check_failure.json')
        if not path.exists():
            path.write_text(json.dumps(dict(status='FAILED; NO RETRY', exception=repr(exc), seed=20261021,
                                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), indent=2)+'\n')
        raise
