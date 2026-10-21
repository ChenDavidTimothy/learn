"""Numpy auto-reset batch adapter for the official AMAGO environment stack.

Wrap with AMAGOEnv(env, batched_envs=env.batched_envs), then SequenceWrapper and
AlreadyVectorizedEnv. Spaces describe ONE member, arrays have a batch dimension.
AMAGO's SequenceWrapper expects the observation returned with done=True to be
the next episode's reset observation, and clears its RL2/action/reward context.
Old terminal states are retained only in ``completed``, never in observations.

The bootstrap must prepend its compatible Gymnasium before importing this file.
No AMAGO core changes, policy features from hidden parameters, or fixed probes.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from innovation_control_env import DT, EPISODE_STEPS, GOAL_STEPS, InnovationControlEnv

OBSERVATION_DIM = 20
OBSERVATION_LAYOUT = {
    'sin_q': (0, 2), 'cos_q': (2, 4), 'v_div_2': (4, 6),
    'all_future_goals': (6, 12), 'phase_one_hot': (12, 15),
    'episode_time_left': (15, 16), 'previous_issued_u': (16, 18),
    'previous_measured_acceleration_div_10': (18, 20),
}


class AMAGOControlEnv(gym.Env):
    """Unchanged physical task, with automatically reset synchronous batches.

    ``fixed_seed=True`` repeats the same complete body/initial-state/goal batch
    on every reset (validation and a separately seeded fresh confirmation).
    Otherwise batch k uses ``seed * 1000 + k``. All three objectives are supplied.
    Reward is exactly negative original per-interval cost, without shaping.
    """

    metadata = {'render_modes': []}

    def __init__(self, n=64, seed=20261021, split='train', fixed_seed=False):
        super().__init__()
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError('positive integer batch size required')
        self.batched_envs = self.num_envs = int(n)
        self.seed_base, self.split, self.fixed_seed = int(seed), split, bool(fixed_seed)
        self._episode_counter = 0
        self.plant = InnovationControlEnv()
        self.completed = []
        self._started = False
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        # No clipping of ordinary observed velocities or measured acceleration.
        low = np.array([-1.]*4 + [-np.inf]*2 + [-.8]*6 + [0.]*4 + [-1.]*2 + [-np.inf]*2, dtype=np.float32)
        high = np.array([1.]*4 + [np.inf]*2 + [.8]*6 + [1.]*4 + [1.]*2 + [np.inf]*2, dtype=np.float32)
        assert low.shape == high.shape == (OBSERVATION_DIM,)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space

    def _observation(self):
        state = self.plant.state
        phase = min(self.plant.t // GOAL_STEPS, 2)
        one_hot = np.zeros((self.batched_envs, 3))
        one_hot[:, phase] = 1.
        remaining = np.full((self.batched_envs, 1), (EPISODE_STEPS-self.plant.t)/EPISODE_STEPS)
        obs = np.concatenate((np.sin(state[:, :2]), np.cos(state[:, :2]), state[:, 2:]/2,
                              self.plant.goals.reshape(self.batched_envs, 6), one_hot, remaining,
                              self._previous_u, self._previous_acceleration/10), axis=1).astype(np.float32)
        if obs.shape != (self.batched_envs, OBSERVATION_DIM) or not np.isfinite(obs).all():
            raise FloatingPointError('invalid ordinary AMAGO observation')
        return obs

    def _reset_batch(self):
        self._episode_seed = self.seed_base*1000 + (0 if self.fixed_seed else self._episode_counter)
        self._episode_counter += 1
        state = self.plant.reset(self._episode_seed, self.batched_envs, self.split)
        self._previous_u = np.zeros((self.batched_envs, 2))
        self._previous_acceleration = np.zeros((self.batched_envs, 2))
        self._states = [state.copy()]
        self._actions, self._costs = [], []
        self._started = True
        return self._observation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('no task-changing reset options are supported')
        if seed is not None:
            self.seed_base = int(seed)
            self._episode_counter = 0
        return self._reset_batch(), {}

    def step(self, action):
        if not self._started:
            raise RuntimeError('reset before step')
        issued_u = np.asarray(action, dtype=np.float64)
        if issued_u.shape != (self.batched_envs, 2):
            raise ValueError('batched issued actions must have shape [n,2]')
        previous_state = self.plant.state.copy()
        state, cost = self.plant.step(issued_u)
        self._previous_u = issued_u.copy()
        self._previous_acceleration = (state[:, 2:]-previous_state[:, 2:])/DT
        self._states.append(state.copy())
        self._actions.append(issued_u.copy())
        self._costs.append(cost.copy())
        reward = (-cost).astype(np.float32)
        terminated = np.full(self.batched_envs, self.plant.done, dtype=bool)
        truncated = np.zeros(self.batched_envs, dtype=bool)
        info = {'AMAGO_LOG_METRIC interval_cost': cost.astype(np.float32)}
        if self.plant.done:
            states = np.stack(self._states, axis=1)
            costs = np.stack(self._costs, axis=1)
            precise = self.plant.precision()
            # This is evaluator storage only. Info sent through AMAGO contains
            # aggregate ordinary task metrics, never body parameters or B.
            self.completed.append(dict(seed=self._episode_seed, split=self.split,
                                       q=states[:, :, :2].copy(), v=states[:, :, 2:].copy(),
                                       u=np.stack(self._actions, axis=1), goals=self.plant.goals.copy(),
                                       cost=costs, precision=precise,
                                       truth=self.plant.evaluator_snapshot()))
            info.update({'AMAGO_LOG_METRIC precision': precise.astype(np.float32).reshape(-1),
                         'AMAGO_LOG_METRIC all_goals_precise': precise.all(axis=1).astype(np.float32),
                         'AMAGO_LOG_METRIC episode_mean_cost': costs.mean(axis=1).astype(np.float32)})
            # Official SequenceWrapper resets Timestep/RL2/agent context on this
            # done mask. Returning the OLD terminal observation would instead
            # initialize its next trajectory with the wrong body/context.
            observation = self._reset_batch()
        else:
            observation = self._observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        return None
