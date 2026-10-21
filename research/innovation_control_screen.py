#!/usr/bin/env python3
"""Bounded innovation-memory versus recurrent/adaptive control screen.

External probe is sampled AFTER actor mean, and is conditioned on in the PPO
likelihood. Physical states and past probes are ordinary fixed data in replay;
no derivative through the simulator or hidden body parameters is taken.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from innovation_control_env import (
    DT, EPISODE_STEPS, GOAL_STEPS, InnovationControlEnv, RLSAdaptiveController,
)

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'innovation_control_plan.md'
UPDATES, BATCH, PROBE = 128, 64, .35
SEED = 20261021
torch.set_num_threads(1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return {p.name: digest(p) for p in (PLAN, Path(__file__),
            ROOT/'innovation_control_env.py', ROOT/'error_transport_data.py')}


def layer(a, b, gain=np.sqrt(2)):
    out = nn.Linear(a, b)
    nn.init.orthogonal_(out.weight, gain)
    nn.init.zeros_(out.bias)
    return out


class Policy(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('moment', 'gru'):
            raise ValueError(kind)
        self.kind = kind
        self.encoder = nn.Sequential(layer(26 if kind == 'moment' else 22, 64), nn.Tanh())
        if kind == 'moment':
            self.core = nn.Sequential(layer(64, 64), nn.Tanh())
        else:
            self.core = nn.GRU(64, 64)
            for name, p in self.core.named_parameters():
                nn.init.zeros_(p) if 'bias' in name else nn.init.orthogonal_(p)
        self.actor = layer(64, 2, .01)
        self.critic = layer(64, 1, 1.)
        self.logstd = nn.Parameter(torch.full((2,), -.7))

    def forward(self, x, hidden=None):
        # x is always [time,batch,ordinary features], including single-step use.
        x = self.encoder(x)
        if self.kind == 'gru':
            x, hidden = self.core(x, hidden)
        else:
            x = self.core(x)
        return self.actor(x), self.critic(x).squeeze(-1), hidden


def observation(env, past_u, past_probe, past_acceleration, moment, kind, zero=False):
    ordinary = env.observation()
    state = ordinary['state']
    phase = min(env.t // GOAL_STEPS, 2)
    time_left = (GOAL_STEPS - env.t % GOAL_STEPS) / GOAL_STEPS
    x = np.concatenate((np.sin(state[:, :2]), np.cos(state[:, :2]), state[:, 2:]/2,
                        ordinary['future_goals'].reshape(env.n, 6),
                        np.broadcast_to(np.eye(3)[phase], (env.n, 3)),
                        np.full((env.n, 1), time_left), past_u, past_probe,
                        past_acceleration/10), axis=1)
    if kind == 'moment':
        x = np.concatenate((x, (moment*0 if zero else moment).reshape(env.n, 4)/10), axis=1)
    assert x.shape == (env.n, 26 if kind == 'moment' else 22)
    return x.astype(np.float32)


def rollout(policy, n, split, body_seed, noise_seed, zero=False):
    env = InnovationControlEnv()
    state = env.reset(body_seed, n, split)
    rng = np.random.default_rng(noise_seed)
    past_u = np.zeros((n, 2)); past_probe = np.zeros((n, 2))
    acceleration = np.zeros((n, 2)); moment = np.zeros((n, 2, 2))
    hidden = None
    obs, zs, probes, logps, values, costs = [], [], [], [], [], []
    states, actions, memories = [state.copy()], [], []
    with torch.no_grad():
        for _ in range(EPISODE_STEPS):
            x = observation(env, past_u, past_probe, acceleration, moment, policy.kind, zero)
            mu, value, hidden = policy(torch.from_numpy(x)[None], hidden)
            # Current xi is unavailable to the network that just chose mu.
            xi = rng.standard_normal((n, 2)).astype(np.float32)
            epsilon = rng.standard_normal((n, 2)).astype(np.float32)
            if env.t >= 32:
                xi *= 0
            if split != 'train':
                epsilon *= 0  # Standard deterministic actor evaluation, probe excepted.
            mean = mu[0] + PROBE*torch.from_numpy(xi)
            distribution = Normal(mean, policy.logstd.exp())
            z = mean + policy.logstd.exp()*torch.from_numpy(epsilon)
            u = z.tanh().numpy().astype(float)
            next_state, cost = env.step(u)
            obs.append(x); zs.append(z.numpy()); probes.append(xi)
            logps.append(distribution.log_prob(z).sum(-1).numpy())
            values.append(value[0].numpy()); costs.append(cost)
            memories.append(moment.copy()); actions.append(u); states.append(next_state.copy())
            acceleration = (next_state[:, 2:]-state[:, 2:])/DT
            if env.t <= 32:
                moment = .98*moment + .02*acceleration[:, :, None]*xi[:, None, :]
            past_u, past_probe, state = u, xi, next_state
    return {k: np.stack(v) for k, v in dict(obs=obs, z=zs, probe=probes,
             logp=logps, value=values, cost=costs, states=states,
             action=actions, moment=memories).items()} | {
             'precise': env.precision(), 'goals': env.goals.copy(),
             **{f'evaluator_{k}': v for k,v in env.evaluator_snapshot().items()}}


def advantages(cost, value):
    reward = -torch.from_numpy(cost).float()
    old_value = torch.from_numpy(value).float()
    adv = torch.zeros_like(reward)
    tail = torch.zeros(reward.shape[1])
    for t in reversed(range(EPISODE_STEPS)):
        next_value = old_value[t+1] if t+1 < EPISODE_STEPS else 0.
        delta = reward[t] + .99*next_value - old_value[t]
        tail = delta + .99*.95*tail
        adv[t] = tail
    return adv, adv+old_value


def optimize(policy, optimizer, data, rng):
    x = torch.from_numpy(data['obs']); z = torch.from_numpy(data['z'])
    probe = torch.from_numpy(data['probe']); old_logp = torch.from_numpy(data['logp'])
    adv, returns = advantages(data['cost'], data['value'])
    logs = []
    for _ in range(4):
        # Keep complete episodes and their order; GRU always begins at zero.
        for ids in np.array_split(rng.permutation(x.shape[1]), 4):
            mu, value, _ = policy(x[:, ids])
            dist = Normal(mu+PROBE*probe[:, ids], policy.logstd.exp())
            logratio = dist.log_prob(z[:, ids]).sum(-1)-old_logp[:, ids]
            ratio = logratio.exp()
            a = adv[:, ids]
            a = (a-a.mean())/(a.std(unbiased=False)+1e-8)
            pg = torch.maximum(-a*ratio, -a*ratio.clamp(.8, 1.2)).mean()
            vf = .5*(value-returns[:, ids]).square().mean()
            loss = pg+.5*vf
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite PPO objective; preserve failed fit')
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            logs.append([float(pg.detach()), float(vf.detach()),
                         float((ratio-1-logratio).mean().detach()), float(grad)])
    return logs


def metrics(data):
    per_body = data['cost'].mean(0)
    return dict(mean_cost=float(per_body.mean()), body_cost=per_body.tolist(),
                precise_fraction=float(data['precise'].mean()),
                precise_goals=data['precise'].astype(int).tolist())


def fit(kind, lr, output, updates=UPDATES):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    start = time.time()
    sources = provenance()
    torch.manual_seed(SEED)
    policy = Policy(kind)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    rng = np.random.default_rng([SEED, 75, 0])
    history, candidates = [], []
    try:
        for update in range(updates):
            data = rollout(policy, BATCH, 'train', SEED*1000+update,
                           [SEED, 75, 1, update])
            for group in optimizer.param_groups:
                group['lr'] = lr*(1-update/updates)
            losses = optimize(policy, optimizer, data, rng)
            history.append(dict(update=update+1, training=metrics(data), losses=losses,
                                logstd=policy.logstd.detach().tolist(), elapsed=time.time()-start))
            if (update+1) % 32 == 0 or update+1 == updates:
                name = f'update_{update+1:03d}'
                torch.save(policy.state_dict(), output/f'{name}.pt')
                val = rollout(policy, 32, 'validation', SEED*1000+500,
                              [SEED, 75, 2])
                np.savez_compressed(output/f'{name}_validation.npz', **val)
                candidates.append(dict(update=update+1, weights=f'{name}.pt',
                                       validation=metrics(val)))
                print(json.dumps(dict(kind=kind, lr=lr, update=update+1,
                                     validation=candidates[-1]['validation']['mean_cost'],
                                     precise=candidates[-1]['validation']['precise_fraction'],
                                     elapsed=time.time()-start)), flush=True)
            (output/'report.json').write_text(json.dumps(dict(status='IN PROGRESS',
                kind=kind, lr=lr, updates=updates, sources=sources,
                history=history, candidates=candidates), indent=2)+'\n')
        assert provenance() == sources, 'frozen source changed during fit'
        result = dict(status='COMPLETE EXPLORATORY FIT; TEST UNINSPECTED',
                      kind=kind, lr=lr, seed=SEED, sources=sources,
                      training_interactions=updates*BATCH*EPISODE_STEPS,
                      validation_interactions=len(candidates)*32*EPISODE_STEPS,
                      parameters=sum(p.numel() for p in policy.parameters()),
                      history=history, candidates=candidates, wall_seconds=time.time()-start)
        (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    except BaseException as exc:
        torch.save(policy.state_dict(), output/'interrupted.pt')
        (output/'failure.json').write_text(json.dumps(dict(status='FAILED / INTERRUPTED',
            exception=repr(exc), kind=kind, lr=lr, sources=sources,
            history=history, candidates=candidates, elapsed=time.time()-start), indent=2)+'\n')
        raise


def adaptive_rollout(rho, split, n=32):
    env = InnovationControlEnv()
    state = env.reset(SEED*1000+(500 if split == 'validation' else 600), n, split)
    controller = RLSAdaptiveController(rho).reset(n, SEED*1000+700)
    states, actions, costs = [state.copy()], [], []
    for _ in range(EPISODE_STEPS):
        # Complete objective is offered; this conventional rule uses current goal.
        ordinary = env.observation()
        _ = ordinary['future_goals']
        u = controller.action(state, ordinary['goal'])
        next_state, cost = env.step(u)
        controller.update(state, u, next_state)
        states.append(next_state.copy()); actions.append(u); costs.append(cost)
        state = next_state
    return dict(states=np.stack(states), action=np.stack(actions), cost=np.stack(costs),
                precise=env.precision(), goals=env.goals.copy(),
                **{f'evaluator_{k}': v for k,v in env.evaluator_snapshot().items()})


def select(directories, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    choices = {kind: [] for kind in ('moment', 'gru')}
    reports = []
    for directory in directories:
        directory = Path(directory)
        report_path = directory/'report.json'
        report = json.loads(report_path.read_text())
        assert report['status'].startswith('COMPLETE') and report['sources'] == provenance()
        reports.append(dict(path=str(directory), report_sha256=digest(report_path)))
        for candidate in report['candidates']:
            weights = directory/candidate['weights']
            choices[report['kind']].append(dict(path=str(weights), sha256=digest(weights),
                cost=candidate['validation']['mean_cost'], update=candidate['update'], lr=report['lr']))
    assert all(len(v) == 8 for v in choices.values()), 'requires both rates and all checkpoints'
    chosen = {kind: min(v, key=lambda x:x['cost']) for kind,v in choices.items()}
    adaptive = []
    for rho in (.98, .995):
        data = adaptive_rollout(rho, 'validation')
        path = output.with_name(f'innovation_control_rls_{rho}_validation.npz')
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **data)
        adaptive.append(dict(rho=rho, metrics=metrics(data), archive=str(path), sha256=digest(path)))
    chosen['rls'] = min(adaptive, key=lambda x:x['metrics']['mean_cost'])
    output.write_text(json.dumps(dict(status='FROZEN BEFORE FRESH TEST', sources=provenance(),
        reports=reports, choices=choices, adaptive_choices=adaptive, selected=chosen), indent=2)+'\n')
    print(json.dumps(chosen, indent=2))


def evaluate(selection_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(Path(selection_path).read_text())
    assert selection['sources'] == provenance()
    start = time.time()
    results = {}
    for split in ('test_id', 'test_ood'):
        results[split] = {}
        for kind in ('moment', 'gru', 'zero_moment', 'rls'):
            if kind == 'rls':
                data = adaptive_rollout(selection['selected']['rls']['rho'], split)
            else:
                name = 'moment' if kind == 'zero_moment' else kind
                chosen = selection['selected'][name]
                assert digest(chosen['path']) == chosen['sha256']
                policy = Policy(name)
                policy.load_state_dict(torch.load(chosen['path'], weights_only=True))
                data = rollout(policy, 32, split, SEED*1000+600,
                               [SEED, 75, 3], zero=kind == 'zero_moment')
            archive = output.with_name(f'innovation_control_{split}_{kind}.npz')
            if archive.exists():
                raise FileExistsError(archive)
            np.savez_compressed(archive, **data)
            results[split][kind] = dict(**metrics(data), archive=str(archive), sha256=digest(archive))
    gates = {}
    for split, threshold in [('test_id', .75), ('test_ood', .5)]:
        row = results[split]
        gates[split] = dict(precision=row['moment']['precise_fraction'] >= threshold,
            beats_gru=row['moment']['mean_cost'] <= .9*row['gru']['mean_cost'],
            beats_rls=row['moment']['mean_cost'] <= .9*row['rls']['mean_cost'],
            uses_moment=row['moment']['mean_cost'] <= .9*row['zero_moment']['mean_cost'])
    passed = all(all(v.values()) for v in gates.values())
    output.write_text(json.dumps(dict(status='EXPLORATORY TEST COMPLETE; NOT INDEPENDENTLY VERIFIED',
        selection_sha256=digest(selection_path), sources=provenance(), results=results,
        gates=gates, decision='CONTINUE TO CONFIRMATION' if passed else 'REJECT / RETIRE',
        test_interactions=8*32*EPISODE_STEPS, wall_seconds=time.time()-start), indent=2)+'\n')
    print(json.dumps(dict(gates=gates, decision='continue' if passed else 'reject'), indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    f = sub.add_parser('fit'); f.add_argument('--kind', choices=['moment','gru'], required=True)
    f.add_argument('--lr', type=float, required=True); f.add_argument('--output', required=True)
    f.add_argument('--updates', type=int, default=UPDATES)
    s = sub.add_parser('select'); s.add_argument('--fits', nargs='+', required=True); s.add_argument('--output', required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--selection', required=True); e.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.mode == 'fit': fit(a.kind, a.lr, a.output, a.updates)
    elif a.mode == 'select': select(a.fits, a.output)
    else: evaluate(a.selection, a.output)


if __name__ == '__main__':
    main()
