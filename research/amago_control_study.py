#!/usr/bin/env python3
"""Run the pinned official AMAGO learner; custom code only adapts/logs our task."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from amago_runtime import activate
activate()

import gin
import numpy as np
import torch
import amago
from amago import cli_utils
from amago.envs import AMAGOEnv
from amago.loading import DiskTrajDataset

from amago_control_env import AMAGOControlEnv

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), ROOT/'amago_control_env.py', ROOT/'amago_runtime.py',
             ROOT/'innovation_control_env.py', ROOT/'error_transport_data.py',
             ROOT/'third_party/amago/UPSTREAM.json']
    if (ROOT/'amago_control_plan.md').exists():
        paths.append(ROOT/'amago_control_plan.md')
    manifest = json.loads((ROOT/'third_party/amago/UPSTREAM.json').read_text())
    for entry in manifest['files']:
        p = ROOT/'third_party/amago'/entry['path']
        content = p.read_bytes()
        assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() == entry['git_blob'], 'upstream source modified'
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def record_metrics(record):
    body_cost = record['cost'].mean(axis=1)
    return dict(mean_cost=float(body_cost.mean()), body_cost=body_cost.tolist(),
                precision_fraction=float(record['precision'].mean()),
                precise_goals=record['precision'].astype(int).tolist(),
                physical_seed=record['seed'])


def save_record(path, record):
    ordinary = {k:v for k,v in record.items() if k != 'truth'}
    ordinary.update({f'evaluator_{k}':v for k,v in record['truth'].items()})
    np.savez_compressed(path, **ordinary)


class RecordedExperiment(amago.Experiment):
    def log(self, metrics_dict, key):
        # Core optimizer/targets/replay are unchanged; only logging is local.
        scalars = {}
        for k,v in metrics_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.detach().cpu().float().item()
            if isinstance(v, (int,float,np.number)):
                scalars[k] = float(v)
        if 'Loss' in scalars and not np.isfinite(scalars['Loss']):
            raise FloatingPointError('nonfinite official training loss; preserve failure')
        self.local_logs.append(dict(key=key, metrics=scalars))
        return super().log(metrics_dict, key)


def run(output, lr, smoke=False, epochs=128, batches=50):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed = 20261021
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n = 4 if smoke else 64
    if smoke:
        epochs, batches = 2, 2
    elif not (ROOT/'amago_control_plan.md').exists():
        raise RuntimeError('freeze a prospective competence plan before substantial training')
    sources = source_hashes()
    config = {}
    trajectory = cli_utils.switch_traj_encoder(config, 'rnn', memory_size=128, layers=1)
    timestep = cli_utils.switch_tstep_encoder(config, 'ff', n_layers=2, d_hidden=128,
                                              d_output=128, normalize_inputs=True)
    agent = cli_utils.switch_agent(config, 'agent')  # official PopArt/multigamma defaults
    # Keep official actor/critic architectures and their default loss coefficients.
    cli_utils.use_config(config)
    train_refs, val_refs = [], []

    def make_env(split, references):
        env = AMAGOControlEnv(n=n, seed=2026102180 if split == 'train' else 2026102181,
                              split=split, fixed_seed=split != 'train')
        references.append(env)
        return AMAGOEnv(env, env_name=f'unchanged_two_link_{split}', batched_envs=n)

    # Keep complete ordinary replay locally; publish source, checkpoints, losses
    # and actual validation/confirmation records rather than thousands of caches.
    dataset = DiskTrajDataset(str(ROOT/'local_datasets'), f'amago_{output.name}', dset_max_size=9000)
    experiment = RecordedExperiment(
        run_name='official_gru', ckpt_base_dir=str(output), max_seq_len=241,
        dataset=dataset, tstep_encoder_type=timestep, traj_encoder_type=trajectory,
        agent_type=agent, make_train_env=lambda:make_env('train',train_refs),
        make_val_env=lambda:make_env('validation',val_refs), parallel_actors=n,
        env_mode='already_vectorized', exploration_wrapper_type=None,
        sample_actions_train=True, sample_actions_val=False,
        val_timesteps_per_epoch=240, traj_save_len=241, stagger_traj_file_lengths=False,
        epochs=epochs, train_timesteps_per_epoch=240, train_batches_per_epoch=batches,
        batch_size=2 if smoke else 16, learning_rate=lr, dloader_workers=0,
        val_interval=None, ckpt_interval=None, always_save_latest=False,
        log_to_wandb=False, verbose=False, log_interval=1, save_trajs_as='npz-compressed')
    experiment.local_logs = []
    start = time.time()
    history, candidates = [], []
    try:
        experiment.start()
        for i in range(epochs):
            # Advance exactly one official epoch. Upstream stores the last epoch
            # index, so explicitly set the next index instead of repeating it.
            experiment.epoch = i
            experiment.epochs = i+1
            t0 = time.time()
            experiment.learn()
            completed = train_refs[-1].completed
            assert len(completed) == 1, 'one physical batch episode per official epoch'
            train = record_metrics(completed.pop())
            history.append(dict(epoch=i+1, training=train, wall_seconds=time.time()-t0,
                                log_end=len(experiment.local_logs)))
            if (i+1) % 32 == 0 or i+1 == epochs:
                before_validation = {k:v.detach().clone() for k,v in experiment.policy.state_dict().items()}
                experiment.evaluate_val()
                assert all(torch.equal(v, before_validation[k]) for k,v in experiment.policy.state_dict().items()), 'validation changed model/normalization state'
                records = val_refs[-1].completed
                assert len(records) == 1
                record = records.pop()
                name = f'epoch_{i+1:03d}'
                assert all(torch.isfinite(p).all() for p in experiment.policy.parameters()), 'nonfinite policy'
                save_record(output/f'{name}_validation.npz', record)
                torch.save(experiment.policy.state_dict(), output/f'{name}.pt')
                if not smoke:
                    experiment.epoch = i+1
                    # CUDA GRU flattens distinct parameters into shared storage.
                    # Safetensors heuristics dropped distinct views in the prefit
                    # profile. Standard pickle serialization preserves all views.
                    experiment.accelerator.save_state(
                        str(output/f'continuation_epoch_{i+1:03d}'), safe_serialization=False)
                candidates.append(dict(epoch=i+1, metrics=record_metrics(record),
                    weights=f'{name}.pt', weights_sha256=file_hash(output/f'{name}.pt'),
                    validation=f'{name}_validation.npz'))
            if (i+1) % 4 == 0 or smoke:
                print(json.dumps(dict(epoch=i+1, train_cost=train['mean_cost'],
                    train_precision=train['precision_fraction'], seconds=time.time()-start)), flush=True)
            result = dict(status='IN PROGRESS', official_commit='54c25ab6da9371614c47352f569a56c0fe938d3b',
                smoke=smoke, lr=lr, sources=sources, history=history, candidates=candidates,
                local_logs=experiment.local_logs, training_interactions=(i+1)*n*240,
                validation_interactions=len(candidates)*n*240, wall_seconds=time.time()-start,
                device=str(experiment.DEVICE), seed=seed, batches_per_epoch=batches,
                parameters=sum(p.numel() for p in experiment.policy.parameters()))
            (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        assert sources == source_hashes(), 'study sources changed during execution'
        result['status'] = ('COMPLETE ENGINEERING SMOKE; NO CAPABILITY CLAIM' if smoke else
                            'COMPLETE MAINTAINED BASELINE FIT; FRESH CONFIRMATION UNINSPECTED')
        result['operative_config'] = gin.operative_config_str()
        (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(status=result['status'], seconds=result['wall_seconds'],
                              validation=candidates[-1]['metrics'])), flush=True)
    except BaseException as exc:
        if hasattr(experiment, 'policy_aclr'):
            torch.save(experiment.policy.state_dict(), output/'interrupted.pt')
        (output/'failure.json').write_text(json.dumps(dict(status='FAILED / INTERRUPTED',
            exception=repr(exc), sources=sources, history=history,
            local_logs=experiment.local_logs, wall_seconds=time.time()-start), indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.output,args.lr,args.smoke)
