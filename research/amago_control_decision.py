#!/usr/bin/env python3
"""Validation-cost selection and gated inference-only fresh confirmation.

Selection uses the Python standard library only. Confirmation imports the frozen
study driver FIRST, so its process-local Gymnasium bootstrap runs exactly once.
No learning, optimizer step, test-informed selection or upstream edits occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
COMMIT = '54c25ab6da9371614c47352f569a56c0fe938d3b'
N = 64
STEPS = 240


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def current_sources():
    paths = [ROOT/name for name in ('amago_control_study.py', 'amago_control_env.py', 'amago_runtime.py',
                                    'innovation_control_env.py', 'error_transport_data.py',
                                    'third_party/amago/UPSTREAM.json', 'amago_control_plan.md')]
    manifest = json.loads((ROOT/'third_party/amago/UPSTREAM.json').read_text())
    assert manifest['commit'] == COMMIT
    for entry in manifest['files']:
        p = ROOT/'third_party/amago'/entry['path']; content = p.read_bytes()
        assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() == entry['git_blob'], 'modified official source'
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def inspect_fits(fits):
    assert len(fits) == 2 and len({Path(p).resolve() for p in fits}) == 2, 'exactly two distinct fit directories required'
    sources = current_sources(); reports, candidates = [], []
    for directory in fits:
        directory = Path(directory).resolve(); rp = directory/'report.json'
        r = json.loads(rp.read_text())
        assert r['status'] == 'COMPLETE MAINTAINED BASELINE FIT; FRESH CONFIRMATION UNINSPECTED'
        assert r['smoke'] is False and r['official_commit'] == COMMIT and r['sources'] == sources
        assert not (directory/'failure.json').exists(), 'failed fit is not eligible'
        assert r['seed'] == 20261021 and r['batches_per_epoch'] == 50
        assert len(r['history']) == 128 and [h['epoch'] for h in r['history']] == list(range(1,129))
        assert r['training_interactions'] == 128*N*STEPS and r['validation_interactions'] == 4*N*STEPS
        assert [c['epoch'] for c in r['candidates']] == [32,64,96,128]
        assert isinstance(r['operative_config'], str) and r['operative_config']
        for h in r['history']:
            assert math.isfinite(h['training']['mean_cost']) and len(h['training']['body_cost']) == N
        report_entry = dict(path=str(rp), sha256=sha(rp), lr=r['lr'], training_interactions=r['training_interactions'],
                            validation_interactions=r['validation_interactions'], official_gradient_minibatches=128*50,
                            operative_config=r['operative_config'])
        reports.append(report_entry)
        for c in r['candidates']:
            wp, vp = directory/c['weights'], directory/c['validation']
            assert wp.parent == vp.parent == directory and wp.is_file() and vp.is_file()
            assert sha(wp) == c['weights_sha256']
            m = c['metrics']; assert math.isfinite(m['mean_cost']) and 0 <= m['precision_fraction'] <= 1
            assert len(m['body_cost']) == len(m['precise_goals']) == N
            assert all(math.isfinite(v) for v in m['body_cost'])
            assert all(len(row) == 3 and all(v in (0,1) for v in row) for row in m['precise_goals'])
            assert math.isclose(sum(m['body_cost'])/N, m['mean_cost'], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(sum(map(sum,m['precise_goals']))/(N*3), m['precision_fraction'], rel_tol=0, abs_tol=1e-12)
            assert m['physical_seed'] == 2026102181*1000
            candidates.append(dict(fit_report=str(rp), fit_report_sha256=sha(rp), training_lr=r['lr'], epoch=c['epoch'],
                                   mean_validation_cost=m['mean_cost'], validation_precision=m['precision_fraction'],
                                   weights=str(wp), weights_sha256=sha(wp), validation=str(vp), validation_sha256=sha(vp),
                                   parameter_count=r['parameters']))
    assert sorted(r['lr'] for r in reports) == [.0001,.0003]
    # Literal min; ties preserve declared input/checkpoint order, not precision.
    selected = min(candidates, key=lambda c:c['mean_validation_cost'])
    return reports, candidates, selected, sources


def select(fits, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    reports, candidates, selected, sources = inspect_fits(fits)
    gate = selected['validation_precision'] >= .75
    value = dict(status=('FROZEN VALIDATION-COST SELECTION; CONFIRMATION GATE PASSED' if gate else
                         'CLOSED BUDGETED PREREQUISITE; NO COMPETENCE CLAIM; NO FRESH CONFIRMATION'),
                 criterion='literal minimum mean validation cost across both learning rates and four checkpoints',
                 selected=selected, candidates=candidates, fit_reports=reports, sources=sources,
                 decision_source_sha256=sha(__file__), validation_gate=gate, validation_precision_threshold=.75,
                 training_interactions=sum(r['training_interactions'] for r in reports),
                 validation_interactions=sum(r['validation_interactions'] for r in reports),
                 official_gradient_minibatches=sum(r['official_gradient_minibatches'] for r in reports),
                 scope='Bounded official GRU baseline prerequisite, one training seed; not full Transformer results or a new learning mechanism.')
    write_new(output,value)
    print(json.dumps(dict(status=value['status'], selected=selected, validation_gate=gate),indent=2))


def validate_selection(path):
    selection = json.loads(Path(path).read_text())
    assert selection['decision_source_sha256'] == sha(__file__), 'decision source changed after selection'
    assert selection['sources'] == current_sources()
    fits = [Path(r['path']).parent for r in selection['fit_reports']]
    reports,candidates,selected,sources = inspect_fits(fits)
    assert reports == selection['fit_reports'] and candidates == selection['candidates'] and selected == selection['selected']
    assert selection['validation_gate'] == (selected['validation_precision'] >= .75)
    return selection


def rls_episode(rho, split):
    # Ordinary feedback only. Identical calibration RNG across rho and test roles.
    import numpy as np
    from innovation_control_env import InnovationControlEnv, RLSAdaptiveController
    seed = (2026102181 if split == 'validation' else 2026102182)*1000
    env = InnovationControlEnv(); state = env.reset(seed,N,split)
    controller = RLSAdaptiveController(rho=rho).reset(N,2026102183*1000)
    states,actions,costs = [state.copy()],[],[]
    for _ in range(STEPS):
        action = controller.action(state,env.current_goal)
        next_state,cost = env.step(action)
        controller.update(state,action,next_state)
        states.append(next_state.copy()); actions.append(action.copy()); costs.append(cost.copy()); state=next_state
    states = np.stack(states,axis=1)
    assert controller.updates == STEPS
    return dict(seed=seed,split=split,q=states[:,:,:2],v=states[:,:,2:],u=np.stack(actions,axis=1),
                goals=env.goals.copy(),cost=np.stack(costs,axis=1),precision=env.precision(),truth=env.evaluator_snapshot())


def confirm(selection_path, output):
    output = Path(output).resolve()
    if output.suffix != '.json': raise ValueError('confirmation output must end in .json')
    if output.exists(): raise FileExistsError(output)
    selection = validate_selection(selection_path)
    if not selection['validation_gate']:
        print('CLOSED BUDGETED PREREQUISITE: validation-cost-selected checkpoint below75%; no fresh sampling or AMAGO import.')
        return
    stem = output.with_suffix('')
    paths = {key:Path(str(stem)+suffix) for key,suffix in
             [('rls_selection','_rls_selection.json'),('failure','_failure.json'),('runtime','_runtime')]}
    for method in ('amago','rls'):
        for split in ('test_id','test_ood'):
            paths[f'{method}_{split}'] = Path(str(stem)+f'_{method}_{split}.npz')
    for rho in (.98,.995): paths[f'rls_validation_{rho}'] = Path(str(stem)+f'_rls_validation_{rho:g}.npz')
    assert not any(p.exists() for p in paths.values()), 'refuse existing partial confirmation; inspect preserved failure first'
    # This import activates compatibility packages before any Gymnasium import.
    # Never call activate() ourselves, and never reload this module.
    import amago_control_study as study
    import numpy as np
    import torch
    import random
    import gin
    start = time.time(); partial = dict(status='STARTED CONDITIONAL CONFIRMATION', selection_sha256=sha(selection_path))
    records = {}; artifact_hashes = {}
    try:
        # Both rho candidates use the NEW validation set; freeze before test envs.
        rls_options = []
        for rho in (.98,.995):
            record = rls_episode(rho,'validation'); path = paths[f'rls_validation_{rho}']
            study.save_record(path,record); metrics=study.record_metrics(record)
            artifact_hashes[path.name]=sha(path)
            rls_options.append(dict(rho=rho,metrics=metrics,record=str(path),record_sha256=sha(path)))
        chosen_rls = min(rls_options,key=lambda r:r['metrics']['mean_cost'])
        write_new(paths['rls_selection'],dict(status='FROZEN RLS VALIDATION-COST CHOICE; FRESH TEST NOT SAMPLED',
                  options=rls_options,selected=chosen_rls,selection_sha256=sha(selection_path),
                  decision_source_sha256=sha(__file__),sources=selection['sources']))
        partial['rls_selection_sha256']=sha(paths['rls_selection'])
        random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021)
        config={}
        trajectory=study.cli_utils.switch_traj_encoder(config,'rnn',memory_size=128,layers=1)
        timestep=study.cli_utils.switch_tstep_encoder(config,'ff',n_layers=2,d_hidden=128,d_output=128,normalize_inputs=True)
        agent=study.cli_utils.switch_agent(config,'agent')
        study.cli_utils.use_config(config)
        dummy_refs=[]
        def dummy_env(split):
            raw=study.AMAGOControlEnv(n=N,seed=2026102180 if split=='train' else 2026102181,
                                      split=split,fixed_seed=True)
            dummy_refs.append(raw)
            return study.AMAGOEnv(raw,env_name=f'unchanged_two_link_{split}',batched_envs=N)
        fit_dir=Path(selection['selected']['fit_report']).parent
        dataset=study.DiskTrajDataset(str(ROOT/'local_datasets'),f'amago_{fit_dir.name}',dset_max_size=9000)
        experiment=study.RecordedExperiment(
            run_name='inference_only',ckpt_base_dir=str(paths['runtime']),max_seq_len=241,dataset=dataset,
            tstep_encoder_type=timestep,traj_encoder_type=trajectory,agent_type=agent,
            make_train_env=lambda:dummy_env('train'),make_val_env=lambda:dummy_env('validation'),
            parallel_actors=N,env_mode='already_vectorized',exploration_wrapper_type=None,
            sample_actions_train=True,sample_actions_val=False,val_timesteps_per_epoch=STEPS,
            traj_save_len=241,stagger_traj_file_lengths=False,epochs=0,train_timesteps_per_epoch=0,
            train_batches_per_epoch=0,has_dset_edit_rights=False,batch_size=16,
            learning_rate=selection['selected']['training_lr'],dloader_workers=0,
            val_interval=None,ckpt_interval=None,always_save_latest=False,log_to_wandb=False,verbose=False,
            log_interval=1,save_trajs_as='npz-compressed')
        experiment.local_logs=[];experiment.start()
        assert sha(selection['selected']['weights']) == selection['selected']['weights_sha256']
        state=torch.load(selection['selected']['weights'],map_location=experiment.DEVICE,weights_only=False)
        experiment.policy.load_state_dict(state,strict=True)
        assert all(torch.isfinite(v).all() for v in experiment.policy.state_dict().values())
        assert sum(p.numel() for p in experiment.policy.parameters()) == selection['selected']['parameter_count']
        before={k:v.detach().clone() for k,v in experiment.policy.state_dict().items()}
        assert not experiment.optimizer.state, 'unexpected pre-existing optimizer steps'
        for split in ('test_id','test_ood'):
            references=[]
            def test_env():
                raw=study.AMAGOControlEnv(n=N,seed=2026102182,split=split,fixed_seed=True)
                references.append(raw)
                return study.AMAGOEnv(raw,env_name=f'unchanged_two_link_{split}',batched_envs=N)
            official_logs=experiment.evaluate_test(test_env,timesteps=STEPS,save_trajs_to=None)
            official_logs={k:float(v) for k,v in official_logs.items()}
            assert len(references)==1 and len(references[0].completed)==1
            record=references[0].completed.pop()
            assert record['cost'].shape == (N,STEPS) and record['q'].shape == (N,STEPS+1,2)
            path=paths[f'amago_{split}'];study.save_record(path,record);artifact_hashes[path.name]=sha(path)
            records[f'amago_{split}']=dict(metrics=study.record_metrics(record),record=str(path),official_logs=official_logs)
            reference=rls_episode(chosen_rls['rho'],split)
            for key in ('parameters','input_matrix','goals'):
                assert np.array_equal(record['truth'][key],reference['truth'][key]), 'unpaired physical draw'
            assert np.array_equal(record['q'][:,0],reference['q'][:,0]) and np.array_equal(record['v'][:,0],reference['v'][:,0])
            path=paths[f'rls_{split}'];study.save_record(path,reference);artifact_hashes[path.name]=sha(path)
            records[f'rls_{split}']=dict(metrics=study.record_metrics(reference),record=str(path))
            assert all(torch.equal(v,before[k]) for k,v in experiment.policy.state_dict().items()), 'inference changed selected model/normalization'
            assert not experiment.optimizer.state, 'inference performed optimizer step'
        assert all(not ref.completed for ref in dummy_refs), 'dummy environment was stepped'
        assert selection['sources']==study.source_hashes()==current_sources()
        gates={}
        for split,threshold in [('test_id',.75),('test_ood',.5)]:
            a,r=records[f'amago_{split}']['metrics'],records[f'rls_{split}']['metrics']
            gates[split]=dict(precision_threshold=threshold,precision_pass=a['precision_fraction']>=threshold,
                              cost_no_worse_than_selected_rls=a['mean_cost']<=r['mean_cost'])
        passed=all(g['precision_pass'] and g['cost_no_worse_than_selected_rls'] for g in gates.values())
        artifact_hashes[paths['rls_selection'].name]=sha(paths['rls_selection'])
        write_new(output,dict(status='COMPLETE CONDITIONAL FRESH CONFIRMATION; INDEPENDENT CHECK PENDING',
                    competence_gate=passed,decision=('BOUNDED BASELINE COMPETENCE CRITERION MET' if passed else
                    'CLOSE BUDGETED PREREQUISITE; NO AUTOMATIC RESCUE'),gates=gates,records=records,
                    rls_validation_options=rls_options,selected_rls=chosen_rls,selected_policy=selection['selected'],
                    selection_sha256=sha(selection_path),decision_source_sha256=sha(__file__),sources=selection['sources'],
                    artifact_sha256=artifact_hashes,rls_validation_interactions=2*N*STEPS,
                    amago_confirmation_interactions=2*N*STEPS,rls_confirmation_interactions=2*N*STEPS,
                    confirmation_gradient_updates=0,dummy_environment_physical_interactions=0,
                    local_logs=experiment.local_logs,operative_config=gin.operative_config_str(),wall_seconds=time.time()-start,
                    scope='One selected trained seed; paired64 fresh ID and64 OOD bodies, ordinary exact q/v/full objectives. Early criterion, not inferential proof, Transformer/SOTA result or new mechanism.'))
    except BaseException as exc:
        write_new(paths['failure'],dict(status='FAILED / INTERRUPTED; NO RETRY',exception=repr(exc),partial=partial,
                    completed_records=records,artifact_sha256=artifact_hashes,decision_source_sha256=sha(__file__),
                    wall_seconds=time.time()-start))
        raise
    print(json.dumps(dict(status='COMPLETE',competence_gate=passed,gates=gates),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('select');p.add_argument('--fits',nargs='+',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('confirm');p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.mode=='select':select(args.fits,args.output)
    else:confirm(args.selection,args.output)


if __name__=='__main__':main()
