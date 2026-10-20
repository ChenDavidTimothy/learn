#!/usr/bin/env python3
"""Paired frozen acquired-library ablation; original online pilot immutable."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cartesian_stream_v17_learn import Predictor,load_selected,digest
from cartesian_stream_v17_control import make_cases,run_episode,json_ready

class FrozenLibraryPredictor(Predictor):
    def __init__(self,base):
        super().__init__('local_frozen_library',base.members,base.scale,base.config)
        self.actual_initial_rows=[m.n_rows for m in self.members]
        self.notifications=0;self.coefficients=[];self.queries=[]

    def prepare(self,observations,commands):
        result=super().prepare(observations,commands)
        self.coefficients.append(np.stack([m.numpy().copy() for m in result[0]]))
        self.queries.append(result[1].numpy().copy())
        return result

    def append_completed(self,observations,commands):
        assert len(commands)%2==0 and len(observations)==len(commands)+1
        assert [m.n_rows for m in self.members]==self.actual_initial_rows
        self.notifications+=1
        # Explicitly retain the acquired library unchanged. Current noisy history
        # still updates normally and prepare refits at the next actual renewal.

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='research/cartesian_stream_v17_rep0');args=p.parse_args()
    prefix=Path(args.prefix);out=Path(str(prefix)+'_retention_results.json');arr=Path(str(prefix)+'_retention_arrays.npz')
    fail=Path(str(prefix)+'_retention_failure.json')
    if any(a.exists() for a in (out,arr,fail)):raise FileExistsError('Refusing to overwrite ablation')
    start=time.time();torch.set_num_threads(1)
    parent_path=Path(str(prefix)+'_control_results.json');parent=json.loads(parent_path.read_text())
    assert parent['status'].startswith('COMPLETE')
    source_dir=Path(__file__).resolve().parent
    assert digest(source_dir/'cartesian_stream_v17_control.py')==parent['source_sha256']
    ordinary=Path(str(prefix)+'_ordinary.npz');assert digest(ordinary)==parent['ordinary_sha256']
    result=dict(status='IN PROGRESS FROZEN ACQUIRED-LIBRARY ABLATION',source_sha256=digest(__file__),
        parent_control_results_sha256=digest(parent_path),parent_control_source_sha256=parent['source_sha256'],
        design_sha256=digest(source_dir/'cartesian_stream_v17_retention_design.md'),episodes=[])
    arrays={};progress={}
    def checkpoint():
        np.savez_compressed(arr,**arrays);result['arrays_sha256']=digest(arr);result['wall_seconds']=time.time()-start
        out.write_text(json.dumps(json_ready(result),indent=2,allow_nan=False)+'\n')
    try:
        with threadpool_limits(limits=1):
            for case_index,case in enumerate(make_cases()):
                selected,meta=load_selected(prefix,ordinary)
                assert meta['weights_sha256']==parent['weights_sha256']
                assert meta['source_sha256']==parent['dependency_sha256']['cartesian_stream_v17_learn.py']
                assert meta['local_source_sha256']==parent['dependency_sha256']['cartesian_stream_v17_local.py']
                model=FrozenLibraryPredictor(selected['local']);progress.clear()
                record,raw=run_episode(case,model,'local_frozen_library',progress)
                assert model.notifications==90 and len(model.coefficients)==90
                assert [m.n_rows for m in model.members]==model.actual_initial_rows
                raw.update(local_fit_coefficients=np.array(model.coefficients),local_fit_queries=np.array(model.queries))
                record.update(case_index=case_index,local_library_initial_rows=model.actual_initial_rows,
                    local_library_final_rows=[m.n_rows for m in model.members],real_pair_notifications=model.notifications,
                    actual_local_rows_appended=0,selected_configuration=model.config,array_keys={})
                for name,value in raw.items():
                    key=f'episode{case_index}_{name}';arrays[key]=value;record['array_keys'][name]=key
                online=next(e for e in parent['ordinary_episodes'] if e['kind']=='local' and e['case_index']==case_index)
                assert record['noise_seed_path']==online['noise_seed_path'] and np.array_equal(record['goals'],online['goals'])
                record['paired_online_task_cost']=online['score']['task_cost']
                record['frozen_minus_online_task_cost']=record['score']['task_cost']-online['score']['task_cost']
                result['episodes'].append(record);checkpoint();print('frozen library case',case_index,'completed',flush=True)
        result['resources']=dict(episodes=8,plant_intervals=1568,physical_seconds=156.8,decisions=720,
            camera_read_opportunities=1576,scalar_noise_opportunities=3152,new_independent_noise_paths_relative_to_parent=0,
            actual_local_rows_appended=0,plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in result['episodes']),
            modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for e in result['episodes'] for d in e['decisions']))
        result['status']='COMPLETE FROZEN ACQUIRED-LIBRARY ABLATION; INDEPENDENT AUDIT PENDING';checkpoint()
    except BaseException as error:
        for name in ('states','observed','commands','nfev'):
            if name in progress:arrays['failed_partial_'+name]=np.array(progress[name])
        if 'model' in locals():
            arrays['failed_partial_local_coefficients']=np.array(model.coefficients);arrays['failed_partial_local_queries']=np.array(model.queries)
        failure=dict(error_type=type(error).__name__,error=str(error),active_episode=progress.get('record'))
        fail.write_text(json.dumps(json_ready(failure),indent=2,allow_nan=False)+'\n')
        result['status']='FAILED/INTERRUPTED ABLATION; PARTIAL ARRAYS PRESERVED';checkpoint();raise

if __name__=='__main__':main()
