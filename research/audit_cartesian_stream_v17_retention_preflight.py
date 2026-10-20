#!/usr/bin/env python3
"""No-physics equality/no-op fixture for the frozen acquired-library ablation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'cartesian_stream_v17_retention.py'
    spec=importlib.util.spec_from_file_location('retention_unit_under_test',source)
    unit=importlib.util.module_from_spec(spec);spec.loader.exec_module(unit)
    import cartesian_stream_v17_control as control
    def forbidden(*args,**kwargs):raise AssertionError('Preflight must not execute physical flows')
    control.flow=forbidden
    prefix=ROOT/'cartesian_stream_v17_rep0';ordinary=Path(str(prefix)+'_ordinary.npz')
    original,_=unit.load_selected(prefix,ordinary);separate,_=unit.load_selected(prefix,ordinary)
    online=original['local'];frozen=unit.FrozenLibraryPredictor(separate['local'])
    data=np.load(ordinary,allow_pickle=False);o=data['observed'][24];a=data['commands'][24]
    obs=o[:41];commands=a[:40];plan=a[40:52:2].copy();goal=.95*obs[-1]
    with threadpool_limits(limits=1):
        po=online.prepare(obs,commands);pf=frozen.prepare(obs,commands)
        assert np.array_equal(po[1].numpy(),pf[1].numpy())
        coefficient_error=max(float(np.max(abs(x.numpy()-y.numpy()))) for x,y in zip(po[0],pf[0]))
        path_online=online.paths(obs,commands,torch.tensor(plan),po)
        path_frozen=frozen.paths(obs,commands,torch.tensor(plan),pf)
        path_error=float(torch.max(abs(path_online-path_frozen)))
        def objective(model,prepared):
            p=torch.tensor(plan,requires_grad=True)
            loss=control.modeled_loss(model,obs,commands,goal,p,prepared)
            return float(loss.detach()),torch.autograd.grad(loss,p)[0].numpy()
        co,go=objective(online,po);cf,gf=objective(frozen,pf)
        cost_error=abs(co-cf);gradient_error=float(np.max(abs(go-gf)))
        assert max(coefficient_error,path_error,cost_error,gradient_error)==0.
        snapshots=[(m.context_train.copy(),m.command_train.copy(),m.target_train.copy()) for m in frozen.members]
        initial=[m.n_rows for m in frozen.members]
        frozen.append_completed(o[:43],a[:42]);online.append_completed(o[:43],a[:42])
        assert frozen.notifications==1 and [m.n_rows for m in frozen.members]==initial
        assert [m.n_rows for m in online.members]==[n+1 for n in initial]
        for member,before in zip(frozen.members,snapshots):
            assert all(np.array_equal(x,y) for x,y in zip((member.context_train,member.command_train,member.target_train),before))
        after=frozen.prepare(obs,commands)
        assert all(np.array_equal(x.numpy(),y.numpy()) for x,y in zip(pf[0],after[0]))
        assert len(frozen.coefficients)==2 and len(frozen.queries)==2
    files=[source,ROOT/'cartesian_stream_v17_retention_design.md',ROOT/'cartesian_stream_v17_control.py',ROOT/'cartesian_stream_v17_learn.py',ROOT/'cartesian_stream_v17_local.py',Path(str(prefix)+'_fit.json'),Path(str(prefix)+'_weights.npz'),ordinary]
    result=dict(status='PASS RETENTION PREFLIGHT; NO ACTUAL PHYSICS OR CONTROL EXECUTED',artifact_sha256={p.name:sha(p) for p in files},
        exact_equalities=dict(local_coefficients=coefficient_error,mean_paths=path_error,modeled_loss=cost_error,plan_gradient=gradient_error),
        initial_member_rows=initial,frozen_member_rows_after_notification=[m.n_rows for m in frozen.members],online_member_rows_after_same_notification=[m.n_rows for m in online.members],
        frozen_library_bytes_unchanged=True,captured_preparations=2,
        limitations='Canonical subclass/parent units imported as tested adapters; no physical flow or optimizer run. Actual720 decisions, coefficient provenance and paired physical trajectories require later independent artifact replay.')
    (ROOT/'audit_cartesian_stream_v17_retention_preflight_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
