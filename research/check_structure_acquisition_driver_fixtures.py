#!/usr/bin/env python3
"""Small synthetic driver checks. Does not load data, fit, score or run physics."""
import json
import numpy as np
import torch
import structure_acquisition_screen as s
from error_transport_baselines import ALPaCA, CoDA


def main():
    torch.set_num_threads(1); torch.manual_seed(14)
    x = torch.arange(92*8*96*8, dtype=torch.float32).reshape(92,8,96,8)
    y = x[...,:2].clone(); counts = {n:0 for n in s.BUDGETS}
    for phase,pid in [('old',0), ('continue',1)]:
        rng = np.random.default_rng(np.random.SeedSequence([20261020,18,41,pid]))
        for _ in range(50):
            sx,sy,qx,qy,tids,m = s.episode(rng,phase,x,y)
            n=m['support']; counts[n]+=1
            assert sx.shape==(8,n,8) and qx.shape==(8,32,8)
            ids=np.array(m['body_ids'])
            if phase=='old': assert np.all(ids<32)
            else: assert np.all(ids[:4]<32) and np.all((ids[4:]>=56)&(ids[4:]<64))
            assert torch.equal(tids,torch.tensor(np.where(ids>=56,ids-24,ids)))
            for j,st in enumerate(m['support_trajectories']):
                query_trajectories=(qx[j,:,0].to(torch.int64)//(96*8))%8
                assert torch.all(query_trajectories!=st)
    assert all(counts.values())
    old,new=ALPaCA(feature_dim=3),ALPaCA(feature_dim=5)
    with torch.no_grad(): old.raw_precision_cholesky[1,0]=.4; old.raw_precision_cholesky[2,0]=-.2
    original_graph,original_selected,original_load=s.graph_for,s.selected_fit,torch.load
    original_coda=s.selected_original_coda
    try:
        s.graph_for=lambda method:({'features':list(range(3 if method=='programold' else 5)),
                                   'nodes':[dict(op='input',index=j) for j in range(3 if method=='programold' else 5)]},None)
        s.selected_fit=lambda *args:{'weights':'synthetic'}
        torch.load=lambda *args,**kwargs:{'state_dict':old.state_dict()}
        s.initialize('meta','continue',new,'cpu')
        assert torch.allclose(new.prior_precision()[:3,:3],old.prior_precision())
        assert torch.count_nonzero(new.prior_precision()[:3,3:])==0
        oldc,newc=CoDA(num_tasks=32),CoDA(num_tasks=40)
        pad=newc.contexts.weight.detach()[32:].clone()
        s.selected_original_coda=lambda:{'weights':'synthetic'}
        torch.load=lambda *args,**kwargs:{'state_dict':oldc.state_dict()}
        s.initialize('coda','continue',newc,'cpu')
        assert torch.equal(newc.contexts.weight[:32],oldc.contexts.weight)
        assert torch.equal(newc.contexts.weight[32:],pad)
        assert torch.equal(newc.theta0,oldc.theta0) and torch.equal(newc.W,oldc.W)
    finally:
        s.graph_for,s.selected_fit,torch.load=original_graph,original_selected,original_load
        s.selected_original_coda=original_coda
    original_evaluate=s.evaluate
    try:
        s.evaluate=lambda model,method,x,y,ids,**kw:({}, {str(n):[1. if int(ids[0])==32 else 9.]*len(ids) for n in (4,16,64)})
        best,_=s.validation(None,'neural','continue',None,None)
        assert best['criterion']==5.
    finally: s.evaluate=original_evaluate
    result=dict(status='PASS SYNTHETIC DRIVER FIXTURES; NO FIT/SCORING RUN',source_sha256=s.digest(s.__file__),
                fixture_source_sha256=s.digest(__file__),episode_cases=100,observed_support_counts=counts,
                support_query_trajectories_disjoint=True,old_new_body_mix_and_coda_remap=True,
                expanded_prior_oldblock_and_independent_unit_newblock=True,
                coda_old_contexts_global_weights_preserved_new_random_contexts_retained=True,
                unequal_group_validation_equal_weight=best['criterion'])
    s.save_json(s.ROOT/'check_structure_acquisition_driver_fixtures.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
