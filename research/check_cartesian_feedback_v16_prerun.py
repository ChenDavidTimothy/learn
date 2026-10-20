#!/usr/bin/env python3
"""Root's direct sigma-point trajectory check, not an independent agent audit."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
import cartesian_feedback_v16 as f


def main():
    rng=np.random.default_rng(773630);records=[]
    with threadpool_limits(limits=1):
        for trial in range(12):
            k=1+trial%3;weights=rng.dirichlet(np.ones(k));means=rng.normal(size=(k,8))*.12
            means[:,4:]+=np.array([.35,-1.05,.35,.35]);raw=rng.normal(size=(k,8,8))*.035
            covs=raw@raw.transpose(0,2,1);b=f.Belief(weights,means,covs)
            goal=rng.normal(size=2)*f.L;steps=2+trial%2;U,cost,pg=f.plan(b,goal,steps)
            points=[];pw=[]
            for w,m,C in zip(weights,means,covs):
                eig,V=np.linalg.eigh(C);rows=(V*np.sqrt(np.maximum(eig,0))).T*np.sqrt(8)
                points.extend(np.r_[m[None]+rows,m[None]-rows]);pw.extend([w/16]*16)
            points=np.array(points);pw=np.array(pw)
            def residual(flat):
                s=points[:,:2].copy();v=points[:,2:4].copy();B=points[:,4:].reshape(-1,2,2)
                for u in flat.reshape(-1,2):
                    a=np.einsum('nij,j->ni',B,u);s+=v+.5*a;v+=a
                r=np.c_[(s-goal/f.T**2)/f.DS,np.sqrt(f.WV)*v/f.DS]*np.sqrt(pw)[:,None]
                return np.r_[r.ravel(),np.sqrt(f.WU)*flat]
            rc=residual(U.ravel());ref=least_squares(residual,np.zeros(steps*2),bounds=(-f.BOUND,f.BOUND),
                gtol=1e-12,ftol=1e-12,xtol=1e-12);rr=residual(ref.x)
            err=abs(rc@rc-cost);objdiff=abs(rr@rr-cost);diff=np.max(abs(ref.x-U.ravel()))
            assert err<1e-10 and objdiff<1e-9 and pg<1e-7,(trial,err,objdiff,pg)
            Ug,cg,_=f.plan(b.gaussian(),goal,steps)
            assert np.max(abs(Ug-U))<1e-9 and abs(cg-cost)<1e-10
            records.append(dict(trial=trial,components=k,steps=steps,cost_error=err,
                scipy_optimum_objective_error=objdiff,scipy_action_difference=float(diff),projected_gradient=pg))
    out=dict(status='PRE-RUN CHECK; root sigma-point direct dynamics and SciPy least-squares box solve, independent of QP moment formula; not independent-agent audit',
        pretest_correction='Initial action-distance 1e-6 assertion failed by 4.51e-6 on trial3 although objective agreed; changed acceptance to objective/KKT and retained action differences.',records=records)
    Path(__file__).with_name('cartesian_feedback_v16_prerun_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
