#!/usr/bin/env python3
"""Independent terminal-mean/variance diagnostic for saved known-parameter KFs.

No controller or canonical simulator imports. Reads already saved arrays only.
This diagnoses an unannounced final control cutoff; it is not a rerun/repair.
"""
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def diagnostic(theta,sensor_sd,episodes):
    rho,lam,c,beta=theta
    assert c==0
    A=np.array([[1.,rho,1.],[0.,rho,1.],[0.,0.,lam]])
    B=np.array([0.,0.,beta]);Q=np.diag([0.,0.,.03**2])
    P=np.zeros((3,3))
    for t in range(70):
        K=P[:,0]/(P[0,0]+sensor_sd**2)
        P=P-np.outer(K,P[0])
        if t<69:P=A@P@A.T+Q
    F=np.linalg.matrix_power(A,17);G=np.linalg.matrix_power(A,16)
    V=float((F@P@F.T+G@Q@G.T)[0,0]);k=float((G@B)[0])
    rows=[]
    for seed,task,ep in episodes:
        # Independent replay verifies the saved estimate really is the known KF.
        mu=np.zeros(3);cov=np.zeros((3,3));estimate_error=0.
        for t in range(70):
            gain=cov[:,0]/(cov[0,0]+sensor_sd**2)
            mu+=gain*(ep['observations'][t]-mu[0])
            cov-=np.outer(gain,cov[0])
            estimate_error=max(estimate_error,float(np.max(np.abs(mu-ep['estimates'][t]))))
            if t<69:
                mu=A@mu+B*ep['commands'][t]
                cov=A@cov@A.T+Q
        assert estimate_error<1e-9
        u=ep['commands'][69];goal=ep['goals'][69]
        b=float((F@mu+G@B*u)[0]-goal)
        error=float(ep['states'][86][0]-goal)
        probability=.5*(math.erf((.3-b)/math.sqrt(2*V))-
                          math.erf((-.3-b)/math.sqrt(2*V)))
        centered_u=u-b/k
        plan=ep['optimizers'][69]['selected_action_plan']
        rows.append({'seed':seed,'task':task,'posterior_mean':mu.tolist(),
            'saved_estimate_error':estimate_error,'last_action':u,
            'discarded_future_action_plan':plan[1:],
            'conditional_terminal_error_mean':b,
            'conditional_terminal_error_mean_squared':b*b,
            'conditional_terminal_mse':b*b+V,
            'conditional_endpoint_success_probability':probability,
            'actual_terminal_error_squared':error*error,
            'actual_endpoint_success':bool(abs(error)<=.3),
            'actual_whole_stop_success':ep['stop_success'],
            'endpoint_centering_last_action':centered_u,
            'endpoint_centering_feasible':bool(abs(centered_u)<=1.)})
    def mean(key):return float(np.mean([r[key] for r in rows]))
    return {'true_theta':theta,'sensor_sd':sensor_sd,'posterior_covariance69':P.tolist(),
        'terminal_variance':V,'last_action_terminal_sensitivity':k,'episodes':rows,
        'summary':{'mean_squared_conditional_terminal_bias':mean('conditional_terminal_error_mean_squared'),
            'mean_predicted_terminal_mse':mean('conditional_terminal_mse'),
            'mean_realized_terminal_squared_error':mean('actual_terminal_error_squared'),
            'mean_conditional_endpoint_success_probability':mean('conditional_endpoint_success_probability'),
            'actual_endpoint_successes':sum(r['actual_endpoint_success'] for r in rows),
            'actual_whole_stop_successes':sum(r['actual_whole_stop_success'] for r in rows),
            'all_endpoint_centering_actions_feasible':all(r['endpoint_centering_feasible'] for r in rows),
            'endpoint_centering_action_range':[min(r['endpoint_centering_last_action'] for r in rows),
                                             max(r['endpoint_centering_last_action'] for r in rows)]}}


def main():
    prior=json.loads((ROOT/'goal_control_v10_results.json').read_text())
    v10=[]
    for p in prior['plants']:
        if p['true_theta'][2]:continue
        episodes=[(r['seed'],j,t['controllers']['oracle_model_EKF'])
                  for r in p['runs'] for j,t in enumerate(r['tasks'])]
        v10.append(diagnostic(p['true_theta'],.2,episodes))
    current=json.loads((ROOT/'precision_information_v11_results.json').read_text())
    v11=[]
    for sd in ('0.2','0.02'):
        episodes=[(r['seed'],j,t['controllers']['known_parameters'])
                  for r in current['runs'] for j,t in enumerate(r['cases'][sd]['tasks'])]
        v11.append(diagnostic(current['meta']['plant'],float(sd),episodes))
    print(json.dumps({'scope':'Independent saved known-KF mean/variance decomposition; '
        'endpoint centering is not a whole-path stopping controller; no canonical edits',
        'v10':v10,'v11':v11},indent=2,allow_nan=False))


if __name__=='__main__':main()
