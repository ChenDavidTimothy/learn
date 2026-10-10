"""Cycle 51: posthoc diagnostic rejection of unsupported sensor repairs.

Analyze exact Cycle49 frozen external POMDP results. A spatial/skew
residual can flag when *none* of the 48 supplied invertible channel
transformations explains target observations. We report a broad sweep of
thresholds instead of selecting a favorable one. This analysis is
EXPLORATORY and uses already-inspected outcomes; it is NOT independent
proof of reliable open-set detection or a novel algorithm.

No model retraining or additional test environment interaction.
"""
import json,numpy as np
SHIFTS=('identity','invert_rgb','brighten_48','permute_rgb','erase_blue')
THRESHOLDS=(.005,.01,.02,.05,.1,.25,1.,3.)
def main():
    data=json.load(open('research/cycle49_palette_alignment_results.json'))
    rows=data['runs']
    report={'protocol':{'source':'previously inspected Cycle49',
       'seeds':len(rows),'loss':'source-target spatial template MSE plus 0.25 standardized skewness mismatch',
       'thresholds':THRESHOLDS,
       'selection_status':'posthoc exploratory; needs independent task-family replication',
       'decision':'if fit loss <= threshold use spatial signed-channel repair, otherwise use frozen raw policy and flag need for a new mechanism',
       'no_new_algorithm':True,'no_extra_interactions':True},'results':{}}
    for threshold in THRESHOLDS:
        q={}
        for shift in SHIFTS:
            vals=[];accept=[];raw=[];adapt=[]
            for row in rows:
                case=row['cases'][shift]
                fit=case['selected_transform']['fit_objective']
                accepted=bool(fit<=threshold)
                results=case['scores']
                val=results['spatial_signed_align' if accepted else 'raw']['mean_completed_episode_return']
                vals.append(val);accept.append(accepted)
                raw.append(results['raw']['mean_completed_episode_return'])
                adapt.append(results['spatial_signed_align']['mean_completed_episode_return'])
            q[shift]={'accepted_fraction':float(np.mean(accept)),
                'gated_return_mean':float(np.mean(vals)),
                'raw_return_mean':float(np.mean(raw)),
                'always_repair_mean':float(np.mean(adapt)),
                'fit_loss_range':[float(min(row['cases'][shift]['selected_transform']['fit_objective'] for row in rows)),
                                  float(max(row['cases'][shift]['selected_transform']['fit_objective'] for row in rows))]}
        report['results'][str(threshold)]=q
        print('THRESHOLD',threshold,[
            (k,round(v['accepted_fraction'],2),round(v['gated_return_mean'],3))
            for k,v in q.items()],flush=True)
    with open('research/cycle51_open_set_rejection_results.json','w') as f:json.dump(report,f,indent=2)
    print('SAVED')
if __name__=='__main__':main()
