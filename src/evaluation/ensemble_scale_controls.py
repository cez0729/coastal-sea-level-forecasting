"""Post-hoc controls on frozen caches, without retraining or test selection.

All distinct-seed pairs are retained. Interval estimates condition on the saved
expert bank and resample time jointly, not overlapping pairs independently.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'controls'
SEEDS = [42, 123, 2024, 2025, 3407]
REPS = 5000
BLOCKS = [84, 168, 336]
SOURCES = {
    'historical_7': ROOT/'data/prediction_caches/historical_7',
    'external_10': ROOT/'data/prediction_caches/external_10',
}

def load(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].astype(np.float64) for k in
                ['true_residual','pred_residual','eta_pred','sigma_residual']} | {
                'multi': a['multi_pred_states'][...,0].astype(np.float64),
                'stations': a['station_ids'].astype(str)}

def bootstrap_totals(values, block, seed):
    """Exact n-origin paired circular blocks; final block truncated to n."""
    n = len(values)
    if n <= block:
        raise ValueError('Insufficient origins for requested block')
    ext = np.concatenate([values, values[:block]], axis=0)
    cum = np.vstack([np.zeros((1,values.shape[1])), np.cumsum(ext,axis=0)])
    full, rem = divmod(n, block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,n,size=(REPS,full + bool(rem)))
    sums = cum[starts[:,:full]+block] - cum[starts[:,:full]]
    result = sums.sum(axis=1)
    if rem:
        last = starts[:,-1]
        result += cum[last+rem] - cum[last]
    return result

def errors(y,p):
    e=(y-p)**2
    return np.column_stack([e.sum(axis=(1,2)),e[...,-1].sum(axis=1),
                            np.abs(y-p).sum(axis=(1,2))])

def point(y,stats):
    den=np.sum((y-y.mean())**2)
    dl=np.sum((y[...,-1]-y[...,-1].mean())**2)
    totals=stats.sum(axis=0)
    return {'sequence_R2':1-totals[0]/den,'lead24_R2':1-totals[1]/dl,
            'sequence_MSE':totals[0]/y.size,
            'sequence_RMSE':np.sqrt(totals[0]/y.size),
            'sequence_MAE':totals[2]/y.size}

def prob_origin(y,mu,sigma):
    sigma=np.broadcast_to(np.maximum(sigma,1e-12),y.shape)
    z=(y-mu)/sigma
    crps=sigma*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    nll=.5*z*z+np.log(sigma)+.5*np.log(2*np.pi)
    return np.column_stack([crps.mean(axis=(1,2)),nll.mean(axis=(1,2)),
        (np.abs(z)<=.6744897501960817).mean(axis=(1,2)),
        (np.abs(z)<=1.959963984540054).mean(axis=(1,2)),
        (2*1.959963984540054*sigma).mean(axis=(1,2))])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ensemble_rows=[]; pair_rows=[]; ensemble_ci=[]; scale_rows=[]; scale_ci=[]; files=[]
    for region,root in SOURCES.items():
        bank=[]; vals=[]
        for seed in SEEDS:
            tp=root/f'seed_{seed}'/'test_predictions.npz'
            vp=root/f'seed_{seed}'/'validation_predictions.npz'
            bank.append(load(tp)); vals.append(load(vp));files.extend([tp,vp])
        y=bank[0]['true_residual']
        for a,v in zip(bank,vals):
            assert np.array_equal(y,a['true_residual']), 'Across-seed test target mismatch'
            assert np.array_equal(vals[0]['true_residual'],v['true_residual']), 'Across-seed validation target mismatch'
            assert np.array_equal(bank[0]['stations'],a['stations']), 'Station order mismatch'
            assert np.array_equal(a['stations'],v['stations']), 'Validation/test station order mismatch'
            assert not np.array_equal(a['true_residual'],v['true_residual']), 'Validation identical to test'
            assert all(np.isfinite(a[k]).all() and np.isfinite(v[k]).all() for k in
                       ['true_residual','pred_residual','eta_pred','multi','sigma_residual'])
        stats={k:[] for k in ['Eta+Eta','Multi+Multi','Eta+Multi_distinct','HS-DT_distinct','HS-DT_same_seed','Eta+Multi_same_seed']}
        for i,j in combinations(range(5),2):
            ps={'Eta+Eta':(bank[i]['eta_pred']+bank[j]['eta_pred'])/2,
                'Multi+Multi':(bank[i]['multi']+bank[j]['multi'])/2}
            for name,p in ps.items():
                st=errors(y,p);stats[name].append(st)
                pair_rows.append({'region':region,'model':name,'seed1':SEEDS[i],'seed2':SEEDS[j],**point(y,st)})
            for a,b in [(i,j),(j,i)]:
                p=(bank[a]['eta_pred']+bank[b]['multi'])/2
                st=errors(y,p);stats['Eta+Multi_distinct'].append(st)
                pair_rows.append({'region':region,'model':'Eta+Multi_distinct','seed1':SEEDS[a],'seed2':SEEDS[b],**point(y,st)})
                h=p.copy();h[...,-1]=bank[b]['multi'][...,-1]
                stats['HS-DT_distinct'].append(errors(y,h))
        for a in bank:
            p=(a['eta_pred']+a['multi'])/2
            stats['Eta+Multi_same_seed'].append(errors(y,p))
            p[...,-1]=a['multi'][...,-1]
            stats['HS-DT_same_seed'].append(errors(y,p))
        averaged={k:np.mean(v,axis=0) for k,v in stats.items()}
        for name,st in averaged.items():
            ensemble_rows.append({'region':region,'model':name,'pair_count':len(stats[name]),**point(y,st)})
        compares=[('Eta+Multi_distinct','Eta+Eta'),('Eta+Multi_distinct','Multi+Multi'),
                  ('HS-DT_distinct','Multi+Multi'),('HS-DT_same_seed','Multi+Multi')]
        yy=np.column_stack([y.sum(axis=(1,2)),(y*y).sum(axis=(1,2)),
                             y[...,-1].sum(axis=1),(y[...,-1]**2).sum(axis=1)])
        for c,b in compares:
            reduced=averaged[b][:,:2]-averaged[c][:,:2]
            for block in BLOCKS:
                total=bootstrap_totals(np.column_stack([yy,reduced]),block,20261009+block)
                den=total[:,1]-total[:,0]**2/y.size
                dl=total[:,3]-total[:,2]**2/y[...,-1].size
                for h,metric,d in [(0,'sequence_R2',den),(1,'lead24_R2',dl)]:
                    draws=total[:,4+h]/d
                    ensemble_ci.append({'region':region,'comparison':c+' minus '+b,'metric':metric,
                        'block_hours':block,'delta':point(y,averaged[c])[metric]-point(y,averaged[b])[metric],
                        'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),
                        'replicates':REPS,'inference':'time uncertainty conditional on fixed expert bank'})
        fixed=[]; dynamic=[]
        for seed,a,v in zip(SEEDS,bank,vals):
            err=v['true_residual']-v['pred_residual']
            const=float(np.sqrt(np.mean(err**2)))
            factor=float(np.clip(np.sqrt(np.mean((err/np.maximum(v['sigma_residual'],1e-12))**2)),.25,4))
            f=prob_origin(y,a['pred_residual'],const)
            d=prob_origin(y,a['pred_residual'],a['sigma_residual']*factor)
            fixed.append(f);dynamic.append(d)
            for name,st in [('fixed',f),('dynamic',d)]:
                scale_rows.append({'region':region,'seed':seed,'scale':name,'validation_fixed_sigma':const,
                    'validation_dynamic_multiplier':factor,**dict(zip(['CRPS','NLL','coverage_50','coverage_95','width_95'],st.mean(axis=0)))})
        diff=np.mean(np.stack(fixed)-np.stack(dynamic),axis=0)
        for block in BLOCKS:
            draws=bootstrap_totals(diff,block,20261009+block)/len(y)
            for j,name in enumerate(['CRPS','NLL','coverage_50','coverage_95','width_95']):
                scale_ci.append({'region':region,'metric':name,'comparison':'fixed minus dynamic',
                    'delta':diff[:,j].mean(),'ci_low':np.quantile(draws[:,j],.025),
                    'ci_high':np.quantile(draws[:,j],.975),'block_hours':block,'replicates':REPS,
                    'seed_positive_count':sum((f[:,j]-d[:,j]).mean()>0 for f,d in zip(fixed,dynamic))})
        print('REGION',region,'origins',len(y),'stations',y.shape[1],flush=True)
    for name,rows in [('ensemble_scores',ensemble_rows),('ensemble_all_pairs',pair_rows),
                      ('ensemble_bootstrap',ensemble_ci),('scale_scores_by_seed',scale_rows),('scale_bootstrap',scale_ci)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    manifest={'seeds':SEEDS,'blocks':BLOCKS,'replicates':REPS,'model_training':False,
        'cross_supervision_primary':'all 20 distinct-seed Eta/Multi pairs, 50/50 at every lead',
        'homogeneous_controls':'all 10 unordered distinct-seed pairs per family',
        'pair_dependence':'expert bank held fixed; same sampled time blocks across every pair and seed',
        'scale_fit':'validation RMSE for constant sigma; existing clipped global multiplier for dynamic sigma',
        'evidence':'posthoc diagnostic; no new untouched confirmation',
        'input_files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nENSEMBLE SCORES\n',pd.DataFrame(ensemble_rows).to_string(index=False),flush=True)
    print('\nENSEMBLE PRIMARY CI\n',pd.DataFrame(ensemble_ci).query('block_hours == 168').to_string(index=False),flush=True)
    print('\nSCALE MEANS\n',pd.DataFrame(scale_rows).groupby(['region','scale'])[['CRPS','NLL','coverage_50','coverage_95','width_95']].mean().to_string(),flush=True)
    print('\nSCALE PRIMARY CI\n',pd.DataFrame(scale_ci).query('block_hours == 168').to_string(index=False),flush=True)

if __name__ == '__main__':
    main()
