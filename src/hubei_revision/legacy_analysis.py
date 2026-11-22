"""Paired uncertainty for disclosed old-code controls and source-change audit."""
from pathlib import Path
import ast
import hashlib
import math
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import t,wilcoxon
from .common import ROOT,read,save,sha,utc
from .analyses import refs,table
from .engine import protocol

def check_analysis_scope():
    before=ROOT/'docs/analysis_before_legacy_addendum.py.txt';after=ROOT/'src/hubei_revision/analyses.py'
    def functions(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text(encoding='utf8')).body if isinstance(node,ast.FunctionDef)}
    old,new=functions(before),functions(after)
    unchanged=['seed_statistics','recompute_derived','robustness','graphs']
    assert all(old[name]==new[name] for name in unchanged)
    save(ROOT/'reports/analysis_addendum_change_scope.json',dict(status='PASS',original_source_sha256=sha(before),
        current_source_sha256=sha(after),unchanged_functions=unchanged,
        changed_functions=['scenarios','complexity'],scope='Rerun scenarios and complexity after legacy completion; unaffected analyses retain their original source archive.'))

def legacy_statistics():
    cfg=protocol();ours={entry['seed']:pd.read_csv(path/'test_predictions.csv') for entry,path in refs('E03','seed','Ours')}
    other={}
    for entry,path in refs('LEGACY','seed'):
        other.setdefault(entry['model'],{})[entry['seed']]=pd.read_csv(path/'test_predictions.csv')
    assert len(other)==6 and all(set(values)==set(cfg['seeds']) for values in other.values())
    tests=[];daily=[];summary=[];rng=np.random.default_rng(20261120)
    output=ROOT/'results/paired_bootstrap_legacy';output.mkdir(parents=True,exist_ok=True)
    for name,values in sorted(other.items()):
        losses=[]
        for metric in ['mae_gwh','rmse_gwh','mape_pct']:
            from .units import metrics_gwh
            measurements=np.array([metrics_gwh(values[seed].true_gwh,values[seed].pred_gwh,'GWh')[metric] for seed in cfg['seeds']])
            mean=float(measurements.mean());sd=float(measurements.std(ddof=1));half=float(t.ppf(.975,4)*sd/math.sqrt(5))
            summary.append(dict(model=name,metric=metric,n_seeds=5,mean=mean,sd=sd,seed_ci95_low=mean-half,seed_ci95_high=mean+half))
        for seed in cfg['seeds']:
            a,b=ours[seed],values[seed]
            assert a.date.tolist()==b.date.tolist() and np.abs(a.true_gwh-b.true_gwh).max()<1e-8
            difference=np.abs(a.pred_gwh-a.true_gwh).to_numpy()-np.abs(b.pred_gwh-b.true_gwh).to_numpy()
            losses.append(difference)
            daily.extend(dict(baseline=name,seed=seed,date=date,difference_gwh=float(value)) for date,value in zip(a.date,difference))
        losses=np.stack(losses);observed=float(losses.mean());n=losses.shape[1]
        for block in [3,7,14]:
            seed_draw=rng.integers(0,5,size=(5000,5));starts=rng.integers(0,n,size=(5000,math.ceil(n/block)))
            date_draw=((starts[:,:,None]+np.arange(block))%n).reshape(5000,-1)[:,:n]
            bootstrap=losses[seed_draw[:,:,None],date_draw[:,None,:]].mean(axis=(1,2))
            lo,hi=np.quantile(bootstrap,[.025,.975]);p=float((1+np.sum(np.abs(bootstrap-observed)>=abs(observed)))/5001)
            tests.append(dict(baseline=name,block_days=block,n_seeds=5,n_dates=n,
                mean_absolute_loss_difference_gwh=observed,ci95_low_gwh=float(lo),ci95_high_gwh=float(hi),
                centered_bootstrap_two_sided_p=p,implementation_scope='Disclosed legacy local implementation; named-paper fidelity unverified.'))
            tests[-1]['wilcoxon_auxiliary_p']=float(wilcoxon(losses.mean(axis=0)).pvalue)
            np.savez_compressed(output/f'{name}_block{block}.npz',bootstrap_mean_loss_difference_gwh=bootstrap)
    local=table('Table_Legacy_Significance_Tests',tests);table('Legacy_Paired_Daily_Losses',daily)
    table('Table_Legacy_Seed_Statistics',summary)
    previous=pd.read_csv(ROOT/'tables/Table_E03_Significance_Tests.csv')
    previous=previous.rename(columns={'holm_p_primary_block7':'holm_p_original_family7'})
    combined=pd.concat([previous,local],ignore_index=True);primary=combined.block_days==7
    combined.loc[primary,'holm_p_expanded_family13']=multipletests(combined.loc[primary,'centered_bootstrap_two_sided_p'],method='holm')[1]
    assert int(primary.sum())==13
    table('Table_All_Implemented_Significance_Tests',combined)
    check_analysis_scope()
    save(ROOT/'reports/legacy_statistics_manifest.json',dict(status='PASS',created=utc(),models=list(other),
        retained_seeds=cfg['seeds'],bootstrap_replicates=5000,blocks=[3,7,14],expanded_family=13,
        source_sha256=sha(__file__),addendum_sha256=sha(ROOT/'configs/legacy_comparison_addendum.json')))

if __name__=='__main__':legacy_statistics()
