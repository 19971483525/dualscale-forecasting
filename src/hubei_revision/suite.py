"""Fixed experiment schedule. Every successful run is reusable by full signature."""
from pathlib import Path
import traceback
import pandas as pd
from .common import ROOT,read,save,sha,utc
from .engine import protocol,run_training,ensure_test

BLOCKED={
    'MG-ASTL':'Existing old code is a convolutional surrogate; exact multi-task decomposition/graph implementation and compatible regional inputs are not verified.',
    'TL-STLF':'Existing old code is a Mixer surrogate; a verified method specification, source-domain observations and transfer procedure are missing.',
    'LoadFormer':'The cited IEEE paper is identifiable, but the old generic PatchTransformer has not been verified as that method.',
    'STGNN-Load':'Existing old code is a variable Transformer, not a verified implementation of the cited residential spatiotemporal model.',
    'GAT-LoadMixer':'Existing old code is a Mixer; the generic GAT citation does not define this full method.'}

def selected(name,fold=None):
    return read(ROOT/'manifests'/('selected_'+(fold+'_' if fold else '')+name+'.json'))

def select_trials(group,label,name,fold=None):
    cfg=protocol();trials=[]
    for i,opt in enumerate(cfg['hpo']):
        run_dir=run_training(group,label+'trial'+str(i),name,opt,42,fold,test_after=False)
        trials.append(dict(trial=i,run_dir=str(run_dir),options=opt,
            validation_mae_gwh=read(run_dir/'metrics.json')['val']['mae_gwh']))
    choice=min(trials,key=lambda row:(row['validation_mae_gwh'],row['trial']))
    result=dict(**choice,trials=trials,selection='Validation MAE only; test is evaluated after the choice is saved.')
    key=(label.rstrip('_')+'_' if label else '')+name
    save(ROOT/'manifests'/('selected_'+key+'.json'),result)
    ensure_test(Path(choice['run_dir']))
    return result

def run(group):
    cfg=protocol();assert cfg['stage']=='FROZEN'
    for rel,digest in cfg['numerical_source_sha256'].items():assert sha(ROOT/'src/hubei_revision'/rel)==digest,rel
    save(ROOT/'manifests/current_stage.json',dict(status='RUNNING',group=group,started=utc()))
    if group=='E01':
        save(ROOT/'reports/unverified_related_methods.json',BLOCKED)
        for name in cfg['models']:select_trials(group,'',name)
        candidates=[(selected(name)['validation_mae_gwh'],name) for name in cfg['models'] if name!='Ours']
        save(ROOT/'manifests/strongest_deep.json',dict(model=min(candidates)[1],selection='Validation MAE only'))
    elif group=='E02':
        from .traditional import run_traditional,METHODS
        candidates=[]
        for name in METHODS:
            run_dir=run_traditional(name)
            candidates.append((read(run_dir/'metrics.json')['val']['mae_gwh'],name,str(run_dir)))
        best=min(candidates)
        save(ROOT/'manifests/strongest_traditional.json',dict(validation_mae_gwh=best[0],model=best[1],run_dir=best[2]))
    elif group=='E03':
        for name in cfg['models']:
            for seed in cfg['seeds']:run_training(group,'seed',name,selected(name)['options'],seed)
    elif group=='E04':
        base=selected('Ours')['options']
        variants={**{'K'+str(k):dict(topk=k) for k in [2,4,6,8,12,16]},
            **{'prior'+str(v):dict(prior_weight=v) for v in [0.,.25,1.,2.]},
            **{'embed'+str(v):dict(embedding_dim=v) for v in [8,16,32,64]},
            'relation_static':dict(relation='static'),'relation_dynamic':dict(no_static_prior=True),
            'relation_no_topk':dict(no_topk=True),'relation_attention':dict(relation='attention'),
            'relation_mlp':dict(relation='mlp'),'relation_full':{}}
        for variant,opt in variants.items():
            for seed in cfg['graph_sensitivity_seeds']:run_training(group,variant,'Ours',{**base,**opt},seed)
    elif group=='E05':
        base=selected('Ours')['options']
        for mode in ['F0','F1','F2','F3','F4','F5','F6','F7','F8']:
            for seed in cfg['seeds']:run_training(group,mode,'Ours',{**base,'fusion':mode},seed)
        for seed in cfg['seeds']:run_training(group,'no_flat','Ours',{**base,'no_flat':True},seed)
    elif group=='E06':
        base=selected('Ours')['options']
        variants={
            'full':{},'no_intraday':dict(no_intraday=True),'no_graph':dict(no_graph=True),
            'no_anchor':dict(no_anchor=True),'no_persistence':dict(anchor='no_persistence'),
            'no_industry_anchor':dict(anchor='no_industry_anchor'),
            'no_reconciliation':dict(no_reconciliation=True),'direct':dict(target_mode='direct'),
            'delta':dict(target_mode='delta'),'no_sales':dict(remove=['sales_total']),
            'no_all_industry':dict(remove=['all_industry_total']),
            'no_both_totals':dict(remove=['sales_total','all_industry_total']),
            'no_totals_derived':dict(remove=['sales_total','all_industry_total','sales_gap','sales_ratio','sector_sum','load_minus_sector_sum']),
            'no_aggregate_nodes':dict(remove=['sales_total','all_industry_total','primary_industry','secondary_industry','tertiary_industry','residential_total']),
            'no_load_aux':dict(aux_weight=0.),'no_industry_aux':dict(industry_aux_weight=0.),
            'no_both_aux':dict(aux_weight=0.,industry_aux_weight=0.)}
        for variant,opt in variants.items():
            for seed in cfg['seeds']:run_training(group,variant,'Ours',{**base,**opt},seed)
    elif group=='E08':
        from .traditional import run_traditional
        for label,fold in cfg['rolling_folds'].items():
            for name in cfg['seasonal_models']:
                result=select_trials(group,label+'_',name,fold)
                for seed in cfg['seeds']:run_training(group,label+'_seed',name,result['options'],seed,fold)
            for name in ['Last-week Seasonal Naive','XGBoost','LightGBM']:
                run_traditional(name,fold,label)
    elif group in ['E07','E09','E10','E11']:
        from .analyses import analyze
        analyze(group)
    else:raise ValueError(group)
    save(ROOT/f'manifests/{group}_status.json',dict(status='PARTIAL' if group in ['E01','E03','E08','E10'] else 'PASS',
        numerical_experiments='PASS',unverified_methods='BLOCKED' if group in ['E01','E03','E08','E10'] else None,
        group=group,finished=utc(),note='Unverified methods are excluded from numerical comparisons; cross-region data unavailable.'))
    print('GROUP_COMPLETE',group,utc(),flush=True)

def run_groups(groups):
    for group in groups:
        try:run(group)
        except BaseException as error:
            save(ROOT/f'manifests/{group}_status.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),time=utc()))
            raise
