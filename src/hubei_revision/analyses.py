"""Paired statistics and auditable post-training analyses; no score filtering."""
from pathlib import Path
import itertools
import math
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import t,wilcoxon
from statsmodels.stats.multitest import multipletests
from .common import ROOT,read,save,sha,utc
from .engine import protocol,load_data,load_run,predict,inputs
from .legacy.data import DailyForecastDataset
from .units import metrics_gwh

def table(name,rows):
    df=pd.DataFrame(rows);path=ROOT/'tables'/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig');return df

def refs(group,variant=None,model=None):
    result=[]
    for path in sorted((ROOT/'experiments'/group).glob('*.json')):
        entry=read(path)
        if variant is not None and entry['variant']!=variant:continue
        if model is not None and entry['model']!=model:continue
        run_dir=Path(entry['run_dir']);manifest=read(run_dir/'run_manifest.json')
        assert manifest['status']=='PASS'
        result.append((entry,run_dir))
    return result

def chosen(name):return Path(read(ROOT/'manifests'/('selected_'+name+'.json'))['run_dir'])

def metadata(group,**details):
    save(ROOT/f'reports/{group}_analysis_manifest.json',dict(status='PASS',created=utc(),
        analysis_source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'configs/unified_protocol.json'),**details))

def seed_statistics():
    cfg=protocol();predictions={};rows=[]
    for entry,run_dir in refs('E03'):
        frame=pd.read_csv(run_dir/'test_predictions.csv')
        predictions.setdefault(entry['model'],{})[entry['seed']]=frame
        rows.append(dict(model=entry['model'],seed=entry['seed'],run_dir=str(run_dir),
            **metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')))
    for name in cfg['models']:assert set(predictions[name])==set(cfg['seeds'])
    raw=table('E03_All_Seed_Metrics',rows);summary=[]
    for name,group in raw.groupby('model'):
        for metric in ['mae_gwh','rmse_gwh','mape_pct']:
            values=group[metric].to_numpy();mean=float(values.mean());sd=float(values.std(ddof=1))
            half=float(t.ppf(.975,len(values)-1)*sd/math.sqrt(len(values)))
            summary.append(dict(model=name,metric=metric,n_seeds=len(values),mean=mean,sd=sd,
                seed_ci95_low=mean-half,seed_ci95_high=mean+half,
                scope='Training randomness on the fixed inspected test period; not cross-region uncertainty.'))
    table('Table_E03_Seed_Statistics',summary)
    tests=[];daily=[];acfs=[];rng=np.random.default_rng(20261120)
    destination=ROOT/'results/paired_bootstrap';destination.mkdir(parents=True,exist_ok=True)
    for name in cfg['models']:
        if name=='Ours':continue
        differences=[]
        for seed in cfg['seeds']:
            ours=predictions['Ours'][seed];other=predictions[name][seed]
            assert ours.date.tolist()==other.date.tolist()
            assert np.abs(ours.true_gwh-other.true_gwh).max()<1e-8
            loss=np.abs(ours.pred_gwh-ours.true_gwh).to_numpy()-np.abs(other.pred_gwh-other.true_gwh).to_numpy()
            differences.append(loss)
            daily.extend(dict(baseline=name,seed=seed,date=date,difference_gwh=float(value)) for date,value in zip(ours.date,loss))
        differences=np.stack(differences);average=differences.mean(axis=0);n=len(average)
        for lag in range(1,15):
            acfs.append(dict(baseline=name,lag_days=lag,paired_loss_difference_acf=float(np.corrcoef(average[:-lag],average[lag:])[0,1])))
        for block in [3,7,14]:
            seed_draw=rng.integers(0,len(cfg['seeds']),size=(5000,len(cfg['seeds'])))
            starts=rng.integers(0,n,size=(5000,math.ceil(n/block)))
            date_draw=((starts[:,:,None]+np.arange(block))%n).reshape(5000,-1)[:,:n]
            bootstrap=differences[seed_draw[:,:,None],date_draw[:,None,:]].mean(axis=(1,2))
            observed=float(average.mean());lo,hi=np.quantile(bootstrap,[.025,.975])
            p=float((1+np.sum(np.abs(bootstrap-observed)>=abs(observed)))/5001)
            auxiliary=float(wilcoxon(average).pvalue) if np.any(average!=0) else 1.
            tests.append(dict(baseline=name,block_days=block,n_seeds=5,n_dates=n,
                mean_absolute_loss_difference_gwh=observed,ci95_low_gwh=float(lo),ci95_high_gwh=float(hi),
                centered_bootstrap_two_sided_p=p,wilcoxon_auxiliary_p=auxiliary,
                bootstrap_scope='Joint paired resampling of seed pairs and circular date blocks; 5000 replicates.',
                interpretation='Negative favors Ours. Wilcoxon does not account for daily serial dependence.'))
            np.savez_compressed(destination/f'{name}_block{block}.npz',bootstrap_mean_loss_difference_gwh=bootstrap)
    results=pd.DataFrame(tests)
    primary=results.block_days==7
    results.loc[primary,'holm_p_primary_block7']=multipletests(results.loc[primary,'centered_bootstrap_two_sided_p'],method='holm')[1]
    table('Table_E03_Significance_Tests',results);table('E03_Paired_Daily_Losses',daily);table('E03_Loss_Difference_ACF',acfs)
    metadata('E03',retained_seeds=cfg['seeds'],models=cfg['models'],
        note='No requirement that Ours ranks first. Unverified TL-STLF is not substituted with an unrelated model.')

def scenarios():
    data=load_data();df=data['df'];dates=pd.to_datetime(df.date)
    train=(dates>=data['bounds']['train'][0])&(dates<=data['bounds']['train'][1])
    hot=float(df.loc[train,'temp_max'].quantile(.95));cold=float(df.loc[train,'temp_min'].quantile(.05))
    change=df.daily_load.diff().abs();jump=float(change[train].quantile(.9))
    holiday=(dates>='2025-01-28')&(dates<='2025-02-04')
    makeup=dates.isin(pd.to_datetime(['2025-01-26','2025-02-08']))
    weekend=(dates.dt.weekday>=5)&~makeup
    masks=dict(ordinary_workday=~weekend&~holiday,weekend=weekend&~holiday,
        statutory_holiday=holiday,spring_festival=holiday,
        spring_transition=((dates>='2025-01-21')&(dates<='2025-02-11'))&~holiday,
        high_temperature=df.temp_max>=hot,low_temperature=df.temp_min<=cold,large_day_change=change>=jump)
    table('E07_Scenario_Definitions',[dict(subset=key,training_hot_q95_c=hot,training_cold_q05_c=cold,
        training_change_q90_gwh=jump*10,overlap='Ordinary/weekend/holiday mutually exclusive; weather and transition may overlap.',
        source='State Council 2025 holiday arrangement; Jan28-Feb4 festival, Jan26/Feb8 makeup workdays.',
        source_url='https://www.gov.cn/zhengce/content/202411/content_6986382.htm') for key in masks])
    entries=[]
    for group,variant,label in [('E03','seed','Ours'),('E04','relation_static','Static correlation graph'),
                               ('E06','no_persistence','No explicit persistence anchor')]:
        for entry,run_dir in refs(group,variant,'Ours'):
            entries.append((label,entry['seed'],run_dir))
    best_deep=read(ROOT/'manifests/strongest_deep.json')['model']
    for entry,run_dir in refs('E03','seed',best_deep):entries.append((best_deep,entry['seed'],run_dir))
    traditional=read(ROOT/'manifests/strongest_traditional.json')
    entries.append((traditional['model'],42,Path(traditional['run_dir'])))
    if (ROOT/'manifests/LEGACY_status.json').exists() and read(ROOT/'manifests/LEGACY_status.json').get('status')=='PASS':
        for entry,run_dir in refs('LEGACY','seed','Legacy_TL-STLF'):
            entries.append(('TL-STLF (legacy local)',entry['seed'],run_dir))
    rows=[]
    for label,seed,run_dir in entries:
        frame=pd.read_csv(run_dir/'test_predictions.csv');fdates=pd.to_datetime(frame.date)
        for subset,mask in masks.items():
            part=frame[fdates.isin(dates[mask])]
            metric=metrics_gwh(part.true_gwh,part.pred_gwh,'GWh') if len(part) else dict(mae_gwh=np.nan,rmse_gwh=np.nan,mape_pct=np.nan,n=0)
            rows.append(dict(model=label,seed=seed,subset=subset,run_dir=str(run_dir),**metric,
                qualification='No samples' if len(part)==0 else 'Fewer than 10 dates; descriptive' if len(part)<10 else ''))
    raw=table('E07_All_Scenario_Seed_Results',rows)
    grouped=raw.groupby(['model','subset']).agg(n_dates=('n','first'),n_runs=('seed','size'),
        mae_gwh_mean=('mae_gwh','mean'),mae_gwh_sd=('mae_gwh','std'),
        rmse_gwh_mean=('rmse_gwh','mean'),rmse_gwh_sd=('rmse_gwh','std'),
        mape_pct_mean=('mape_pct','mean'),mape_pct_sd=('mape_pct','std')).reset_index()
    table('Table_E07_Scenario_Results',grouped)
    metadata('E07',note='No pre-Jan28 test targets exist in this protocol. No artificial pre-festival or summer test samples. Temperature and load-change thresholds use training dates only. Large-change membership is retrospective descriptive labeling.')

def recompute_derived(frame):
    q=frame[[f'q_{i:02d}' for i in range(96)]]
    frame['daily_load']=.25*q.sum(axis=1);frame['load_peak']=q.max(axis=1)
    frame['load_valley']=q.min(axis=1);frame['load_mean_96']=q.mean(axis=1);frame['load_std_96']=q.std(axis=1)
    frame['sales_gap']=frame.daily_load-frame.sales_total;frame['sales_ratio']=frame.daily_load/frame.sales_total
    if 'sector_sum' in frame:
        frame['sector_sum']=frame[['primary_industry','secondary_industry','tertiary_industry','residential_total']].sum(axis=1)
    if 'load_minus_sector_sum' in frame:frame['load_minus_sector_sum']=frame.daily_load-frame.sector_sum
    return frame

def robustness():
    run_dir=chosen('Ours');model,data,run_cfg=load_run(run_dir);cfg=protocol()
    original=data['df'];dates=pd.to_datetime(original.date);history=cfg['history_days']
    training=original[(dates>=data['bounds']['train'][0])&(dates<=data['bounds']['train'][1])]
    q=[f'q_{i:02d}' for i in range(96)];weather=['temp_min','temp_avg','temp_max']
    industry=data['industry'];truth=original.set_index('date').daily_load*10
    destination=ROOT/'results/predictions/E09';destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    def export(label,frame,perturbation_seed,category,strength):
        frame=frame.copy();frame['true_gwh']=frame.date.dt.strftime('%Y-%m-%d').map(truth)
        assert frame.true_gwh.notna().all()
        frame['true_model']=frame.true_gwh/10
        path=destination/f'{label}_s{perturbation_seed}.csv';frame.to_csv(path,index=False,encoding='utf-8-sig')
        saved=pd.read_csv(path)
        rows.append(dict(condition=label,category=category,strength=strength,perturbation_seed=perturbation_seed,
            model_seed=run_cfg['seed'],source_run_dir=str(run_dir),prediction_path=str(path),
            **metrics_gwh(saved.true_gwh,saved.pred_gwh,'GWh')))
    def evaluate_frame(label,frame,seed,category,strength):
        perturbed=load_data(run_cfg['options'],override=frame,fitted=data['fitted'])
        pred=predict(model,perturbed,'test',run_cfg['device'])[0]
        export(label,pred,seed,category,strength)
    evaluate_frame('unperturbed',original.copy(),0,'reference',0.)
    for category in ['intraday','industry','weather']:
        for strength in [0.,.01,.03,.05,.1]:
            for seed in cfg['perturbation_seeds']:
                rng=np.random.default_rng(seed);frame=original.copy()
                if strength:
                    if category=='intraday':
                        frame[q]=np.maximum(0,original[q].to_numpy()+rng.normal(size=(len(frame),96))*training[q].std().to_numpy()*strength)
                    elif category=='industry':
                        scale=float(training.sales_total.std()/training.sales_total.mean())
                        factor=np.maximum(.1,1+rng.normal(size=(len(frame),1))*strength*scale)
                        frame[industry]=original[industry].to_numpy()*factor
                    else:
                        # One shared daily offset preserves min <= mean <= max.
                        frame[weather]=original[weather].to_numpy()+rng.normal(size=(len(frame),1))*float(training.temp_avg.std())*strength
                    frame=recompute_derived(frame)
                evaluate_frame(category+'_noise_'+str(strength),frame,seed,category,strength)
    for condition in ['missing_1','missing_3','missing_7','random_missing','delay_1','delay_2','delay_3','delay_7','weather_outlier']:
        for seed in cfg['perturbation_seeds'] if condition in ['random_missing','weather_outlier'] else [0]:
            rng=np.random.default_rng(seed);frames=[]
            for index in data['indices']['test']:
                target=pd.Timestamp(data['raw']['dates'][index]);day=int(np.where(dates==target)[0][0])
                frame=original.copy()
                if condition=='random_missing':
                    missing=rng.random(history)<.1
                    for offset in range(history):
                        if not missing[offset]:continue
                        row=day-history+offset
                        value=training[industry].mean().to_numpy() if offset==0 else frame.loc[row-1,industry].to_numpy()
                        frame.loc[row,industry]=value
                elif condition=='weather_outlier':
                    frame.loc[day-1,weather]=frame.loc[day-1,weather]+5*float(training.temp_avg.std())
                else:
                    lag=int(condition.rsplit('_',1)[1]);frame.loc[day-lag:day-1,industry]=original.loc[day-lag-1,industry].to_numpy()
                frame=recompute_derived(frame)
                perturbed=load_data(run_cfg['options'],override=frame,fitted=data['fitted'])
                perturbed['indices']['test']=np.array([index]);perturbed['ds']['test']=DailyForecastDataset(perturbed['scaled'],np.array([index]))
                frames.append(predict(model,perturbed,'test',run_cfg['device'])[0])
            export(condition,pd.concat(frames,ignore_index=True),seed,
                'weather_outlier' if condition=='weather_outlier' else 'availability',np.nan)
    result=table('Table_E09_Robustness',rows)
    controls=result[(result.strength==0)&result.category.isin(['reference','intraday','industry','weather'])]
    assert controls.mae_gwh.max()-controls.mae_gwh.min()<.001
    metadata('E09',model_seed=run_cfg['seed'],conditions=len(result),
        note='Frozen seed42 checkpoint; three perturbation seeds, not retrained models. Raw-variable perturbations preserve energy integration and original aggregate relationships. Missing suffix and release lag use the same causal last-observation fill and are not independent evidence. Targets are always the original unperturbed observations. Noise stress tests are not evidence for real changes of statistical accounting methodology.')
    del model;torch.cuda.empty_cache()

def complexity():
    cfg=protocol();names=list(dict.fromkeys(['Ours','iTransformer',read(ROOT/'manifests/strongest_deep.json')['model']]))
    if (ROOT/'manifests/selected_Legacy_TL-STLF.json').exists():
        from .legacy_factory import install
        install();names.append('Legacy_TL-STLF')
    rows=[]
    for name in names:
        run_dir=chosen(name);model,data,run_cfg=load_run(run_dir);device=run_cfg['device'];model.eval()
        loader=torch.utils.data.DataLoader(data['ds']['test'],batch_size=16,shuffle=False)
        batch=inputs({key:value.to(device) for key,value in next(iter(loader)).items()})
        model.zero_grad()
        with torch.backends.cudnn.flags(enabled=False):model(batch)['prediction'][:,:1].sum().backward()
        active=sum(value.numel() for value in model.parameters() if value.grad is not None);model.zero_grad()
        measured={}
        with torch.no_grad():
            for size in [1,16]:
                sample={key:value[:size] for key,value in batch.items()}
                for _ in range(20):model(sample)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
                for _ in range(100):
                    torch.cuda.synchronize();start=time.perf_counter();model(sample);torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                measured.update({f'batch{size}_latency_ms_mean':float(np.mean(times)),
                    f'batch{size}_latency_ms_sd':float(np.std(times,ddof=1)),
                    f'batch{size}_inference_peak_allocated_gpu_mb':torch.cuda.max_memory_allocated()/1024**2})
            try:
                from torch.profiler import profile,ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU],with_flops=True) as profiler:model({key:value[:1] for key,value in batch.items()})
                flops=sum(item.flops for item in profiler.key_averages())
            except Exception as error:flops=np.nan;measured['flops_error']=repr(error)
        manifest=read(run_dir/'run_manifest.json')
        rows.append(dict(model=name,run_dir=str(run_dir),registered_parameters=sum(value.numel() for value in model.parameters()),
            main_path_tensor_parameters=active,profiled_supported_flops=flops,
            flops_scope='Supported-operator lower bound; GRU/attention kernels may be omitted.',
            parameter_scope='Tensor-level main-target connectivity, not count of individual nonzero gradient elements.',
            training_seconds=manifest['training_seconds'],epoch_seconds_mean=manifest['epoch_seconds_mean'],
            whole_run_peak_allocated_gpu_mb=manifest['training_peak_allocated_gpu_mb'],
            run_peak_scope='Whole train/evaluate/reload run; includes fresh-instance replay, not isolated training memory.',
            **measured,**manifest['environment'],**read(run_dir/'metrics.json')['test']))
        del model;torch.cuda.empty_cache()
    traditional=read(ROOT/'manifests/strongest_traditional.json');run_dir=Path(traditional['run_dir'])
    if (run_dir/'best_model.joblib').exists():
        import joblib
        from .traditional import design
        model=joblib.load(run_dir/'best_model.joblib');x=design(load_data(),'test')[0];measured={}
        for size in [1,16]:
            for _ in range(20):model.predict(x[:size])
            times=[]
            for _ in range(100):
                start=time.perf_counter();model.predict(x[:size]);times.append((time.perf_counter()-start)*1000)
            measured.update({f'batch{size}_latency_ms_mean':float(np.mean(times)),f'batch{size}_latency_ms_sd':float(np.std(times,ddof=1))})
        rows.append(dict(model=traditional['model'],device='cpu',run_dir=str(run_dir),checkpoint_bytes=(run_dir/'best_model.joblib').stat().st_size,
            parameter_scope='Neural parameter/FLOP counts not applicable to this estimator.',**measured,**read(run_dir/'metrics.json')['test']))
    table('Table_E10_Complexity',rows)
    metadata('E10',note='20 warmups and 100 GPU-synchronized inference timings per batch size. CPU estimator and GPU networks are labeled by device. No fabricated full FLOP count for unsupported kernels.')

def graphs():
    all_runs=refs('E03','seed','Ours');graphs_by_seed={};rows=[];edges=[];conditions=[]
    def jaccard(a,b):return float(np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1))
    for entry,run_dir in all_runs:
        saved=np.load(run_dir/'graphs.npz');adj=saved['adjacency'];dates=pd.to_datetime(saved['dates']);nodes=saved['nodes']
        graphs_by_seed[entry['seed']]=(adj,dates,nodes)
        for index in range(1,len(adj)):
            rows.append(dict(comparison='adjacent_dates',seed=entry['seed'],date=str(dates[index].date()),
                frobenius_difference=float(np.linalg.norm(adj[index]-adj[index-1])),
                edge_jaccard_including_self=jaccard(adj[index]>0,adj[index-1]>0),density=float((adj[index]>0).mean())))
        for i,source in enumerate(nodes):
            for j,target in enumerate(nodes):
                edges.append(dict(seed=entry['seed'],query_node=source,key_node=target,
                    positive_edge_frequency=float((adj[:,i,j]>0).mean()),mean_weight=float(adj[:,i,j].mean())))
        holiday=(dates>='2025-01-28')&(dates<='2025-02-04')
        weekend=(dates.weekday>=5)&~dates.isin(pd.to_datetime(['2025-02-08']))
        workday=~weekend&~holiday;weekend=weekend&~holiday
        post=(dates>='2025-02-05')&(dates<='2025-02-11')
        for name,a,b in [('workday_vs_weekend',workday,weekend),('festival_vs_post',holiday,post)]:
            assert a.sum()>0 and b.sum()>0
            conditions.append(dict(seed=entry['seed'],comparison=name,n_dates_a=int(a.sum()),n_dates_b=int(b.sum()),
                frobenius_difference=float(np.linalg.norm(adj[a].mean(0)-adj[b].mean(0)))))
    for a,b in itertools.combinations(protocol()['seeds'],2):
        ga,da,na=graphs_by_seed[a];gb,db,nb=graphs_by_seed[b]
        assert np.array_equal(da,db) and np.array_equal(na,nb)
        for index in range(len(ga)):
            rows.append(dict(comparison='between_seeds_same_date',seed_a=a,seed_b=b,date=str(da[index].date()),
                frobenius_difference=float(np.linalg.norm(ga[index]-gb[index])),
                edge_jaccard_including_self=jaccard(ga[index]>0,gb[index]>0)))
    for entry,run_dir in refs('E06','no_aggregate_nodes','Ours'):
        changed=np.load(run_dir/'graphs.npz');original,dates,nodes=graphs_by_seed[entry['seed']]
        keep=[list(nodes).index(name) for name in changed['nodes']]
        common=original[:,keep][:,:,keep];common=common/np.maximum(common.sum(axis=2,keepdims=True),1e-12)
        rows.append(dict(comparison='remove_aggregates_common_subgraph',seed=entry['seed'],
            frobenius_difference=float(np.linalg.norm(common-changed['adjacency'],axis=(1,2)).mean()),
            edge_jaccard_including_self=jaccard(common>0,changed['adjacency']>0)))
    degrees=[]
    for seed,(adj,dates,nodes) in graphs_by_seed.items():
        for index,name in enumerate(nodes):
            degrees.append(dict(seed=seed,node=name,mean_in_degree=float((adj[:,:,index]>0).sum(axis=1).mean()),
                mean_out_degree=float((adj[:,index,:]>0).sum(axis=1).mean()),self_loop_frequency=float((adj[:,index,index]>0).mean())))
    table('Table_E11_Graph_Stability',rows);table('E11_Condition_Graph_Differences',conditions)
    table('E11_Edge_Frequency',edges);table('E11_Node_Degrees',degrees)
    # Before-festival graph context is from validation, explicitly separated.
    run_dir=chosen('Ours');model,data,run_cfg=load_run(run_dir)
    frame,_,adj,att=predict(model,data,'val',run_cfg['device'],graphs=True)
    keep=(pd.to_datetime(frame.date)>='2025-01-21')&(pd.to_datetime(frame.date)<='2025-01-27')
    destination=ROOT/'results/predictions/E11';destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'pre_festival_VALIDATION_context.npz',adjacency=adj[keep],
        dates=frame.loc[keep,'date'].to_numpy(dtype=str),nodes=np.array(data['industry']),split=np.array(['validation']))
    metadata('E11',note='Predictive associations, not industrial causality. Top-K includes diagonal candidates but does not force self edges. Pre-festival context is validation-only and is not counted as unseen test evidence.',
        model_seeds=protocol()['seeds'])
    del model;torch.cuda.empty_cache()

def analyze(group):
    return {'E03':seed_statistics,'E07':scenarios,'E09':robustness,'E10':complexity,'E11':graphs}[group]()
