"""Reconcile every retained run and build tables from real saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import ROOT,read,save,sha,utc
from .engine import protocol
from .analyses import refs,table,seed_statistics
from .units import metrics_gwh

def collect(group,variant=None):
    rows=[]
    for entry,run_dir in refs(group,variant):
        manifest=read(run_dir/'run_manifest.json');metric=read(run_dir/'metrics.json')
        rows.append(dict(**entry,parameters=manifest.get('parameters'),
            best_epoch=manifest.get('best_epoch'),stopped_epoch=manifest.get('stopped_epoch'),
            training_seconds=manifest.get('training_seconds'),
            **{f'val_{key}':metric['val'][key] for key in ['mae_gwh','rmse_gwh','mape_pct']},
            **metric.get('test',{})))
    return pd.DataFrame(rows)

def grouped(frame,keys):
    rows=[]
    for key,part in frame.groupby(keys,sort=False,dropna=False):
        key=(key,) if not isinstance(key,tuple) else key
        row=dict(zip(keys,key));row['n_runs']=len(part);row['n_dates']=int(part.n.iloc[0])
        row['parameters']=float(part.parameters.iloc[0]) if part.parameters.notna().any() else np.nan
        for column in ['val_mae_gwh','mae_gwh','rmse_gwh','mape_pct']:
            row[column+'_mean']=float(part[column].mean())
            row[column+'_sd']=float(part[column].std(ddof=1)) if len(part)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def build_tables():
    cfg=protocol()
    if not (ROOT/'tables/Table_Window_Alignment_Controls.csv').exists():
        table('Table_Window_Alignment_Controls',pd.read_csv(ROOT/'experiments/window_selection/tables/window_alignment_controls.csv'))
    original=pd.read_csv(ROOT/'data_private/model_space.csv')
    dates=pd.to_datetime(original.date);physical=[]
    for split,bounds in [('all',None),*cfg['main_dates'].items()]:
        values=original.daily_load if bounds is None else original.loc[(dates>=bounds[0])&(dates<=bounds[1]),'daily_load']
        values=values*10
        physical.append(dict(split=split,n_dates=len(values),mean_gwh=float(values.mean()),
            sd_gwh=float(values.std(ddof=1)),minimum_gwh=float(values.min()),maximum_gwh=float(values.max())))
    table('Table_Target_Physical_Statistics',physical)
    seeds=collect('E03');assert len(seeds)==40
    table('E03_All_Run_References',seeds)
    modern=grouped(seeds,['model'])
    modern['implementation']=modern.model.map(lambda name:'Checkpoint-compatible original full model' if name=='Ours' else 'ASTGCN authors recent-component feature-graph adaptation' if name=='ASTGCN' else 'Pinned TSLib reference backbone with daily-total readout')
    modern['status']='PASS'
    table('Table_E01_Baseline_Comparison',modern)
    tuning=[]
    for name in cfg['models']:
        selected=read(ROOT/f'manifests/selected_{name}.json')
        for trial in selected['trials']:
            tuning.append(dict(model=name,trial=trial['trial'],validation_mae_gwh=trial['validation_mae_gwh'],
                selected=trial['trial']==selected['trial'],options=json.dumps(trial['options'],sort_keys=True),run_dir=trial['run_dir']))
    table('E01_All_Validation_Trials',tuning)
    implementation=[]
    for name in cfg['models']:
        selected=read(ROOT/f'manifests/selected_{name}.json');opt=selected['options']
        implementation.append(dict(model=name,history_days=42,hidden_dim=opt['hidden_dim'],
            dropout=opt['dropout'],lr=opt['lr'],depth=opt.get('depth'),
            patch_len_days=opt.get('patch_len') if name=='PatchTST' else np.nan,
            parameters=int(modern.loc[modern.model==name,'parameters'].iloc[0])))
    table('Table_Selected_Implementation_Settings',implementation)
    classic=collect('E02');assert len(classic)==8
    table('Table_E02_Traditional_Comparison',classic)
    legacy=collect('LEGACY','seed');assert len(legacy)==30
    table('Legacy_All_Seed_Results',legacy)
    local=grouped(legacy,['model'])
    addendum=read(ROOT/'configs/legacy_comparison_addendum.json')
    local['implementation']=local.model.map(lambda name:addendum['architectures'][name[len('Legacy_'):]]['description'])
    local['status']='LOCAL_LEGACY_NAMED_METHOD_FIDELITY_UNVERIFIED'
    table('Table_Legacy_Seed_Comparison',local)
    local_settings=[]
    for name in local.model:
        selected=read(ROOT/f'manifests/selected_{name}.json');architecture=addendum['architectures'][name[len('Legacy_'):]]
        local_settings.append(dict(model=name,hidden_dim=architecture['hidden_dim'],dropout=architecture['dropout'],
            lr=selected['options']['lr'],description=architecture['description']))
    table('Table_Legacy_Implementation_Settings',local_settings)
    table('Table_All_Implemented_Seed_Comparison',pd.concat([modern,local],ignore_index=True))
    table('Table_Legacy_Related_Comparison',pd.concat([modern[modern.model=='Ours'],local],ignore_index=True))
    related=pd.concat([modern[modern.model.isin(['Ours','ASTGCN'])],local],ignore_index=True)
    table('Table_Related_Method_Status',related)
    graph=collect('E04');assert len(graph)==60
    table('E04_All_Seed_Results',graph);table('Table_E04_Graph_Sensitivity',grouped(graph,['variant']))
    density=[]
    for _,row in graph.iterrows():
        saved=np.load(Path(row.run_dir)/'graphs.npz');adj=saved['adjacency']
        density.append(dict(variant=row.variant,seed=int(row.seed),n_nodes=adj.shape[-1],
            positive_density=float((adj>0).mean()),mean_positive_out_degree=float((adj>0).sum(axis=2).mean()),
            self_edge_frequency=float((np.diagonal(adj,axis1=1,axis2=2)>0).mean())))
    table('E04_Observed_Graph_Density',density)
    density_frame=pd.DataFrame(density)
    table('Table_E04_Graph_Density',density_frame.groupby('variant',sort=False).agg(
        n_runs=('seed','size'),n_nodes=('n_nodes','first'),
        positive_density_mean=('positive_density','mean'),positive_density_sd=('positive_density','std'),
        positive_out_degree_mean=('mean_positive_out_degree','mean'),
        self_edge_frequency_mean=('self_edge_frequency','mean')).reset_index())
    fusion=collect('E05');assert len(fusion)==50
    table('E05_All_Seed_Results',fusion);table('Table_E05_Fusion_Ablation',grouped(fusion,['variant']))
    ablation=collect('E06');assert len(ablation)==85
    table('E06_All_Seed_Results',ablation);table('Table_E06_All_Ablations',grouped(ablation,['variant']))
    for name,variants in [
        ('Anchor',['full','no_anchor','no_persistence','no_industry_anchor','no_reconciliation','direct','delta']),
        ('Total_Feature',['full','no_sales','no_all_industry','no_both_totals','no_totals_derived','no_aggregate_nodes']),
        ('Auxiliary_Loss',['full','no_load_aux','no_industry_aux','no_both_aux'])]:
        table('Table_E06_'+name+'_Ablation',grouped(ablation[ablation.variant.isin(variants)],['variant']))
    # Original module table uses the three matched graph-sensitivity seeds.
    original_rows=[]
    for source,variant,label in [
        ('E06','full','Full model'),('E06','no_intraday','Without intraday branch'),
        ('E06','no_graph','Without industry graph branch'),('E04','relation_static','Static correlation graph'),
        ('E04','relation_dynamic','Without learnable static association prior'),
        ('E04','relation_no_topk','Without Top-K sparsification'),
        ('E05','F3','Concatenation instead of cross-guided fusion'),
        ('E06','no_anchor','Without statistical anchor'),('E06','no_reconciliation','Without reconciliation residual'),
        ('E06','no_both_aux','Without auxiliary objectives')]:
        frame=collect(source,variant);frame=frame[frame.seed.isin(cfg['graph_sensitivity_seeds'])]
        assert len(frame)==3
        summary=grouped(frame,['variant']).iloc[0].to_dict();summary['model_variant']=label
        original_rows.append(summary)
    table('Table_Module_Ablation_Matched_Three_Seeds',original_rows)
    rolling=collect('E08')
    rolling=rolling[rolling.variant.str.endswith('_seed')|rolling.model.isin(['XGBoost','LightGBM','Last-week Seasonal Naive'])].copy()
    rolling['season']=rolling.variant.str.replace('_seed','',regex=False)
    assert len(rolling)==72,len(rolling)
    additional=collect('LEGACY_ROLL')
    additional=additional[additional.variant.str.endswith('_seed')].copy()
    assert len(additional)==20
    additional['season']=additional.variant.str.replace('_seed','',regex=False)
    table('Legacy_All_Rolling_Seed_Results',additional)
    rolling=pd.concat([rolling,additional],ignore_index=True)
    table('E08_All_Rolling_Seed_Results',rolling)
    table('Table_E08_Rolling_Results',grouped(rolling,['season','model']))
    seed_statistics()
    from .legacy_analysis import legacy_statistics
    legacy_statistics()
    save(ROOT/'reports/table_generation.json',dict(status='PASS',created=utc(),source_sha256=sha(__file__),
        main_neural_runs=40,legacy_neural_runs=30,graph_sensitivity_records=60,fusion_records=50,ablation_records=85,rolling_records=92))
    return modern

def verify_all():
    cfg=protocol();records=[];truth_by_dates={};registered=[]
    receipt=ROOT/'manifests/frozen_protocol_receipt.json'
    if not receipt.exists():receipt=ROOT/'configs/frozen_protocol_receipt.json'
    assert sha(ROOT/'configs/unified_protocol.json')==read(receipt)['protocol_sha256']
    assert sha(ROOT/'src/hubei_revision/suite.py')==cfg['experiment_schedule_sha256']
    assert sha(ROOT/'data_private/model_space.csv')==cfg['data_sha256']
    for group in [f'E{index:02d}' for index in range(1,12)]:
        assert read(ROOT/f'manifests/{group}_status.json')['numerical_experiments']=='PASS',group
    assert read(ROOT/'manifests/LEGACY_status.json')['numerical_experiments']=='PASS'
    addendum=read(ROOT/'configs/legacy_comparison_addendum.json')
    for relative,digest in addendum['source_sha256'].items():assert sha(ROOT/relative)==digest
    for file in sorted((ROOT/'results/checkpoints').glob('*/run_manifest.json')):
        manifest=read(file);run_dir=file.parent
        if manifest.get('smoke'):continue
        assert manifest['status']=='PASS',(run_dir,manifest['status'])
        if manifest['model'].startswith('Legacy_'):
            assert read(run_dir/'run_config.json')['options']['legacy_addendum_sha256']==sha(ROOT/'configs/legacy_comparison_addendum.json')
        if (run_dir/'best_model.pth').exists():
            assert sha(run_dir/'best_model.pth')==manifest['checkpoint_sha256']
            assert manifest['reload_max_error_gwh']<1e-4
        elif (run_dir/'best_model.joblib').exists():assert sha(run_dir/'best_model.joblib')==manifest['checkpoint_sha256']
        else:assert (run_dir/'statistical_computation_record.json').exists()
        metric=read(run_dir/'metrics.json')
        for split in metric:
            frame=pd.read_csv(run_dir/f'{split}_predictions.csv')
            assert frame.date.nunique()==len(frame)
            assert np.isfinite(frame[['true_gwh','pred_gwh']]).all().all()
            if 'true_model' in frame:
                assert np.abs(frame.true_gwh-frame.true_model*10).max()<1e-7
                assert np.abs(frame.pred_gwh-frame.pred_model*10).max()<1e-7
            recomputed=metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')
            error=max(abs(recomputed[key]-metric[split][key]) for key in ['mae_gwh','rmse_gwh','mape_pct'])
            assert error<1e-6,(run_dir,split,error)
            key=tuple(frame.date)
            if key in truth_by_dates:assert np.abs(frame.true_gwh.to_numpy()-truth_by_dates[key]).max()<.001
            else:truth_by_dates[key]=frame.true_gwh.to_numpy()
            records.append(dict(run_id=run_dir.name,split=split,n=len(frame),metric_max_difference=error,
                prediction_sha256=sha(run_dir/f'{split}_predictions.csv'),status='PASS'))
        registered.append(dict(run_id=run_dir.name,group=manifest['group'],model=manifest['model'],seed=manifest['seed'],
            status=manifest['status'],checkpoint_sha256=manifest.get('checkpoint_sha256'),
            training_seconds=manifest.get('training_seconds'),best_epoch=manifest.get('best_epoch'),stopped_epoch=manifest.get('stopped_epoch')))
    for rel,digest in cfg['numerical_source_sha256'].items():assert sha(ROOT/'src/hubei_revision'/rel)==digest
    window_records=[]
    windows=list((ROOT/'experiments/window_selection/checkpoints').glob('W_*/run_manifest.json'))
    assert len(windows)==35,len(windows)
    for file in windows:
        manifest=read(file);run_dir=file.parent
        assert manifest['status']=='PASS'
        assert sha(run_dir/'best_model.pt')==manifest['checkpoint_sha256']
        scores=read(run_dir/'metrics.json')
        for split in ['train','val','test']:
            frame=pd.read_csv(run_dir/f'{split}_predictions.csv')
            assert frame.date.nunique()==len(frame)
            score=metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')
            difference=max(abs(score[key]-scores[split][key]) for key in ['mae_gwh','rmse_gwh','mape_pct'])
            assert difference<1e-6,(run_dir,split,difference)
            window_records.append(dict(run_id=run_dir.name,split=split,n=len(frame),
                metric_max_difference=difference,checkpoint_sha256=manifest['checkpoint_sha256'],status='PASS'))
    table('历史窗口全部权重与预测核验',window_records)
    table('全部运行注册表',registered);table('全部预测与权重核验',records)
    save(ROOT/'reports/实验结果自动验收.json',dict(status='PASS',checked_unique_formal_runs=len(registered),
        checked_window_runs=len(windows),checked_window_prediction_files=len(window_records),
        checked_prediction_files=len(records),all_weights_verified=True,code_matches_frozen_protocol=True,
        blocked_methods=[],legacy_local_implementations_completed=True,
        unverified_literature_method_fidelity=list(addendum['architectures']),
        cross_region='NOT_RUN_NO_COMPATIBLE_DATA',created=utc()))
    save(ROOT/'manifests/current_stage.json',dict(status='NUMERICAL_COMPLETE',created=utc(),
        note='Saved predictions and checkpoints verified; document and workbook QA are separate delivery gates.'))

if __name__=='__main__':
    build_tables();verify_all();print('ALL_RESULT_TABLES_AND_CHECKPOINTS_VERIFIED')
