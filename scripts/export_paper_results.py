"""Export only the aggregate results corresponding to Tables 5--16.

No observations, dated predictions, run paths or checkpoints are exported.
Requires private experiments to have been completed and summarized locally.
"""
from pathlib import Path
import argparse
import json
import pandas as pd

NATIVE = ['PatchTST','TSMixer','TimesNet','iTransformer','TimeMixer','TimeXer','Ours']
LOCAL = ['Legacy_MG-ASTL','Legacy_TL-STLF','Legacy_LoadFormer','Legacy_STGNN-Load',
         'Legacy_ASTGCN-Load','Legacy_GAT-LoadMixer','Ours']
METRICS = ['mae_gwh','rmse_gwh','mape_pct']
MATCHED = [42,123,3407]
SEEDS = [42,123,2024,2025,3407]

def export_paper_results(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    exports = []
    def read(name):
        return pd.read_csv(root/'tables'/name,encoding='utf-8-sig')
    def ordered(df,key,names):
        return df.set_index(key).loc[names].reset_index()
    def save(number,name,df,scope,integer=(),two_decimal=()):
        df = df.copy()
        for col in df:
            if col in integer:
                df[col] = df[col].astype(int)
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2 if col in two_decimal else 3)
        path=destination/f'table_{number:02d}_{name}.csv'
        df.to_csv(path,index=False,float_format='%.3f',encoding='utf-8')
        exports.append(dict(table=number,file=path.name,rows=len(df),scope=scope))
    stats=[x+s for x in METRICS for s in ['_mean','_sd']]
    w=read('Table_Window_Alignment_Controls.csv')
    if not {'val_rmse_gwh_mean','val_mape_pct_mean'}.issubset(w.columns):
        # Fresh history-driver summaries retain the additional validation
        # metrics in their per-run aggregate log, not the short control table.
        log=pd.read_csv(root/'experiments/window_selection/checkpoints/all_history_controls.csv')
        extra=log.groupby(['intraday_days','industry_days'],as_index=False).agg(
            val_rmse_gwh_mean=('val_rmse_gwh','mean'),
            val_mape_pct_mean=('val_mape_pct','mean'))
        w=w.drop(columns=[c for c in extra.columns[2:] if c in w.columns]).merge(
            extra,on=['intraday_days','industry_days'],validate='one_to_one')
    w=ordered(w,'window',['28/28','30/30','34/34','38/38','42/42','28/42','42/28'])
    save(5,'history_window_validation',w[['intraday_days','industry_days',
        'val_mae_gwh_mean','val_rmse_gwh_mean','val_mape_pct_mean']],
        'Validation only; five seeds; fixed learning rate 0.0005.',integer=['intraday_days','industry_days'])
    main=read('Table_All_Implemented_Seed_Comparison.csv')
    save(6,'general_forecasting',ordered(main,'model',NATIVE)[['model']+stats],
        'Test; all five training seeds; sample SD.')
    c=ordered(read('Table_Selected_Implementation_Settings.csv'),'model',NATIVE)
    c=c[['model','lr','dropout','parameters']].rename(columns={'lr':'learning_rate'})
    # Learning rates need greater precision than forecast metrics.
    p=destination/'table_07_selected_configurations.csv'
    c['parameters']=c['parameters'].astype(int)
    c.to_csv(p,index=False,encoding='utf-8')
    exports.append(dict(table=7,file=p.name,rows=len(c),scope='Selected configurations; hidden width 96 for these models.'))
    save(8,'local_deep_learning',ordered(main,'model',LOCAL)[['model']+stats],
        'Test; five seeds; local labels do not authenticate named-paper reproduction.')
    t=ordered(read('Table_E02_Traditional_Comparison.csv'),'model',
        ['Last-day Naive','Last-week Seasonal Naive','ARIMA','Exponential Smoothing',
         'Linear Regression','Ridge Regression','XGBoost','LightGBM'])[['model']+METRICS]
    ours=main.loc[main.model.eq('Ours'),['model']+[x+'_mean' for x in METRICS]].copy()
    ours.columns=['model']+METRICS
    save(9,'statistical_machine_learning',pd.concat([t,ours],ignore_index=True),
        'Test; Ours is the five-seed mean; other rows are individual baseline evaluations.')
    a=read('Table_Module_Ablation_Matched_Three_Seeds.csv')
    save(10,'module_ablation',a[['variant']+stats],
        'Test; matched seeds 42, 123, 3407; sample SD.')
    f=ordered(read('Table_E05_Fusion_Ablation.csv'),'variant',[f'F{i}' for i in range(9)])
    save(11,'fusion_strategies',f[['variant','parameters']+stats],
        'Test; all five seeds; complete registered parameter counts.',integer=['parameters'])
    g=read('Table_E04_Graph_Sensitivity.csv')
    k=ordered(g,'variant',[f'K{i}' for i in [2,4,6,8,12,16]])
    k.insert(1,'K',[2,4,6,8,12,16])
    k.insert(2,'density_pct',100*k.K/23)
    save(12,'graph_topk',k[['K','density_pct','val_mae_gwh_mean','val_mae_gwh_sd',
        'mae_gwh_mean','mae_gwh_sd','rmse_gwh_mean','rmse_gwh_sd']],
        'Validation and test are labeled separately; three matched seeds; density includes diagonals.',
        integer=['K'],two_decimal=['density_pct'])
    names=['relation_full','relation_static','relation_dynamic','relation_no_topk',
           'relation_attention','relation_mlp']
    q=ordered(g,'variant',names)
    save(13,'graph_relation_controls',q[['variant','parameters','mae_gwh_mean','mae_gwh_sd',
        'rmse_gwh_mean','rmse_gwh_sd']],
        'Test; three matched seeds; actual parameter counts, not asserted exact matches.',integer=['parameters'])
    scenarios=read('E07_All_Scenario_Seed_Results.csv')
    scenarios=scenarios[scenarios.seed.isin(MATCHED)]
    conds=['ordinary_workday','weekend','low_temperature','spring_festival','spring_transition','large_day_change']
    rows=[]
    for subset in ['all_test_dates']+conds:
        row=dict(condition=subset)
        for model,prefix in [('Ours','complete'),('Static correlation graph','static')]:
            if subset=='all_test_dates':
                name='relation_full' if model=='Ours' else 'relation_static'
                z=g[g.variant.eq(name)].iloc[0]
                row['n_dates']=63
                row[prefix+'_mae_gwh_mean']=z.mae_gwh_mean
                row[prefix+'_mae_gwh_sd']=z.mae_gwh_sd
            else:
                z=scenarios[scenarios.model.eq(model)&scenarios.subset.eq(subset)]
                assert set(z.seed.astype(int))==set(MATCHED),(model,subset)
                row['n_dates']=int(z.n.iloc[0])
                row[prefix+'_mae_gwh_mean']=z.mae_gwh.mean()
                row[prefix+'_mae_gwh_sd']=z.mae_gwh.std(ddof=1)
        rows.append(row)
    save(14,'graph_operating_conditions',pd.DataFrame(rows),
        'Test; matched three seeds; date subsets overlap where applicable; small subsets are descriptive.',integer=['n_dates'])
    seasonal=read('Table_E08_Rolling_Results.csv')
    rows=[]
    for model in ['Ours','iTransformer','TSMixer']:
        row=dict(model=model)
        for season in ['summer','autumn','winter','spring']:
            z=seasonal[seasonal.model.eq(model)&seasonal.season.eq(season)].iloc[0]
            row[season+'_mae_gwh_mean']=z.mae_gwh_mean
            row[season+'_mae_gwh_sd']=z.mae_gwh_sd
        rows.append(row)
    save(15,'seasonal_forecasting',pd.DataFrame(rows),
        'Four 30-day next-day forecast windows; five seeds per model and fold; sample SD.')
    noise=read('Table_E09_Robustness.csv')
    rows=[]
    z=noise[noise.condition.eq('unperturbed')].iloc[0]
    rows.append(dict(input_group='unperturbed',noise_coefficient=0.,**{m:z[m] for m in METRICS}))
    for cat in ['intraday','industry','weather']:
        for alpha in [.01,.03,.05,.1]:
            z=noise[noise.category.eq(cat)&noise.strength.eq(alpha)]
            assert set(z.perturbation_seed)=={101,202,303}
            rows.append(dict(input_group=cat,noise_coefficient=alpha,**{m:z[m].mean() for m in METRICS}))
    save(16,'input_noise_robustness',pd.DataFrame(rows),
        'Fixed model seed 42; metrics averaged over three perturbation seeds, not training seeds.')
    return exports

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    exports=export_paper_results(a.root,a.output or a.root/'paper_results')
    print(json.dumps(exports,indent=2))
