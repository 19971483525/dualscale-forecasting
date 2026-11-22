"""Reproduce the separate historical-window phase at the original learning rate."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hubei_revision.common import ROOT,read,save,sha
from hubei_revision import window_trainer as trainer
from hubei_revision.legacy.data import (DailyForecastConfig,build_sample_arrays,
    build_standardizers,apply_standardizers,build_target_transform,DailyForecastDataset,
    INTRADAY_DAILY_FEATURES,GLOBAL_DAILY_FEATURES,TARGET_CALENDAR_FEATURES)
from hubei_revision.legacy.model import DualScaleLoadForecastModel,ModelConfig
from hubei_revision.units import energy_model_to_gwh,metrics_gwh

p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true')
p.add_argument('--output',type=Path,default=ROOT/'experiments/window_selection/checkpoints')
a=p.parse_args()
windows=[(28,28),(30,30),(34,34),(38,38),(42,42),(28,42),(42,28)]
seeds=[42,123,2024,2025,3407];protocol=read(ROOT/'configs/unified_protocol.json')
assert read(ROOT/'data_private/unit_audit.json')['status']=='PASS'
assert sha(ROOT/'data_private/model_space.csv')==protocol['data_sha256']
industry=read(ROOT/'data_private/unit_audit.json')['industry_features']
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def data_for(left,right):
    df=pd.read_csv(ROOT/'data_private/model_space.csv')
    raw=build_sample_arrays(df,DailyForecastConfig(left,right,1),industry)
    dates=pd.to_datetime(raw['dates'])
    indices={key:np.where((dates>=value[0])&(dates<=value[1]))[0] for key,value in protocol['main_dates'].items()}
    assert [len(indices[key]) for key in ['train','val','test']]==[289,62,63]
    stats=build_standardizers(raw,indices['train']);target=build_target_transform(raw,indices['train'])
    scaled=apply_standardizers(raw,stats);scaled['target']=target.transform(raw['target'],raw['base_load'])
    datasets={key:DailyForecastDataset(scaled,index) for key,index in indices.items()}
    return raw,stats,target,datasets

def predict(model,datasets,target,split):
    _,_,actual,predicted,dates=trainer.evaluate_model(model,trainer.build_loader(datasets[split],16,False),target.inverse,'cuda')
    actual=actual[:,0].astype(np.float64);predicted=predicted[:,0].astype(np.float64)
    return pd.DataFrame(dict(date=pd.to_datetime(dates),true_model=actual,pred_model=predicted,
        true_gwh=energy_model_to_gwh(actual),pred_gwh=energy_model_to_gwh(predicted)))

rows=[]
for left,right in windows:
    raw,stats,target,datasets=data_for(left,right)
    if a.dry_run:
        print('VERIFIED',left,right,{key:len(value) for key,value in datasets.items()});continue
    assert torch.cuda.is_available(),'Original-window training requires CUDA.'
    for seed in seeds:
        output=a.output/f'W_{left}_{right}_s{seed}'
        manifest_path=output/'run_manifest.json'
        if output.exists():
            manifest=read(manifest_path)
            assert manifest['status']=='PASS' and manifest.get('history_driver_sha256')==sha(__file__),'Preserve this existing run before resuming.'
            assert sha(output/'best_model.pt')==manifest['checkpoint_sha256']
        else:
            output.mkdir(parents=True)
            trainer.set_seed(seed)
            model_config=ModelConfig(intraday_history=left,industry_history=right,
                intraday_feat_dim=len(INTRADAY_DAILY_FEATURES),industry_nodes=len(industry),
                global_dim=len(GLOBAL_DAILY_FEATURES),target_calendar_dim=len(TARGET_CALENDAR_FEATURES),
                hidden_dim=96,dropout=.04)
            model=DualScaleLoadForecastModel(model_config)
            training=trainer.TrainConfig(epochs=120,batch_size=16,lr=.0005,weight_decay=.0001,
                patience=25,min_epochs=30,aux_weight=.2,industry_aux_weight=.15,seed=seed,device='cuda')
            manifest=dict(status='RUNNING',seed=seed,intraday_days=left,industry_days=right,
                history_driver_sha256=sha(__file__),trainer_sha256=sha(Path(trainer.__file__)),
                data_sha256=protocol['data_sha256'],
                model_config=asdict(model_config),training_config=asdict(training))
            save(manifest_path,manifest)
            torch.save(dict(stats=stats,target=target),output/'normalization.pth')
            start=time.perf_counter()
            native=trainer.train_model(model,datasets,target.inverse,output,training)
            save(output/'metrics_model_space.json',native)
            scores={}
            # Ordered inference avoids the original shuffled training-export date mismatch.
            for split in ['train','val','test']:
                frame=predict(model,datasets,target,split)
                frame.to_csv(output/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
                scores[split]=metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')
            fresh=DualScaleLoadForecastModel(model_config).cuda()
            fresh.load_state_dict(torch.load(output/'best_model.pt',map_location='cuda',weights_only=True),strict=True)
            replay=predict(fresh,datasets,target,'val');original=pd.read_csv(output/'val_predictions.csv')
            error=float(np.abs(replay.pred_gwh-original.pred_gwh).max());assert error<1e-4
            history=pd.read_csv(output/'training_history.csv')
            for split in ['train','val']:
                for metric in ['mae','rmse']:
                    history[split+'_'+metric+'_gwh']=history.pop(split+'_'+metric)*10
                history.rename(columns={split+'_mape':split+'_mape_pct'},inplace=True)
            history.to_csv(output/'training_history.csv',index=False,encoding='utf-8-sig')
            save(output/'metrics.json',scores)
            manifest.update(status='PASS',training_seconds=time.perf_counter()-start,
                checkpoint_sha256=sha(output/'best_model.pt'),reload_max_error_gwh=error)
            save(manifest_path,manifest);del model,fresh;torch.cuda.empty_cache()
        score=read(output/'metrics.json')
        rows.append(dict(intraday_days=left,industry_days=right,seed=seed,
            run_dir=str(output),**{'val_'+key:value for key,value in score['val'].items()},
            **{'test_'+key:value for key,value in score['test'].items()}))
        pd.DataFrame(rows).to_csv(a.output/'all_history_controls.csv',index=False,encoding='utf-8-sig')
if not a.dry_run:
    raw=pd.DataFrame(rows);phase_root=a.output.parent if a.output.name=='checkpoints' else a.output
    tables=phase_root/'tables';tables.mkdir(parents=True,exist_ok=True)
    aligned=raw[raw.intraday_days==raw.industry_days].copy();aligned['history_days']=aligned.intraday_days
    aligned=aligned.rename(columns={key:'_'.join(key.split('_')[1:]) for key in ['test_mae_gwh','test_rmse_gwh','test_mape_pct']})
    aligned.to_csv(tables/'all_aligned_windows.csv',index=False,encoding='utf-8-sig')
    summary=aligned.groupby('history_days').agg(n_seeds=('seed','size'),
        val_mae_gwh_mean=('val_mae_gwh','mean'),val_mae_gwh_sd=('val_mae_gwh','std'),
        test_mae_gwh_mean=('mae_gwh','mean'),test_mae_gwh_sd=('mae_gwh','std'),
        test_rmse_gwh_mean=('rmse_gwh','mean'),test_rmse_gwh_sd=('rmse_gwh','std'),
        test_mape_pct_mean=('mape_pct','mean'),test_mape_pct_sd=('mape_pct','std')).reset_index()
    summary.to_csv(tables/'aligned_window_summary.csv',index=False,encoding='utf-8-sig')
    control=[]
    for (left,right),part in raw.groupby(['intraday_days','industry_days']):
        row=dict(window=f'{left}/{right}',intraday_days=left,industry_days=right,n_seeds=len(part))
        for key in ['val_mae_gwh','test_mae_gwh','test_rmse_gwh','test_mape_pct']:
            row[key+'_mean']=float(part[key].mean());row[key+'_sd']=float(part[key].std(ddof=1))
        control.append(row)
    pd.DataFrame(control).to_csv(tables/'window_alignment_controls.csv',index=False,encoding='utf-8-sig')
print('HISTORY_DATA_DRY_RUN_PASSED' if a.dry_run else 'HISTORY_CONTROL_REPRODUCTION_COMPLETE')
