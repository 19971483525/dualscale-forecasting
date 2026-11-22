"""One data protocol, original numerical training loop, explicit GWh evaluation."""
from dataclasses import asdict
from pathlib import Path
from functools import lru_cache
import json
import platform
import random
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .common import ROOT,read,save,sha,utc
from .units import energy_model_to_gwh,metrics_gwh
from .legacy.data import (DailyForecastConfig,DailyForecastDataset,build_sample_arrays,
    build_standardizers,apply_standardizers,build_target_transform,
    GLOBAL_DAILY_FEATURES,INTRADAY_DAILY_FEATURES,TARGET_CALENDAR_FEATURES)
from .legacy.trainer import evaluate_model,asymmetric_huber
from .models import ControlledModel,OfficialAdapter

torch.set_num_threads(2)
INPUT_KEYS=['intraday_curve_hist','intraday_daily_hist','industry_hist','global_hist','flat_hist','target_calendar']
DEFAULT_OPTIONS=dict(hidden_dim=96,dropout=.04,lr=.0005,depth=1,patch_len=12,
    topk=8,prior_weight=1.,relation='full',fusion='F0',anchor='full',no_intraday=False,
    no_graph=False,no_anchor=False,no_flat=False,no_reconciliation=False,
    no_static_prior=False,no_topk=False,target_mode='delta',remove=[])

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def inputs(batch):return {key:batch[key] for key in INPUT_KEYS}

def protocol():return read(ROOT/'configs/unified_protocol.json')

def expanded_removals(opt):
    removed=set(opt.get('remove',[]))
    if 'sales_total' in removed:removed.update(['sales_gap','sales_ratio'])
    return sorted(removed)

def load_data(opt=None,fold=None,override=None,fitted=None):
    opt=opt or {};cfg=protocol();history=cfg['history_days']
    audit=read(ROOT/'data_private/unit_audit.json')
    assert audit['status']=='PASS','Unit audit must pass before training'
    df=pd.read_csv(ROOT/'data_private/model_space.csv') if override is None else override.copy()
    industry=[name for name in audit['industry_features'] if name not in expanded_removals(opt)]
    for name in expanded_removals(opt):
        if name in df:df[name]=0.
    arrays=build_sample_arrays(df,DailyForecastConfig(history,history,1),industry)
    dates=pd.to_datetime(arrays['dates'])
    bounds=cfg['main_dates'] if fold is None else fold
    indices={key:np.where((dates>=interval[0])&(dates<=interval[1]))[0] for key,interval in bounds.items()}
    assert all(len(ix)>0 for ix in indices.values())
    assert indices['train'].max()<indices['val'].min() and indices['val'].max()<indices['test'].min()
    if fold is None:assert [len(indices[k]) for k in ['train','val','test']]==[289,62,63]
    if fitted is None:
        stats=build_standardizers(arrays,indices['train'])
        trans=build_target_transform(arrays,indices['train'])
        mode=opt.get('target_mode','delta')
        if mode=='direct':
            target=arrays['target'][indices['train'],0]
            trans.delta_mean=float(target.mean());trans.delta_std=float(max(target.std(),1e-6))
        fitted=dict(stats=stats,trans=trans,mode=mode)
    stats=fitted['stats'];trans=fitted['trans'];mode=fitted['mode']
    scaled=apply_standardizers(arrays,stats)
    base=np.zeros_like(arrays['base_load']) if mode=='direct' else arrays['base_load']
    scaled['target']=trans.transform(arrays['target'],base)
    ds={key:DailyForecastDataset(scaled,ix) for key,ix in indices.items()}
    def correlation(x):
        with np.errstate(divide='ignore',invalid='ignore'):return np.nan_to_num(np.corrcoef(x,rowvar=False),nan=0.,posinf=0.,neginf=0.)
    corr=correlation(arrays['industry_hist'][indices['train']].reshape(-1,len(industry)))
    channels=arrays['flat_hist'].shape[-1]
    flatcorr=np.abs(correlation(arrays['flat_hist'][indices['train']].reshape(-1,channels)))
    np.fill_diagonal(flatcorr,0.)
    return dict(df=df,raw=arrays,scaled=scaled,indices=indices,idx=indices,ds=ds,
        industry=industry,corr=corr,flatcorr=flatcorr,fitted=fitted,bounds=bounds,
        shapes={key:list(value.shape[1:]) for key,value in arrays.items() if key!='dates'})

def inverse(pred,base,data):
    if data['fitted']['mode']=='direct':base=np.zeros_like(base)
    return data['fitted']['trans'].inverse(pred,base)

def make_model(name,data,opt,device):
    if name=='Ours':return ControlledModel(data['shapes'],opt,data['corr'],data['industry'])
    return OfficialAdapter(name,data['shapes']['flat_hist'][-1],data['shapes']['flat_hist'][0],opt,device,data['flatcorr'])

def predict(model,data,split,device,batch_size=16,graphs=False):
    model.eval();zs=[];bases=[];adjs=[];att=[]
    with torch.no_grad():
        for batch in DataLoader(data['ds'][split],batch_size=batch_size,shuffle=False):
            batch={key:value.to(device) for key,value in batch.items()}
            out=model(inputs(batch))
            zs.append(out['prediction'].cpu().numpy());bases.append(batch['base_load'].cpu().numpy())
            if graphs and 'dynamic_adj' in out:
                adjs.append(out['dynamic_adj'].cpu().numpy());att.append(out['node_importance'].cpu().numpy())
    prediction=inverse(np.concatenate(zs),np.concatenate(bases),data)[:,0].astype(np.float64)
    truth=data['raw']['target'][data['indices'][split],0].astype(np.float64)
    frame=pd.DataFrame(dict(date=pd.to_datetime(data['ds'][split].dates),true_model=truth,pred_model=prediction,
        true_gwh=energy_model_to_gwh(truth),pred_gwh=energy_model_to_gwh(prediction)))
    score=metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')
    return frame,score,(np.concatenate(adjs) if adjs else None),(np.concatenate(att) if att else None)

def normalized_options(name,opt,cfg):
    result={**DEFAULT_OPTIONS,'aux_weight':cfg['training']['aux_weight'],
        'industry_aux_weight':cfg['training']['industry_aux_weight'],**opt}
    result.setdefault('embedding_dim',result['hidden_dim'])
    result['remove']=expanded_removals(result)
    if name=='Ours':result.pop('depth');result.pop('patch_len')
    else:
        for key in ['topk','prior_weight','relation','fusion','anchor','no_intraday','no_graph','no_anchor',
            'no_flat','no_reconciliation','no_static_prior','no_topk','embedding_dim']:
            result.pop(key,None)
    return result

def reference(group,variant,name,seed,run_dir):
    save(ROOT/f'experiments/{group}/{variant}_{name}_s{seed}.json',dict(group=group,variant=variant,
        model=name,seed=seed,run_dir=str(run_dir)))

def run_training(group,variant,name,opt,seed=42,fold=None,test_after=True,smoke=False):
    cfg=protocol();device=cfg['device'];opt=normalized_options(name,opt,cfg)
    training={**cfg['training'],'lr':opt['lr'],'aux_weight':opt['aux_weight'],
        'industry_aux_weight':opt['industry_aux_weight']}
    if smoke:training.update(epochs=2,min_epochs=1,patience=2)
    key=dict(model=name,options=opt,training=training,seed=seed,fold=fold,
        history_days=cfg['history_days'],main_dates=cfg['main_dates'],data_sha256=cfg['data_sha256'],
        numerical_source_sha256=cfg['numerical_source_sha256'])
    signature=__import__('hashlib').sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    for file in (ROOT/'results/checkpoints').glob('*/run_manifest.json'):
        manifest=read(file)
        if manifest.get('signature')==signature and manifest['status']=='PASS':
            reference(group,variant,name,seed,file.parent)
            if test_after:ensure_test(file.parent)
            return file.parent
    run_id=f'{group}_{variant}_{name}_s{seed}'+('_smoke' if smoke else '')
    run_dir=ROOT/'results/checkpoints'/run_id
    if run_dir.exists():raise RuntimeError(f'Run exists and is not reusable; preserve and audit before retry: {run_dir}')
    run_dir.mkdir(parents=True)
    manifest=dict(status='RUNNING',run_id=run_id,group=group,variant=variant,model=name,seed=seed,
        signature=signature,started=utc(),data_sha256=cfg['data_sha256'],smoke=smoke)
    save(run_dir/'run_manifest.json',manifest)
    reference(group,variant,name,seed,run_dir)
    run_config=dict(model=name,options=opt,training=training,seed=seed,fold=fold,device=device,
        history_days=cfg['history_days'],protocol_sha256=sha(ROOT/'configs/unified_protocol.json'))
    save(run_dir/'run_config.json',run_config)
    try:
        data=load_data(opt,fold)
        seed_all(seed)
        model=make_model(name,data,opt,device).to(device)
        manifest.update(parameters=sum(p.numel() for p in model.parameters()),input_shapes=data['shapes'],
            environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),device=device,precision='float32'))
        save(run_dir/'run_manifest.json',manifest)
        torch.save(data['fitted'],run_dir/'normalization.pth')
        save(run_dir/'normalization.json',dict(fit_dates=data['bounds']['train'],
            target=dict(mode=data['fitted']['mode'],mean=data['fitted']['trans'].delta_mean,
                std=data['fitted']['trans'].delta_std,aux_mean=data['fitted']['trans'].aux_mean.tolist(),
                aux_std=data['fitted']['trans'].aux_std.tolist()),
            features={key:dict(mean=value.mean.tolist(),std=value.std.tolist()) for key,value in data['fitted']['stats'].items()}))
        # Preserve the original trainer's RNG reset after model construction,
        # shuffled-loader train evaluation, losses, optimizer, and schedule.
        seed_all(seed)
        loaders={key:DataLoader(ds,batch_size=training['batch_size'],shuffle=key=='train') for key,ds in data['ds'].items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=training['lr'],weight_decay=training['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=training['epochs'])
        best=float('inf');bad=0;history=[];start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        inverse_fn=lambda pred,base:inverse(pred,base,data)
        for epoch in range(1,training['epochs']+1):
            epoch_start=time.perf_counter();model.train();losses=[]
            for batch in loaders['train']:
                batch={key:value.to(device) for key,value in batch.items()}
                out=model(inputs(batch));z=out['prediction'];target=batch['target']
                loss=asymmetric_huber(z[:,:1],target[:,:1],1.)
                loss=loss+training['aux_weight']*nn.functional.smooth_l1_loss(z[:,1:],target[:,1:])
                if 'industry_prediction' in out:
                    loss=loss+training['industry_aux_weight']*nn.functional.smooth_l1_loss(out['industry_prediction'],batch['industry_target'])
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.item()))
            scheduler.step()
            train_eval=evaluate_model(model,loaders['train'],inverse_fn,device)
            val_eval=evaluate_model(model,loaders['val'],inverse_fn,device)
            val_mae=val_eval[1]
            item=dict(epoch=epoch,loss=float(np.mean(losses)),train_mae_gwh=train_eval[1]*10,
                val_mae_gwh=val_mae*10,epoch_seconds=time.perf_counter()-epoch_start)
            history.append(item)
            pd.DataFrame(history).to_csv(run_dir/'training_history.csv',index=False,encoding='utf-8-sig')
            with (run_dir/'train.log').open('a',encoding='utf8') as handle:handle.write(json.dumps(item)+'\n')
            if val_mae<best:
                best=val_mae;bad=0;manifest['best_epoch']=epoch
                state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                torch.save(dict(state_dict=state,run_config=run_config),run_dir/'best_model.pth')
            else:bad+=1
            if epoch>=training['min_epochs'] and bad>=training['patience']:break
        elapsed=time.perf_counter()-start
        model.load_state_dict(torch.load(run_dir/'best_model.pth',map_location=device,weights_only=True)['state_dict'],strict=True)
        metrics={}
        for split in ['train','val']+(['test'] if test_after or smoke else []):
            frame,score,adj,att=predict(model,data,split,device,graphs=split=='test')
            frame.to_csv(run_dir/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
            frame=pd.read_csv(run_dir/f'{split}_predictions.csv')
            metrics[split]=metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')
            if adj is not None:np.savez_compressed(run_dir/'graphs.npz',adjacency=adj,readout=att,
                dates=frame.date.to_numpy(dtype=str),nodes=np.array(data['industry']))
        fresh=make_model(name,data,opt,device).to(device)
        fresh.load_state_dict(torch.load(run_dir/'best_model.pth',map_location=device,weights_only=True)['state_dict'],strict=True)
        before=predict(model,data,'val',device)[0].pred_gwh.to_numpy()
        after=predict(fresh,data,'val',device)[0].pred_gwh.to_numpy()
        reload_error=float(np.abs(before-after).max());assert reload_error<1e-4
        manifest.update(status='PASS',finished=utc(),stopped_epoch=epoch,training_seconds=elapsed,
            epoch_seconds_mean=float(np.mean([item['epoch_seconds'] for item in history])),
            training_peak_allocated_gpu_mb=torch.cuda.max_memory_allocated()/1024**2,
            reload_max_error_gwh=reload_error,checkpoint_sha256=sha(run_dir/'best_model.pth'))
        save(run_dir/'metrics.json',metrics);save(run_dir/'run_manifest.json',manifest)
        print('PASS',run_id,'val',round(metrics['val']['mae_gwh'],4),'test',round(metrics.get('test',{}).get('mae_gwh',float('nan')),4),flush=True)
        del model,fresh;torch.cuda.empty_cache()
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),finished=utc())
        save(run_dir/'run_manifest.json',manifest)
        with (run_dir/'train.log').open('a',encoding='utf8') as handle:handle.write(manifest['traceback'])
        torch.cuda.empty_cache();raise
    return run_dir

def load_run(run_dir):
    run_dir=Path(run_dir);cfg=read(run_dir/'run_config.json')
    fitted=torch.load(run_dir/'normalization.pth',map_location='cpu',weights_only=False)
    data=load_data(cfg['options'],cfg['fold'],fitted=fitted)
    model=make_model(cfg['model'],data,cfg['options'],cfg['device']).to(cfg['device'])
    model.load_state_dict(torch.load(run_dir/'best_model.pth',map_location=cfg['device'],weights_only=True)['state_dict'],strict=True)
    return model,data,cfg

def ensure_test(run_dir):
    run_dir=Path(run_dir)
    if (run_dir/'test_predictions.csv').exists():return
    assert read(run_dir/'run_manifest.json')['status']=='PASS'
    model,data,cfg=load_run(run_dir)
    frame,score,adj,att=predict(model,data,'test',cfg['device'],graphs=True)
    frame.to_csv(run_dir/'test_predictions.csv',index=False,encoding='utf-8-sig')
    metrics=read(run_dir/'metrics.json');metrics['test']=score;save(run_dir/'metrics.json',metrics)
    if adj is not None:np.savez_compressed(run_dir/'graphs.npz',adjacency=adj,readout=att,
        dates=frame.date.to_numpy(dtype=str),nodes=np.array(data['industry']))
    del model;torch.cuda.empty_cache()
