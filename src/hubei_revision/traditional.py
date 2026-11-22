"""Causal one-step statistical and full-information machine-learning baselines."""
import time
import traceback
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge
from threadpoolctl import threadpool_limits
from .common import ROOT,read,save,sha,utc
from .engine import load_data,protocol,inverse,reference
from .units import energy_model_to_gwh,metrics_gwh

METHODS=['Last-day Naive','Last-week Seasonal Naive','Linear Regression','Ridge Regression',
    'ARIMA','Exponential Smoothing','XGBoost','LightGBM']

def design(data,split):
    ix=data['indices'][split];a=data['scaled']
    return np.concatenate([a['flat_hist'][ix].reshape(len(ix),-1),a['target_calendar'][ix]],axis=1),a['target'][ix,0]

def regression_predictions(data,split,z):
    ix=data['indices'][split];prediction=np.zeros((len(ix),4),dtype=np.float32)
    prediction[:,0]=z
    pred=inverse(prediction,data['raw']['base_load'][ix],data)[:,0].astype(np.float64)
    truth=data['raw']['target'][ix,0].astype(np.float64)
    return pd.DataFrame(dict(date=pd.to_datetime(data['raw']['dates'][ix]),true_model=truth,pred_model=pred,
        true_gwh=energy_model_to_gwh(truth),pred_gwh=energy_model_to_gwh(pred)))

def run_traditional(name,fold=None,label='main'):
    cfg=protocol();history=cfg['history_days'];data=load_data(fold=fold)
    run_dir=ROOT/'results/checkpoints'/('E02_'+label+'_'+name.replace(' ','_'))
    group='E02' if label=='main' else 'E08'
    if (run_dir/'run_manifest.json').exists():
        assert read(run_dir/'run_manifest.json')['status']=='PASS'
        reference(group,label,name,42,run_dir);return run_dir
    run_dir.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='RUNNING',model=name,group=group,variant=label,seed=42,started=utc(),
        data_sha256=cfg['data_sha256'],history_days=history)
    save(run_dir/'run_manifest.json',manifest)
    reference(group,label,name,42,run_dir)
    start=time.perf_counter()
    try:
        if name in ['Last-day Naive','Last-week Seasonal Naive','Linear Regression']:grid=[{}]
        elif name=='Ridge Regression':grid=[dict(alpha=v) for v in [1.,100.,10000.]]
        elif name=='XGBoost':grid=[dict(max_depth=d,n_estimators=n,learning_rate=lr) for d,n,lr in [(2,150,.03),(3,250,.03),(2,400,.01)]]
        elif name=='LightGBM':grid=[dict(num_leaves=d,n_estimators=n,learning_rate=lr) for d,n,lr in [(7,150,.03),(15,250,.03),(7,400,.01)]]
        elif name=='ARIMA':grid=[dict(order=order) for order in [(1,0,0),(1,1,0),(2,0,1)]]
        else:grid=[dict(trend=trend,seasonal=seasonal) for trend,seasonal in [(None,None),('add',None),('add','add')]]
        statistical=name in ['Last-day Naive','Last-week Seasonal Naive','ARIMA','Exponential Smoothing']
        def stat_predict(split,params):
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            dates=pd.to_datetime(data['df'].date);preds=[];records=[]
            for i in data['indices'][split]:
                target_day=pd.Timestamp(data['raw']['dates'][i]);day=int(np.where(dates==target_day)[0][0])
                observed=data['df'].daily_load.to_numpy()[day-history:day]
                assert len(observed)==history
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if name=='Last-day Naive':pred=observed[-1]
                    elif name=='Last-week Seasonal Naive':pred=observed[-7]
                    elif name=='ARIMA':pred=ARIMA(observed,order=params['order']).fit().forecast(1)[0]
                    else:pred=ExponentialSmoothing(observed,trend=params['trend'],seasonal=params['seasonal'],
                        seasonal_periods=7 if params['seasonal'] else None,initialization_method='estimated').fit(optimized=True).forecast(1)[0]
                preds.append(float(pred));records.append(dict(date=str(target_day.date()),
                    first_history_date=str(dates.iloc[day-history].date()),last_history_date=str(dates.iloc[day-1].date()),
                    history_sha256=__import__('hashlib').sha256(observed.tobytes()).hexdigest()))
            truth=data['raw']['target'][data['indices'][split],0].astype(np.float64)
            frame=pd.DataFrame(dict(date=pd.to_datetime(data['ds'][split].dates),true_model=truth,pred_model=preds,
                true_gwh=energy_model_to_gwh(truth),pred_gwh=energy_model_to_gwh(preds)))
            return frame,records
        train_x,train_y=design(data,'train');val_x,_=design(data,'val')
        trials=[];models=[];frames=[]
        for index,params in enumerate(grid):
            with threadpool_limits(limits=2):
                if statistical:model=None;frame,_=stat_predict('val',params)
                else:
                    if name=='Linear Regression':model=LinearRegression(n_jobs=2)
                    elif name=='Ridge Regression':model=Ridge(**params,solver='lsqr')
                    elif name=='XGBoost':
                        from xgboost import XGBRegressor
                        model=XGBRegressor(**params,random_state=42,n_jobs=2,objective='reg:squarederror',tree_method='hist')
                    else:
                        from lightgbm import LGBMRegressor
                        model=LGBMRegressor(**params,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
                    model.fit(train_x,train_y);frame=regression_predictions(data,'val',model.predict(val_x))
            trials.append(dict(trial=index,params=params,**metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')))
            models.append(model);frames.append(frame)
            save(run_dir/'tuning_trials.json',trials)
        best=min(range(len(trials)),key=lambda i:(trials[i]['mae_gwh'],i))
        model=models[best];params=grid[best]
        save(run_dir/'run_config.json',dict(name=name,history_days=history,fold=fold,seed=42,
            best_params=params,target_mode='delta',input_dimension=int(train_x.shape[1]),
            input_scope='Univariate target history only' if statistical else 'All flattened historical channels and target-day calendar',
            hpo_selection='Validation MAE only',protocol_sha256=sha(ROOT/'configs/unified_protocol.json')))
        payload={}
        for split in ['val','test']:
            if split=='val':frame=frames[best]
            elif statistical:
                with threadpool_limits(limits=2):frame,records=stat_predict(split,params)
                save(run_dir/'statistical_computation_record.json',dict(model=name,params=params,origins=records))
            else:frame=regression_predictions(data,split,model.predict(design(data,split)[0]))
            frame.to_csv(run_dir/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
            frame=pd.read_csv(run_dir/f'{split}_predictions.csv')
            payload[split]=metrics_gwh(frame.true_gwh,frame.pred_gwh,'GWh')
        if model is not None:
            joblib.dump(model,run_dir/'best_model.joblib');manifest['checkpoint_sha256']=sha(run_dir/'best_model.joblib')
            joblib.dump(data['fitted'],run_dir/'normalization.joblib')
        manifest.update(status='PASS',finished=utc(),training_seconds=time.perf_counter()-start,best_trial=best)
        save(run_dir/'metrics.json',payload);save(run_dir/'run_manifest.json',manifest)
        save(run_dir/'train_log.json',dict(trials=trials,manifest=manifest))
        print('PASS',run_dir.name,'val',round(payload['val']['mae_gwh'],4),'test',round(payload['test']['mae_gwh'],4),flush=True)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        save(run_dir/'run_manifest.json',manifest);raise
    return run_dir
