"""Meaningful unit, temporal, masking, architecture and real optimizer checks."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hubei_revision.common import ROOT,read,save,sha,utc
from hubei_revision.engine import protocol,load_data,make_model,seed_all,inputs,run_training,inverse
from hubei_revision.models import ControlledModel
from hubei_revision.legacy.model import DualScaleLoadForecastModel
from hubei_revision.units import metrics_gwh,energy_model_to_gwh

cfg=protocol();data=load_data();assert cfg['history_days']==42
records=[]
def passed(check,**details):records.append(dict(check=check,status='PASS',**details))
q=[f'q_{i:02d}' for i in range(96)]
integration=float(np.max(np.abs(.25*data['df'][q].sum(axis=1)-data['df'].daily_load)))
assert integration<1e-6;passed('96-point power integration',max_model_difference=integration)
for split,n in [('train',289),('val',62),('test',63)]:
    assert len(data['ds'][split])==n
    i=data['indices'][split][0]
    day=int(np.where(pd.to_datetime(data['df'].date)==pd.Timestamp(data['raw']['dates'][i]))[0][0])
    assert np.array_equal(data['raw']['intraday_curve_hist'][i],data['df'].iloc[day-42:day][q].to_numpy(np.float32))
    assert np.array_equal(data['raw']['industry_hist'][i],data['df'].iloc[day-42:day][data['industry']].to_numpy(np.float32))
    passed(split+' same cutoff and dates',n=n,first=str(data['ds'][split].dates[0]),last=str(data['ds'][split].dates[-1]))
for mode in ['delta','direct']:
    d=load_data(dict(target_mode=mode));restored=inverse(d['scaled']['target'],d['raw']['base_load'],d)
    difference=float(np.abs(restored-d['raw']['target']).max()*10)
    assert difference<.001;passed(mode+' target roundtrip',max_gwh_difference=difference)
shock=data['df'].copy();shock.loc[pd.to_datetime(shock.date)>'2024-11-26',['daily_load','temp_avg']]*=10
shocked=load_data(override=shock)
for key,stat in data['fitted']['stats'].items():
    assert np.array_equal(stat.mean,shocked['fitted']['stats'][key].mean)
    assert np.array_equal(stat.std,shocked['fitted']['stats'][key].std)
assert data['fitted']['trans'].delta_mean==shocked['fitted']['trans'].delta_mean
passed('future measurements cannot affect fitted normalization')
removed=load_data(dict(remove=['sales_total']))
assert 'sales_total' not in removed['industry']
for name in ['sales_total','sales_gap','sales_ratio']:assert (removed['df'][name]==0).all()
passed('sales removal includes all derivative input paths')
batch={key:value.to('cuda') for key,value in next(iter(torch.utils.data.DataLoader(data['ds']['val'],batch_size=2))).items()}
seed_all(42);full=make_model('Ours',data,cfg['hpo'][0],'cuda').cuda().eval()
seed_all(42);reference=DualScaleLoadForecastModel(full.config).cuda().eval()
assert full.state_dict().keys()==reference.state_dict().keys()
assert all(torch.equal(full.state_dict()[key],reference.state_dict()[key]) for key in full.state_dict())
with torch.no_grad():assert torch.equal(full(inputs(batch))['prediction'],reference(inputs(batch))['prediction'])
passed('default architecture, initialization and forward equal compatible original',parameters=sum(p.numel() for p in full.parameters()))
for mode in ['F1','F2','F3','F4','F5','F6','F7','F8']:
    model=make_model('Ours',data,dict(fusion=mode),'cuda').cuda()
    out=model(inputs(batch));assert out['prediction'].shape==(2,4)
    out['prediction'].sum().backward()
    assert torch.isfinite(out['prediction']).all();passed('fusion '+mode+' real backward',parameters=sum(p.numel() for p in model.parameters()))
    del model
for opt in [dict(relation='static'),dict(relation='attention'),dict(relation='mlp'),dict(embedding_dim=8),dict(no_graph=True),dict(no_intraday=True),dict(no_anchor=True),dict(no_flat=True)]:
    model=make_model('Ours',data,opt,'cuda').cuda();out=model(inputs(batch));out['prediction'].sum().backward()
    assert torch.isfinite(out['prediction']).all();passed('controlled variant backward '+str(opt))
    del model
del full,reference;torch.cuda.empty_cache()
for name in cfg['models']:
    run_dir=run_training('SMOKE','end_to_end',name,cfg['hpo'][0],seed=991,smoke=True)
    assert read(run_dir/'run_manifest.json')['status']=='PASS'
    passed(name+' optimizer/checkpoint/inverse/metric end-to-end',run_dir=str(run_dir))
save(ROOT/'reports/preflight_audit.json',dict(status='PASS',checks=records,created=utc(),unit='GWh'))
pd.DataFrame(records).to_csv(ROOT/'reports/数据单位与协议自动验证.csv',index=False,encoding='utf-8-sig')
print('ALL_PREFLIGHT_CHECKS_PASS',len(records))
