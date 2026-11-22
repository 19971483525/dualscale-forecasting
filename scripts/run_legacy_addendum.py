"""Retest old local implementations with a separately frozen, disclosed addendum."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hubei_revision.common import ROOT,read,save,sha,utc
from hubei_revision import engine
from hubei_revision.legacy_factory import install

p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--after-main',action='store_true');args=p.parse_args()
if args.after_main:
    print('WAITING_FOR_ORIGINAL_SUITE_TO_FINISH',flush=True)
    while not (ROOT/'manifests/E11_status.json').exists():
        for group in ['E09','E10','E11']:
            status=ROOT/f'manifests/{group}_status.json'
            if status.exists() and read(status).get('status')=='FAILED':raise RuntimeError('Original suite failed: '+group)
        time.sleep(5)
    assert read(ROOT/'manifests/E11_status.json')['numerical_experiments']=='PASS'
path=ROOT/'configs/legacy_comparison_addendum.json'
if not path.exists():
    sources=['src/hubei_revision/legacy_local/baselines.py','src/hubei_revision/legacy_local/model.py',
        'src/hubei_revision/legacy_factory.py']
    architectures={
        'MG-ASTL':dict(hidden_dim=80,dropout=0.,depth=1,description='Old ConvInception backbone; no graph or multitask decomposition.'),
        'TL-STLF':dict(hidden_dim=80,dropout=.1,depth=2,description='Old Mixer backbone; no source-domain transfer stage.'),
        'LoadFormer':dict(hidden_dim=80,dropout=.1,depth=2,patch_len=7,description='Old two-layer patch Transformer; no verified LoadFormer-specific block.'),
        'STGNN-Load':dict(hidden_dim=80,dropout=.1,depth=2,description='Old inverted variable Transformer; not a verified spatiotemporal graph network.'),
        'ASTGCN-Load':dict(hidden_dim=96,dropout=0.,depth=1,description='Old ConvInception backbone; separate from the sourced ASTGCN-r adapter.'),
        'GAT-LoadMixer':dict(hidden_dim=72,dropout=.1,depth=3,description='Old Mixer backbone; no graph-attention layer.')}
    save(path,dict(stage='FROZEN',created=utc(),authorization='User requested retaining old local implementations because official code is unavailable.',
        scope='Disclosed legacy local controls, not authenticated reproductions of the named papers.',
        base_protocol_sha256=sha(ROOT/'configs/unified_protocol.json'),
        source_sha256={relative:sha(ROOT/relative) for relative in sources},architectures=architectures,
        learning_rates=[.0005,.001,.0002],seeds=engine.protocol()['seeds'],
        rolling_method='TL-STLF',rolling_folds=engine.protocol()['rolling_folds']))
cfg=read(path);assert cfg['base_protocol_sha256']==sha(ROOT/'configs/unified_protocol.json')
install()

def options(historical,lr):
    architecture=cfg['architectures'][historical]
    return {key:value for key,value in architecture.items() if key!='description'}|dict(lr=lr,legacy_addendum_sha256=sha(path))

if not (ROOT/'reports/legacy_preflight.json').exists():
    data=engine.load_data();batch=next(iter(torch.utils.data.DataLoader(data['ds']['val'],batch_size=2)))
    records=[]
    for historical in cfg['architectures']:
        name='Legacy_'+historical;opt=engine.normalized_options(name,options(historical,.0005),engine.protocol())
        engine.seed_all(991);model=engine.make_model(name,data,opt,'cpu');optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
        for _ in range(2):
            model.train();optimizer.zero_grad();out=model(engine.inputs(batch))['prediction'];loss=(out-batch['target']).square().mean();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():before=model(engine.inputs(batch))['prediction']
        fresh=engine.make_model(name,data,opt,'cpu');fresh.load_state_dict(model.state_dict(),strict=True);fresh.eval()
        with torch.no_grad():after=fresh(engine.inputs(batch))['prediction']
        error=float((before-after).abs().max());assert error<1e-7
        physical=engine.energy_model_to_gwh(engine.inverse(before.numpy(),batch['base_load'].numpy(),data)[:,0])
        truth=engine.energy_model_to_gwh(data['raw']['target'][data['indices']['val'][:2],0])
        unit_score=engine.metrics_gwh(truth,physical,'GWh')
        records.append(dict(model=name,status='PASS',optimizer_steps=2,reload_normalized_max_difference=error,
            registered_parameters=sum(value.numel() for value in model.parameters()),metric_unit=unit_score['absolute_error_unit']))
    save(ROOT/'reports/legacy_preflight.json',dict(status='PASS',models=records,addendum_sha256=sha(path),created=utc()))
    print('LEGACY_PREFLIGHT_PASS',len(records),flush=True)
if args.prepare_only:raise SystemExit(0)

def select(historical,group,label='',fold=None):
    name='Legacy_'+historical;trials=[]
    for index,lr in enumerate(cfg['learning_rates']):
        opt=options(historical,lr)
        run_dir=engine.run_training(group,label+'trial'+str(index),name,opt,42,fold,test_after=False)
        trials.append(dict(trial=index,run_dir=str(run_dir),options=opt,validation_mae_gwh=read(run_dir/'metrics.json')['val']['mae_gwh']))
    result=min(trials,key=lambda row:(row['validation_mae_gwh'],row['trial']))
    selected=dict(**result,trials=trials,selection='Validation MAE only; unchanged old architecture; three learning rates.')
    save(ROOT/'manifests'/('selected_'+label+name+'.json'),selected)
    engine.ensure_test(Path(result['run_dir']))
    for seed in cfg['seeds']:engine.run_training(group,label+'seed',name,result['options'],seed,fold)

save(ROOT/'manifests/LEGACY_status.json',dict(status='RUNNING',started=utc()))
for historical in cfg['architectures']:select(historical,'LEGACY')
for label,fold in cfg['rolling_folds'].items():select(cfg['rolling_method'],'LEGACY_ROLL',label+'_',fold)
save(ROOT/'manifests/LEGACY_status.json',dict(status='PASS',numerical_experiments='PASS',created=utc(),
    main_seed_records=30,rolling_seed_records=20,literature_fidelity='UNVERIFIED_LOCAL_IMPLEMENTATIONS_DISCLOSED'))
print('LEGACY_LOCAL_COMPARISONS_COMPLETE',flush=True)
