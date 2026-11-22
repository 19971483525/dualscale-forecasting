"""Explicit audited model-factory extension for the user-authorized old controls."""
from .common import ROOT,read,sha
from .legacy_local.baselines import create_load_specific_baselines

def install():
    from . import engine
    if getattr(engine.make_model,'legacy_extension',False):return
    original=engine.make_model
    def extended(name,data,opt,device):
        if not name.startswith('Legacy_'):return original(name,data,opt,device)
        path=ROOT/'configs/legacy_comparison_addendum.json';config=read(path)
        assert config['stage']=='FROZEN'
        assert opt['legacy_addendum_sha256']==sha(path)
        for relative,digest in config['source_sha256'].items():assert sha(ROOT/relative)==digest,relative
        historical=name[len('Legacy_'):]
        assert historical in config['architectures']
        # Preserve the old factory's constructor order and all architecture defaults.
        models=create_load_specific_baselines(data['shapes']['flat_hist'][0],data['shapes']['flat_hist'][-1])
        return models[historical]
    extended.legacy_extension=True
    engine.make_model=extended
