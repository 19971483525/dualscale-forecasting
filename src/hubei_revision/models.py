"""Original full path plus explicit controlled variants and sourced adapters."""
import math
import importlib
import importlib.util
import sys
import types
import numpy as np
import torch
from torch import nn
from .common import ROOT
from .legacy.model import (DualScaleLoadForecastModel,ModelConfig,
    DynamicIndustryGraphBranch,CrossGuidedFusion)

class ControlledGraph(DynamicIndustryGraphBranch):
    def __init__(self,cfg,opt,corr):
        h=cfg.hidden_dim;e=opt.get("embedding_dim",h)
        super().__init__(cfg.industry_history,cfg.industry_nodes,cfg.global_dim,h,
            topk=opt.get("topk",8))
        if e!=h:
            self.node_proj=nn.Sequential(nn.Linear(cfg.industry_history,e),nn.GELU(),nn.Linear(e,e))
            self.graph_linear1=nn.Linear(e,h)
        self.relation=opt.get("relation","full")
        self.alpha=opt.get("prior_weight",1.)
        self.use_topk=self.relation!="no_topk"
        if self.relation=="static":
            self.register_buffer("training_correlation",torch.as_tensor(corr,dtype=torch.float32))
            self.static_adj.requires_grad_(False)
        elif self.relation=="attention":
            self.query=nn.Linear(e,e,bias=False);self.key=nn.Linear(e,e,bias=False)
        elif self.relation=="mlp":
            self.pair=nn.Sequential(nn.Linear(2*e,32),nn.GELU(),nn.Linear(32,1))

    def _dynamic_adjacency(self,z):
        n=z.shape[1]
        if self.relation=="static":
            logits=self.training_correlation.unsqueeze(0).expand(z.shape[0],-1,-1)
        elif self.relation=="attention":
            logits=self.query(z)@self.key(z).transpose(1,2)/math.sqrt(z.shape[-1])
        elif self.relation=="mlp":
            a=z[:,:,None,:].expand(-1,-1,n,-1);b=z[:,None,:,:].expand(-1,n,-1,-1)
            logits=self.pair(torch.cat([a,b],dim=-1)).squeeze(-1)
        else:
            normalized=nn.functional.normalize(z,dim=-1)
            logits=normalized@normalized.transpose(1,2)
        if self.relation not in ["static","dynamic"]:
            logits=logits+self.alpha*self.static_adj.unsqueeze(0)
        if self.use_topk and self.topk<n:
            values,indices=logits.topk(self.topk,dim=-1)
            logits=torch.full_like(logits,-torch.inf).scatter(-1,indices,values)
        return logits.softmax(-1)

class DirectionalFusion(nn.Module):
    def __init__(self,original,mode):
        super().__init__();self.original=original;self.mode=mode

    def forward(self,d,dt,s,st):
        # F1: intraday keys/values guide the industry query. F2 is the reverse.
        dg=torch.zeros_like(d);sg=torch.zeros_like(s)
        if self.mode=="F2":dg=self.original.day_query_month(d[None],st.transpose(0,1),st.transpose(0,1))[0][0]
        if self.mode=="F1":sg=self.original.month_query_day(s[None],dt.transpose(0,1),dt.transpose(0,1))[0][0]
        return self.original.norm(self.original.fusion(torch.cat([d,s,dg,sg],dim=-1)))

class AlternativeFusion(nn.Module):
    def __init__(self,h,mode,dropout,reference_parameters):
        super().__init__();self.mode=mode;self.norm=nn.LayerNorm(h)
        if mode in ["F3","F4"]:
            width=h if mode=="F3" else max(h,round((reference_parameters-3*h)/(3*h+1)))
            self.net=nn.Sequential(nn.Linear(2*h,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,h))
        elif mode=="F5":self.gate=nn.Linear(2*h,h)
        elif mode=="F6":self.gate=nn.Linear(2*h,2)
        elif mode=="F7":self.affine=nn.Linear(h,2*h)
        elif mode=="F8":self.attention=nn.MultiheadAttention(h,4,dropout=dropout,batch_first=True)
        else:raise ValueError(mode)

    def forward(self,d,dt,s,st):
        if self.mode in ["F3","F4"]:out=self.net(torch.cat([d,s],dim=-1))
        elif self.mode=="F5":
            gate=self.gate(torch.cat([d,s],dim=-1)).sigmoid();out=gate*d+(1-gate)*s
        elif self.mode=="F6":
            gate=self.gate(torch.cat([d,s],dim=-1)).softmax(-1);out=gate[:,:1]*d+gate[:,1:]*s
        elif self.mode=="F7":
            scale,shift=self.affine(s).chunk(2,dim=-1);out=d*(1+scale)+shift+s
        else:
            tokens=torch.stack([d,s],dim=1);out=self.attention(tokens,tokens,tokens)[0].mean(dim=1)
        return self.norm(out)

class ControlledAnchor(nn.Module):
    def __init__(self,original,mode,sales_available):
        super().__init__();self.original=original;self.mode=mode;self.sales_available=sales_available

    def forward(self,x,p):
        if not self.sales_available:p=torch.cat([torch.zeros_like(p[:,:1]),p[:,1:]],dim=-1)
        anchor,ratio=self.original(x,p)
        if self.mode=="full":return anchor,ratio
        last=x[:,-1];mean7=x[:,-7:].mean(dim=1)
        a=last[:,:1];b=x[:,-7,:1];peak=last[:,1:2];mean96=last[:,3:4]
        mix=self.original.blend_gate(torch.cat([last,mean7,a,b,peak,mean96,p[:,:1]],dim=-1)).softmax(-1)
        persistence=mix[:,:1]*a+mix[:,1:]*b
        if self.mode=="no_persistence":return anchor-.3*persistence,ratio
        if self.mode=="no_industry_anchor":return .3*persistence,torch.zeros_like(ratio)
        raise ValueError(self.mode)

class ZeroTokens(nn.Module):
    def forward(self,x):return torch.zeros_like(x)

class ControlledModel(DualScaleLoadForecastModel):
    def __init__(self,shapes,opt,corr,industry):
        h=opt.get("hidden_dim",96)
        cfg=ModelConfig(intraday_history=shapes["intraday_curve_hist"][0],
            industry_history=shapes["industry_hist"][0],intraday_feat_dim=shapes["intraday_daily_hist"][-1],
            industry_nodes=shapes["industry_hist"][-1],global_dim=shapes["global_hist"][-1],
            target_calendar_dim=shapes["target_calendar"][-1],hidden_dim=h,dropout=opt.get("dropout",.04),
            patch_sizes=[3,6,12],num_heads=4,use_intraday_branch=not opt.get("no_intraday",False),
            use_industry_graph_branch=not opt.get("no_graph",False),
            use_statistical_anchor=not opt.get("no_anchor",False),
            use_reconciliation=not opt.get("no_reconciliation",False),
            use_static_adjacency=not opt.get("no_static_prior",False),
            use_topk=not opt.get("no_topk",False))
        # Default construction and forward path are identical to the archived,
        # checkpoint-compatible full model; no replacement of default modules.
        super().__init__(cfg)
        graph_change=(opt.get("relation","full")!="full" or opt.get("topk",8)!=8 or
            opt.get("prior_weight",1.)!=1. or opt.get("embedding_dim",h)!=h)
        if graph_change:
            previous=self.industry_branch.state_dict()
            replacement=ControlledGraph(cfg,opt,corr)
            state=replacement.state_dict()
            for key in state:
                if key in previous and state[key].shape==previous[key].shape:
                    state[key]=previous[key].clone()
            replacement.load_state_dict(state,strict=True)
            self.industry_branch=replacement
        fusion=opt.get("fusion","F0")
        if fusion in ["F1","F2"]:self.fusion=DirectionalFusion(self.fusion,fusion)
        elif fusion!="F0":
            ref=sum(p.numel() for p in self.fusion.parameters())
            self.fusion=AlternativeFusion(h,fusion,cfg.dropout,ref)
        anchor=opt.get("anchor","full")
        if anchor!="full" or "sales_total" not in industry:
            self.anchor_head=ControlledAnchor(self.anchor_head,anchor,"sales_total" in industry)
        if opt.get("no_flat",False):self.flat_backbone=ZeroTokens()

class OfficialAdapter(nn.Module):
    """Public reference backbone plus the disclosed daily-total/auxiliary head."""
    def __init__(self,name,channels,history,opt,device,corr):
        super().__init__();self.name=name
        h=opt.get("hidden_dim",96)
        if name=="ASTGCN":
            sys.path.insert(0,str(ROOT/'vendor/astgcn'))
            mod=importlib.import_module('model.ASTGCN_r')
            self.native=mod.make_model(torch.device(device),opt.get('depth',1),1,3,h,h,1,corr,1,history,channels)
        else:
            sys.path.insert(0,str(ROOT/'vendor/tslib'))
            config=types.SimpleNamespace(task_name='long_term_forecast',seq_len=history,label_len=0,pred_len=1,
                enc_in=channels,c_out=channels,d_model=h,d_ff=2*h,e_layers=opt.get('depth',1),
                dropout=opt.get('dropout',.04),factor=3,n_heads=4,activation='gelu',embed='timeF',freq='d',
                top_k=3,num_kernels=3,num_class=1,channel_independence=0,decomp_method='moving_avg',
                down_sampling_layers=1,down_sampling_window=2,down_sampling_method='avg',moving_avg=5,
                use_norm=1,features='M',patch_len=opt.get('patch_len',12))
            spec=importlib.util.spec_from_file_location('hubei_official_'+name,ROOT/'vendor/tslib/models'/(name+'.py'))
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            self.native=mod.Model(config,patch_len=config.patch_len,stride=max(1,config.patch_len//2)) if name=='PatchTST' else mod.Model(config)
        self.readout=nn.Linear(channels+7,4)

    def forward(self,batch):
        x=batch['flat_hist']
        if self.name=='ASTGCN':out=self.native(x.transpose(1,2).unsqueeze(2)).squeeze(-1)
        else:
            out=self.native(x,None,None,None)
            if isinstance(out,tuple):out=out[0]
            out=out[:,-1,:]
        return {'prediction':self.readout(torch.cat([out,batch['target_calendar']],dim=-1))}
