"""Publication figures from the reconciled tables, with complete image extents."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .common import ROOT,read,save,sha
from .analyses import chosen,refs

CN_LABELS={
    'Ours':'本文模型','ASTGCN-r adapted':'ASTGCN-r适配版',
    'No explicit persistence anchor':'无显式持续性锚点',
    'F0':'F0 双向交叉融合','F1':'F1 日内引导行业','F2':'F2 行业引导日内',
    'F3':'F3 直接拼接','F4':'F4 等参数拼接','F5':'F5 门控融合',
    'F6':'F6 自适应加权','F7':'F7 逐特征仿射融合','F8':'F8 共享特征注意力',
    'no_flat':'移除辅助历史路径',
    'Full model':'完整模型','full':'完整模型','static':'静态相关图','dynamic':'仅动态图',
    'no_topk':'不使用Top-K','attention':'注意力关系学习','mlp':'MLP关系学习',
    'no_intraday':'移除日内分支','no_graph':'移除行业图分支','no_anchor':'移除统计锚点',
    'no_persistence':'移除显式持续性项','no_industry_anchor':'移除行业引导锚点',
    'no_reconciliation':'移除校正残差','direct':'直接预测目标','delta':'变化量预测目标',
    'no_sales':'移除总售电量及派生量','no_all_industry':'移除全行业汇总',
    'no_both_totals':'移除两项总量及派生量','no_totals_derived':'移除总量及相关派生字段',
    'no_aggregate_nodes':'移除顶层汇总节点','no_load_aux':'移除负荷统计辅助损失',
    'no_industry_aux':'移除行业辅助损失','no_both_aux':'移除全部辅助损失',
    'Without intraday branch':'移除日内分支','Without industry graph branch':'移除行业图分支',
    'Static correlation graph':'静态相关图','Without learnable static association prior':'移除可学习静态先验',
    'Without Top-K sparsification':'移除Top-K稀疏化',
    'Concatenation instead of cross-guided fusion':'直接拼接代替交叉引导融合',
    'Without statistical anchor':'移除统计锚点','Without reconciliation residual':'移除校正残差',
    'Without auxiliary objectives':'移除辅助目标','ordinary_workday':'普通工作日','weekend':'周末',
    'statutory_holiday':'法定假期','spring_festival':'春节假期','spring_transition':'春节过渡期',
    'high_temperature':'高温日','low_temperature':'低温日','large_day_change':'日电量剧烈变化日',
    'Last-day Naive':'上一日朴素预测','Last-week Seasonal Naive':'上一周同日朴素预测',
    'Linear Regression':'线性回归','Ridge Regression':'岭回归','Exponential Smoothing':'指数平滑',
    'delay_1':'行业延迟1天','delay_2':'行业延迟2天','delay_3':'行业延迟3天','delay_7':'行业延迟7天'}
EN_LABELS={'Legacy_'+name:name+' (legacy local)' for name in ['MG-ASTL','TL-STLF','LoadFormer','STGNN-Load','ASTGCN-Load','GAT-LoadMixer']}
EN_LABELS['ASTGCN']='ASTGCN-r adapted'
EN_LABELS.update({
    'full':'Full model','static':'Static correlation graph','dynamic':'Dynamic graph only',
    'attention':'Attention relation learner','mlp':'MLP relation learner','no_topk':'Without Top-K',
    'no_anchor':'Without statistical anchor','no_industry_anchor':'Without industry-guided anchor',
    'no_persistence':'Without explicit persistence term','no_reconciliation':'Without reconciliation residual',
    'direct':'Direct target','delta':'Delta target','no_sales':'Without total sales and derivatives',
    'no_all_industry':'Without all-industry total','no_both_totals':'Without both totals and derivatives',
    'no_totals_derived':'Without totals and related fields','no_aggregate_nodes':'Without aggregate graph nodes',
    'no_load_aux':'Without profile auxiliary loss','no_industry_aux':'Without industry auxiliary loss',
    'no_both_aux':'Without both auxiliary losses','no_flat':'Without flat-history path',
    'F0':'F0 Bidirectional cross-attention','F1':'F1 Intraday guides industry','F2':'F2 Industry guides intraday',
    'F3':'F3 Concatenation','F4':'F4 Parameter-matched concatenation','F5':'F5 Gated fusion',
    'F6':'F6 Adaptive weighted fusion','F7':'F7 Feature-wise affine fusion','F8':'F8 Shared-feature attention',
    'ordinary_workday':'Ordinary workday','weekend':'Weekend','statutory_holiday':'Statutory holiday',
    'spring_festival':'Spring Festival','spring_transition':'Festival transition',
    'high_temperature':'High temperature','low_temperature':'Low temperature','large_day_change':'Large daily change',
    'delay_1':'Industry release delay 1 day','delay_2':'Industry release delay 2 days',
    'delay_3':'Industry release delay 3 days','delay_7':'Industry release delay 7 days',
    'adjacent_dates':'Adjacent dates','between_seeds_same_date':'Same date across seeds',
    'remove_aggregates_common_subgraph':'Remove aggregates: common subgraph',
    'summer':'Summer','autumn':'Autumn','winter':'Winter','spring':'Spring'})
CN_LABELS['ASTGCN']='ASTGCN-r适配版'
CN_LABELS.update({'Legacy_'+name:name+'（旧代码本地实现）' for name in ['MG-ASTL','TL-STLF','LoadFormer','STGNN-Load','ASTGCN-Load','GAT-LoadMixer']})
CN_LABELS['TL-STLF (legacy local)']='TL-STLF（旧代码本地实现）'

def build_figures(windows_only=False):
    fonts={entry.name for entry in font_manager.fontManager.ttflist}
    assert 'Times New Roman' in fonts and 'SimSun' in fonts
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':11,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False})
    destination=ROOT/'figures';destination.mkdir(exist_ok=True)
    records=[]
    def csv(name):return pd.read_csv(ROOT/'tables'/(name+'.csv'))
    def export(fig,number,name,lang,number_is_final=False):
        if number>=3 and not number_is_final:number+=1
        if not fig.get_constrained_layout():fig.tight_layout()
        stem=destination/f'{number:02d}_{name}_{lang}'
        for extension in ['png','pdf']:
            path=stem.with_suffix('.'+extension);fig.savefig(path,dpi=300)
            records.append(dict(number=number,subject=name,language=lang,path=str(path),sha256=sha(path)))
        plt.close(fig)
    def hbar(frame,label,title,cn,key='mae_gwh_mean',sd='mae_gwh_sd',highlight='Ours'):
        frame=frame.copy();labels=[textwrap.fill(CN_LABELS.get(str(value),str(value)) if cn else EN_LABELS.get(str(value),str(value)),34) for value in frame[label]]
        height=max(3.2,len(frame)*.43+1.1)
        fig,ax=plt.subplots(figsize=(7.0,height))
        errors=frame[sd].to_numpy() if sd in frame else None
        ax.barh(np.arange(len(frame)),frame[key],xerr=errors,
            color=['#356A9A' if str(value)==highlight else '#BDC7CF' for value in frame[label]],capsize=3)
        ax.set_yticks(np.arange(len(frame)),labels);ax.invert_yaxis();ax.set_xlim(left=0)
        ax.set_xlabel('MAE（GWh）' if cn else 'MAE (GWh)');ax.set_title(title)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        return fig
    for lang in ['EN','CN']:
        cn=lang=='CN'
        window=pd.read_csv(ROOT/'experiments/window_selection/tables/aligned_window_summary.csv')
        fig,axes=plt.subplots(2,1,figsize=(6.5,6.8));ax=axes[0]
        ax.errorbar(window.history_days,window.test_mae_gwh_mean,yerr=window.test_mae_gwh_sd,
            marker='o',capsize=4,color='#356A9A',label='测试集' if cn else 'Test')
        ax.errorbar(window.history_days,window.val_mae_gwh_mean,yerr=window.val_mae_gwh_sd,
            marker='s',capsize=4,color='#B47142',label='验证集' if cn else 'Validation')
        ax.set_xticks(window.history_days);ax.set_xlabel('两分支共同回看天数' if cn else 'Common history length (days)')
        ax.set_ylabel('MAE（GWh）' if cn else 'MAE (GWh)');ax.set_title('等长窗口比较 五种子均值与标准差' if cn else 'Aligned windows: mean and SD over five seeds')
        ax.legend();ax.grid(alpha=.2)
        controls=csv('Table_Window_Alignment_Controls').set_index('window').loc[['28/42','42/28','42/42']]
        axes[1].bar(np.arange(3),controls.test_mae_gwh_mean,yerr=controls.test_mae_gwh_sd,
            color=['#BDC7CF','#BDC7CF','#356A9A'],capsize=4)
        axes[1].set_xticks(np.arange(3),controls.index)
        axes[1].set_xlabel('日内历史天数 / 行业历史天数' if cn else 'Intraday history / industry history (days)')
        axes[1].set_ylabel('测试MAE（GWh）' if cn else 'Test MAE (GWh)')
        axes[1].set_title('不等长窗口对照 五种子均值与标准差' if cn else 'Asymmetric controls: five-seed mean and SD')
        export(fig,1,'Aligned_windows',lang)
        if windows_only:continue
        modern=csv('Table_E01_Baseline_Comparison')
        modern['model']=modern.model.replace({'ASTGCN':'ASTGCN-r adapted'})
        export(hbar(modern,'model','神经网络比较 五种子均值与标准差' if cn else 'Neural comparison: five-seed mean and SD',cn),2,'Neural_comparison',lang)
        local=csv('Table_Legacy_Related_Comparison')
        export(hbar(local,'model','旧代码本地实现 五种子均值与标准差' if cn else 'Legacy local implementations: five-seed mean and SD',cn),3,'Legacy_related_comparison',lang,number_is_final=True)
        classic=csv('Table_E02_Traditional_Comparison').rename(columns={'mae_gwh':'mae_gwh_mean'})
        own=modern[modern.model=='Ours'][['model','mae_gwh_mean','mae_gwh_sd']]
        classic=pd.concat([own,classic[['model','mae_gwh_mean']]],ignore_index=True)
        export(hbar(classic,'model','传统方法比较 主模型为五种子均值' if cn else 'Traditional methods; Ours shows five-seed mean',cn),3,'Traditional_comparison',lang)
        statistics=csv('Table_All_Implemented_Significance_Tests');statistics=statistics[statistics.block_days==7]
        fig,axes=plt.subplots(2,1,figsize=(7.5,7.0));legacy_mask=statistics.baseline.str.startswith('Legacy_')
        for ax,part,title in zip(axes,[statistics[~legacy_mask],statistics[legacy_mask]],
            ['已核实参考骨干','旧代码本地实现'] if cn else ['Verified reference backbones','Legacy local implementations']):
            center=part.mean_absolute_loss_difference_gwh.to_numpy();y=np.arange(len(part))
            ax.errorbar(center,y,xerr=np.stack([center-part.ci95_low_gwh,part.ci95_high_gwh-center]),fmt='o',capsize=3,color='#356A9A')
            labels=[CN_LABELS.get(value,value) if cn else EN_LABELS.get(value,value) for value in part.baseline]
            ax.axvline(0,color='#666666',ls='--',lw=1);ax.set_yticks(y,labels);ax.invert_yaxis()
            ax.set_xlabel('主模型减基线的MAE差值（GWh）' if cn else 'MAE difference: Ours minus baseline (GWh)');ax.set_title(title)
        fig.suptitle('种子与7日块联合重采样 95%区间' if cn else '95% intervals: joint seed and 7-day block bootstrap')
        export(fig,4,'Paired_uncertainty',lang)
        distributions=[];names=[]
        for name in read(ROOT/'configs/unified_protocol.json')['models']:
            losses=[]
            for entry,run_dir in refs('E03','seed',name):
                frame=pd.read_csv(run_dir/'test_predictions.csv')
                losses.append(np.abs(frame.pred_gwh-frame.true_gwh).to_numpy())
            distributions.append(np.mean(losses,axis=0));names.append(CN_LABELS.get(name,name) if cn else EN_LABELS.get(name,name))
        fig,ax=plt.subplots(figsize=(7.0,4.1))
        ax.boxplot(distributions,vert=False,tick_labels=names,showmeans=True)
        ax.invert_yaxis();ax.set_xlabel('每日绝对误差的五种子均值（GWh）' if cn else 'Daily absolute error averaged over five seeds (GWh)')
        ax.set_title('63个测试日的描述性分布' if cn else 'Descriptive distribution over 63 test dates')
        export(fig,5,'Daily_error_distribution',lang)
        graph=csv('Table_E04_Graph_Sensitivity')
        for number,prefix,label,title in [(6,'K','Top-K','Top-K'),(7,'prior','静态先验系数' if cn else 'Static-prior coefficient','Prior weight'),
                                          (8,'embed','节点嵌入维度' if cn else 'Node embedding dimension','Embedding dimension')]:
            part=graph[graph.variant.str.startswith(prefix)].copy()
            part['x']=part.variant.str[len(prefix):].astype(float);part=part.sort_values('x')
            if prefix=='embed':
                original=graph[graph.variant=='relation_full'].copy();original['x']=96.
                part=pd.concat([part,original],ignore_index=True).sort_values('x')
            fig,axes=plt.subplots(1,2,figsize=(8.0,3.6))
            for ax,metric in zip(axes,['mae','rmse']):
                ax.errorbar(part.x,part[metric+'_gwh_mean'],yerr=part[metric+'_gwh_sd'],marker='o',capsize=4,color='#356A9A')
                ax.set_xlabel(label);ax.set_ylabel(metric.upper()+' (GWh)');ax.set_xticks(part.x)
                ax.grid(alpha=.2)
            fig.suptitle((label+'敏感性 三个共同种子') if cn else title+': three matched seeds')
            export(fig,number,'Graph_'+prefix,lang)
        relations=graph[graph.variant.str.startswith('relation_')].copy()
        relations['variant']=relations.variant.str.replace('relation_','',regex=False)
        export(hbar(relations,'variant','图关系替代方法 三个共同种子' if cn else 'Alternative graph relations: three matched seeds',cn,highlight='full'),9,'Graph_relations',lang)
        fusion=csv('Table_E05_Fusion_Ablation');fusion=fusion[fusion.variant!='no_flat']
        export(hbar(fusion,'variant','融合方式比较 五种子均值与标准差' if cn else 'Fusion alternatives: five-seed mean and SD',cn,highlight='F0'),10,'Fusion_ablation',lang)
        modules=csv('Table_Module_Ablation_Matched_Three_Seeds')
        export(hbar(modules,'model_variant','模块消融 三个共同种子' if cn else 'Module ablation: three matched seeds',cn,highlight='Full model'),11,'Module_ablation',lang)
        for number,part in [(12,'Anchor'),(13,'Total_Feature'),(14,'Auxiliary_Loss')]:
            frame=csv('Table_E06_'+part+'_Ablation')
            title_cn={'Anchor':'统计锚点与目标形式消融','Total_Feature':'总量特征消融','Auxiliary_Loss':'辅助目标消融'}[part]
            export(hbar(frame,'variant',title_cn if cn else part.replace('_',' ')+' ablation',cn,highlight='full'),number,'Ablation_'+part,lang)
        deep=read(ROOT/'manifests/strongest_deep.json')['model']
        fig,ax=plt.subplots(figsize=(7.0,3.8))
        for i,name in enumerate(['Ours',deep]):
            frame=pd.read_csv(chosen(name)/'test_predictions.csv',parse_dates=['date']);frame=frame[frame.date<='2025-02-15']
            if i==0:ax.plot(frame.date,frame.true_gwh,label='实际值' if cn else 'Observed',color='#333333',lw=1.5)
            ax.plot(frame.date,frame.pred_gwh,label=CN_LABELS.get(name,name) if cn else name,lw=1.2,ls=['-','--'][i])
        ax.axvspan(pd.Timestamp('2025-01-28'),pd.Timestamp('2025-02-04'),color='#E5CBA8',alpha=.4)
        ax.set_ylabel('日总用电量（GWh）' if cn else 'Daily electricity consumption (GWh)')
        ax.set_title('春节及节后预测 固定种子42' if cn else 'Spring Festival and post-holiday forecasts: seed42')
        ax.legend(ncol=3,fontsize=10);fig.autofmt_xdate();export(fig,15,'Festival_forecasts',lang)
        scenario=csv('Table_E07_Scenario_Results');scenario=scenario[(scenario.model=='Ours')&(scenario.n_dates>0)]
        export(hbar(scenario,'subset','主模型各情境表现 五种子均值' if cn else 'Scenario performance: Ours, five-seed mean',cn),16,'Scenario_performance',lang)
        rolling=csv('Table_E08_Rolling_Results');seasons=['summer','autumn','winter','spring']
        models=['Ours','iTransformer','TSMixer','Legacy_TL-STLF','XGBoost','LightGBM']
        fig,axes=plt.subplots(1,2,figsize=(8.0,4.1),layout='constrained');ax=axes[0]
        width=.8/len(models);palette=['#356A9A','#B47142','#63987A','#8E7FA8','#8E969C','#637280']
        for i,name in enumerate(models):
            part=rolling[rolling.model==name].set_index('season').loc[seasons]
            ax.bar(np.arange(4)-.4+width/2+i*width,part.mae_gwh_mean,width,yerr=part.mae_gwh_sd.fillna(0),
                label=CN_LABELS.get(name,name) if cn else EN_LABELS.get(name,name),color=palette[i%len(palette)],capsize=2)
        part=rolling[rolling.model=='Last-week Seasonal Naive'].set_index('season').loc[seasons]
        axes[1].bar(np.arange(4),part.mae_gwh_mean,color='#C7AB61',width=.6)
        for axis in axes:
            axis.set_xticks(np.arange(4),['夏','秋','冬','春'] if cn else ['Summer','Autumn','Winter','Spring'])
            axis.set_ylabel('MAE（GWh）' if cn else 'MAE (GWh)')
            if not cn:axis.tick_params(axis='x',labelrotation=20)
        ax.set_title('神经网络与树模型' if cn else 'Neural and tree models')
        axes[1].set_title('上一周同日朴素预测' if cn else 'Weekly persistence')
        neural_tree=rolling[rolling.model.isin(models)]
        ax.legend(fontsize=8,ncol=2,loc='upper center',frameon=False)
        ax.set_ylim(0,(neural_tree.mae_gwh_mean+neural_tree.mae_gwh_sd.fillna(0)).max()*1.35)
        axes[1].set_ylim(0,part.mae_gwh_mean.max()*1.15)
        fig.suptitle('扩展窗口四季滚动验证' if cn else 'Expanding-window seasonal evaluation')
        export(fig,17,'Seasonal_rolling',lang)
        robust=csv('Table_E09_Robustness')
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        for i,category in enumerate(['intraday','industry','weather']):
            part=robust[robust.category==category].groupby('strength').mae_gwh.agg(['mean','std'])
            category_cn={'intraday':'日内负荷功率','industry':'行业售电量','weather':'气温'}[category]
            ax.errorbar(part.index*100,part['mean'],yerr=part['std'],label=category_cn if cn else category,marker=['o','s','^'][i],capsize=3)
        ax.set_xlabel('训练尺度的噪声比例（%）' if cn else 'Noise scale relative to training scale (%)')
        ax.set_ylabel('MAE（GWh）' if cn else 'MAE (GWh)');ax.set_title('固定权重 三个扰动种子均值与标准差' if cn else 'Frozen weights: three perturbation seeds, mean and SD');ax.legend()
        export(fig,18,'Noise_robustness',lang)
        delays=robust[robust.condition.str.startswith('delay_')].rename(columns={'mae_gwh':'mae_gwh_mean'})
        export(hbar(delays,'condition','行业记录发布延迟模拟' if cn else 'Simulated release delays in industry records',cn),19,'Release_delay',lang)
        cost=csv('Table_E10_Complexity');fig,ax=plt.subplots(figsize=(6.5,3.8))
        for index,(_,row) in enumerate(cost.iterrows()):
            name=CN_LABELS.get(row.model,row.model) if cn else EN_LABELS.get(row.model,row.model)
            ax.scatter(row.batch1_latency_ms_mean,row.mae_gwh,s=55,marker=['o','s','^','D'][index%4],
                label=f'{name} ({row.device})')
        ax.set_xlabel('单样本推理耗时（ms）' if cn else 'Single-sample inference latency (ms)');ax.set_ylabel('MAE（GWh）' if cn else 'MAE (GWh)')
        ax.set_xscale('log');ax.margins(.3);ax.legend(fontsize=9);ax.grid(alpha=.15)
        ax.set_title('固定种子42的精度与推理耗时' if cn else 'Accuracy and inference latency: seed42');export(fig,20,'Inference_cost',lang)
        graphfiles=[np.load(run_dir/'graphs.npz') for entry,run_dir in refs('E03','seed','Ours')]
        dates=pd.to_datetime(graphfiles[0]['dates'])
        assert all(np.array_equal(pd.to_datetime(item['dates']).to_numpy(),dates.to_numpy()) for item in graphfiles)
        adj=np.stack([item['adjacency'] for item in graphfiles]).mean(axis=0)
        holiday=(dates>='2025-01-28')&(dates<='2025-02-04');weekend=(dates.weekday>=5)&~dates.isin(pd.to_datetime(['2025-02-08']))
        matrices=[adj[~weekend&~holiday].mean(0),adj[weekend&~holiday].mean(0)]
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained');vmax=max(m.max() for m in matrices)
        for ax,matrix,title in zip(axes,matrices,['工作日','周末'] if cn else ['Workdays','Weekends']):
            im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap='YlOrRd');ax.set_title(title)
            ax.set_xlabel('节点索引' if cn else 'Node index');ax.set_ylabel('节点索引' if cn else 'Node index')
        fig.suptitle('五种子平均邻接' if cn else 'Mean adjacency across five seeds')
        fig.colorbar(im,ax=list(axes),shrink=.75);export(fig,21,'Workday_weekend_graphs',lang)
        stability=csv('Table_E11_Graph_Stability')
        keys=['adjacent_dates','between_seeds_same_date','remove_aggregates_common_subgraph']
        labels=['相邻日期','同日期不同种子','移除汇总节点'] if cn else ['Adjacent dates','Same date across seeds','Remove aggregate nodes']
        fig,axes=plt.subplots(1,2,figsize=(8.0,3.7))
        for index,(ax,key,label) in enumerate(zip(axes,['edge_jaccard_including_self','frobenius_difference'],
            ['边集合Jaccard','矩阵差异'] if cn else ['Edge-set Jaccard','Matrix difference'])):
            ax.boxplot([stability.loc[stability.comparison==item,key] for item in keys],vert=False,tick_labels=labels if index==0 else ['']*3,showmeans=True)
            ax.invert_yaxis();ax.set_xlabel(label)
        fig.suptitle('图关系稳定性 描述性分布' if cn else 'Graph stability: descriptive distributions')
        export(fig,22,'Graph_stability',lang)
        saved=np.load(chosen('Ours')/'graphs.npz');own_adj=saved['adjacency'];own_dates=pd.to_datetime(saved['dates'])
        before=np.load(ROOT/'results/predictions/E11/pre_festival_VALIDATION_context.npz')['adjacency'].mean(axis=0)
        festival=own_adj[(own_dates>='2025-01-28')&(own_dates<='2025-02-04')].mean(axis=0)
        after=own_adj[(own_dates>='2025-02-05')&(own_dates<='2025-02-11')].mean(axis=0)
        fig,axes=plt.subplots(1,3,figsize=(8.1,3.5),layout='constrained');matrices=[before,festival,after]
        titles=['节前 验证期背景','春节 测试期','节后 测试期'] if cn else ['Before: validation context','Festival: test','After: test']
        vmax=max(matrix.max() for matrix in matrices)
        for index,(ax,matrix,title) in enumerate(zip(axes,matrices,titles)):
            im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap='YlOrRd');ax.set_title(title,fontsize=10)
            ax.set_xlabel('节点索引' if cn else 'Node index')
            if index==0:ax.set_ylabel('节点索引' if cn else 'Node index')
        fig.suptitle('春节前后邻接 固定种子42' if cn else 'Adjacency around Spring Festival: seed42')
        fig.colorbar(im,ax=list(axes),shrink=.7);export(fig,23,'Festival_graph_context',lang)
    save(ROOT/('reports/window_figure_manifest.json' if windows_only else 'reports/figure_manifest.json'),records)
    print('SCIENTIFIC_FIGURES_CREATED',len(records))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--windows-only',action='store_true')
    build_figures(parser.parse_args().windows_only)
