from pathlib import Path
import json,hashlib,csv,shutil,itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'evidence'
RUN=BASE/'replication'
A=ROOT/'data'

rows=[json.loads(l) for l in (RUN/'records.jsonl').read_text().splitlines()];m=json.loads((RUN/'manifest.json').read_text())
checks={'rows':len(rows),'unique_conditions':len({tuple(r[k] for k in ['dataset','seed','fraction','model','noise_ratio']) for r in rows}),'original_prediction_checks':0,'original_prediction_mismatches':0,'repeated_prediction_mismatches':0,'first_second_mismatches':0}
seen={}
for r in rows:
 d,s,f,mod,q,rep=[r[k] for k in ['dataset','seed','fraction','model','noise_ratio','repeat']]
 z=np.load(RUN/'predictions'/f'{d}_{s}_{f}_{mod}_{q}_{rep}.npz');key=(d,s,f,mod,q)
 if key in seen and not np.array_equal(z['predicted'],seen[key]):checks['repeated_prediction_mismatches']+=1
 seen[key]=z['predicted'].copy()
 checks['first_second_mismatches']+=int(not np.array_equal(z['predicted'],z['second_prediction']))
 if mod!='CatBoost':
  name=f'{d}_{s}_{f:.2f}_{mod}'+(f'_noise{q:.2f}' if q else '')+'.csv'
  old=pd.read_csv(BASE/'prior_predictions'/('noise' if q else 'baseline')/'predictions'/name)
  checks['original_prediction_checks']+=1
  checks['original_prediction_mismatches']+=int(not np.array_equal(old['predicted'],z['predicted']))
checks['source_input_hash_mismatches']=[n for n,h in m['source_hashes'].items() if n!='protocol.json' and (BASE/'runner'/n).exists() and hashlib.sha256((BASE/'runner'/n).read_bytes()).hexdigest()!=h]
t=pd.read_csv(A/'dataset_all_condition_contrasts.csv');sens=[]
for model,g in t.groupby('baseline'):
 for d in g.dataset:sens.append({'baseline':model,'omitted':d,'contrast_pp':g[g.dataset!=d].baseline_minus_tabpfn_loss_pp.mean()})
pd.DataFrame(sens).to_csv(ROOT/'data/lodo_all_conditions.csv',index=False)
rdf=pd.DataFrame(rows);times=[]
for (d,s,f,mod),g in rdf.groupby(['dataset','seed','fraction','model']):
 clean=g[g.noise_ratio==0].set_index('repeat').fit_first_predict_seconds
 for q in [.5,1.]:
  aug=g[g.noise_ratio==q].set_index('repeat').fit_first_predict_seconds
  ratios=aug/clean
  times.append(dict(dataset=d,seed=s,fraction=f,model=mod,noise_ratio=q,paired_median=np.median(ratios),ratio_medians=np.median(aug)/np.median(clean),geomean=np.exp(np.log(ratios).mean())))
tm=pd.DataFrame(times);td=tm.groupby(['dataset','fraction','model','noise_ratio']).mean(numeric_only=True).reset_index();to=td.groupby(['fraction','model','noise_ratio'])[['paired_median','ratio_medians','geomean']].mean().reset_index();to.to_csv(ROOT/'data/timing_sensitivity.csv',index=False)
checks['timing_lodo_half_fullnoise']=[100*(td[(td.model=='TabPFN')&(td.fraction==.5)&(td.noise_ratio==1)&(td.dataset!=d)].paired_median.mean()-1) for d in m['datasets']]
(ROOT/'audit/verification.json').write_text(json.dumps(checks,indent=2))
allc=pd.read_csv(A/'all_conditions.csv');tasks=pd.read_csv(A/'dataset_cells.csv');mods=['LogisticRegression','RandomForest','CatBoost','TabPFN'];names=['LR','RF','CatBoost','TabPFN V2'];colors=['#0072B2','#D55E00','#9467BD','#009E73'];markers=['s','^','D','o']
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
for metric,filename,ylabel in [('loss_pp','augmentation_loss','Balanced-accuracy loss (pp)'),('median_paired_time_ratio','runtime_overhead','Pipeline overhead (%)')]:
 fig,axes=plt.subplots(1,2,figsize=(7.1,2.35),sharey=True)
 for ax,q in zip(axes,[.5,1.]):
  for mod,name,color,marker in zip(mods,names,colors,markers):
   g=allc[(allc.model==mod)&(allc.noise_ratio==q)].sort_values('fraction');y=g[metric] if metric=='loss_pp' else 100*(g[metric]-1)
   ax.plot(g.fraction,y,label=name,color=color,marker=marker,ms=4,lw=1.5)
  ax.axhline(0,color='.4',lw=.6,ls=':');ax.set_xticks([.1,.5,1.],['10%','50%','100%']);ax.set_xlabel('Training-pool fraction');ax.set_title(f'Added columns: {int(q*100)}% of original',fontsize=9);ax.grid(axis='y',alpha=.2)
  ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel(ylabel);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
 fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(ROOT/'figures'/f'{filename}.pdf');fig.savefig(ROOT/'figures'/f'{filename}.png',dpi=200);plt.close(fig)
# Exact table rows generated from validated output, not hand-entered.
lines=[]
for model,name in zip(mods,names):
 g=allc[(allc.model==model)&(allc.fraction==.5)].set_index('noise_ratio')
 lines.append(f"{name} & {100*g.loc[0,'balanced_accuracy']:.2f} & {100*g.loc[1,'balanced_accuracy']:.2f} & {1000*g.loc[0,'median_fit_first_predict_seconds']:.2f} & {1000*g.loc[1,'median_fit_first_predict_seconds']:.2f} \\\\")
(ROOT/'data/absolute_rows.tex').write_text('\n'.join(lines))
ct=pd.read_csv(A/'all_condition_contrasts.csv');lines=[]
for model,name in zip(mods[:3],names[:3]):
 g=ct[ct.baseline==model].iloc[0];l=pd.DataFrame(sens);v=l[l.baseline==model].contrast_pp
 lines.append(f"{name} & {g.baseline_minus_tabpfn_loss_pp:.3f} & {int(g.positive_datasets)}/6 & {v.min():.3f} to {v.max():.3f} \\\\")
(ROOT/'data/contrast_rows.tex').write_text('\n'.join(lines))
print(json.dumps(checks,indent=2));print('All-condition mean losses',allc[allc.noise_ratio>0].groupby('model').loss_pp.mean().to_dict());print('Sensitivities',to[(to.model=='TabPFN')&(to.noise_ratio==1)].to_dict('records'));print('Task contrasts',t.to_dict('records'))
