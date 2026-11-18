"""Validate a complete single session, then produce descriptive paired summaries."""
from pathlib import Path
import argparse,json,csv,itertools,datetime
from collections import defaultdict
import numpy as np

def mean(v):return float(np.mean(v))
def write_csv(p,rows):
 if not rows:return
 with p.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def analyze(path,out,allow_test=False):
 path=Path(path);out=Path(out)
 m=json.loads((path/'manifest.json').read_text());c=json.loads((path/'completion.json').read_text())
 if m['mode'] in ['smoke','pilot'] and not allow_test:raise ValueError('Smoke/pilot is not publication evidence')
 if not c['complete'] or c['session_id']!=m['session_id']:raise ValueError('Incomplete or mixed session')
 if c['hardware_end']!=m['hardware']:raise ValueError('Hardware metadata changed')
 rows=[json.loads(l) for l in (path/'records.jsonl').read_text().splitlines()]
 expected=set(itertools.product(m['datasets'],m['seeds'],m['fractions'],m['models'],m['ratios'],range(m['repeats'])))
 if len(rows)!=len(expected) or c['completed']!=len(expected) or m['expected']!=len(expected):raise ValueError('Wrong evaluation count')
 keys=set();last=m['process_started_monotonic'];matched={};checked=0;warnings=[]
 for r in rows:
  key=tuple(r[k] for k in ['dataset','seed','fraction','model','noise_ratio','repeat'])
  if key not in expected or key in keys or r['session_id']!=m['session_id']:raise ValueError('Duplicate, unexpected, or mixed-session record')
  keys.add(key)
  if r['monotonic_end']<=last:raise ValueError('Nonmonotonic session clock')
  last=r['monotonic_end']
  if not m['started']<=r['start_utc']<=r['end_utc']<=c['ended']:raise ValueError('Timestamp outside session')
  for k in ['construct_seconds','fit_seconds','first_predict_seconds','repeat_predict_seconds','fit_first_predict_seconds']:
   if not np.isfinite(r[k]) or r[k]<=0:raise ValueError('Invalid duration')
  if not np.isclose(r['fit_first_predict_seconds'],r['fit_seconds']+r['first_predict_seconds']):raise ValueError('Timing endpoint mismatch')
  stem='_'.join(map(str,key));z=np.load(path/'predictions'/f'{stem}.npz');y=z['true'];pred=z['predicted'];idx=z['row_index']
  if len(y)!=r['n_test'] or y.shape!=pred.shape or len(np.unique(idx))!=len(y):raise ValueError('Prediction shape or duplicate test rows')
  ba=mean([mean(pred[y==v]==v) for v in np.unique(y)])
  if not np.isclose(ba,r['balanced_accuracy'],atol=1e-12,rtol=0):raise ValueError('Saved accuracy disagrees with predictions')
  k=(r['dataset'],r['seed'])
  if k in matched:
   if not np.array_equal(matched[k][0],idx) or not np.array_equal(matched[k][1],y):raise ValueError('Unmatched test examples')
  else:matched[k]=(idx.copy(),y.copy())
  if not np.array_equal(pred,z['second_prediction']):warnings.append({'key':key,'issue':'repeat prediction differs'})
  if r['warnings']:warnings.append({'key':key,'issue':r['warnings']})
  checked+=1
 if keys!=expected:raise ValueError('Missing grid cells')
 # First average accuracy over technical repetitions, never count them as datasets.
 bycell=defaultdict(list)
 for r in rows:bycell[tuple(r[k] for k in ['dataset','seed','fraction','model','noise_ratio'])].append(r)
 cells=[]
 for k,v in sorted(bycell.items()):
  d,s,f,model,q=k;clean=bycell[(d,s,f,model,0.0)];clean_rep={r['repeat']:r for r in clean}
  ba=mean([r['balanced_accuracy'] for r in v]);cleanba=mean([r['balanced_accuracy'] for r in clean])
  ratios=[r['fit_first_predict_seconds']/clean_rep[r['repeat']]['fit_first_predict_seconds'] for r in v]
  cells.append(dict(dataset=d,seed=s,fraction=f,model=model,noise_ratio=q,balanced_accuracy=ba,loss_pp=100*(cleanba-ba),median_paired_time_ratio=float(np.median(ratios)),trial_ratio_q25=float(np.quantile(ratios,.25)),trial_ratio_q75=float(np.quantile(ratios,.75)),median_fit_first_predict_seconds=float(np.median([r['fit_first_predict_seconds'] for r in v])),median_construct_seconds=float(np.median([r['construct_seconds'] for r in v])),median_fit_seconds=float(np.median([r['fit_seconds'] for r in v])),median_first_predict_seconds=float(np.median([r['first_predict_seconds'] for r in v])),median_repeat_predict_seconds=float(np.median([r['repeat_predict_seconds'] for r in v]))))
 def aggregate(rows,group,metrics):
  groups=defaultdict(list)
  for r in rows:groups[tuple(r[k] for k in group)].append(r)
  return [dict(zip(group,k),**{x:mean([r[x] for r in v]) for x in metrics}) for k,v in sorted(groups.items())]
 metrics=['balanced_accuracy','loss_pp','median_paired_time_ratio','median_fit_first_predict_seconds','median_construct_seconds','median_fit_seconds','median_first_predict_seconds','median_repeat_predict_seconds']
 tasks=aggregate(cells,['dataset','fraction','model','noise_ratio'],metrics)
 overall=aggregate(tasks,['fraction','model','noise_ratio'],metrics)
 contrasts=[];index={(r['dataset'],r['fraction'],r['model'],r['noise_ratio']):r for r in tasks}
 if 'TabPFN' in m['models']:
  for r in tasks:
   if r['noise_ratio']==0 or r['model']=='TabPFN':continue
   t=index[(r['dataset'],r['fraction'],'TabPFN',r['noise_ratio'])]
   contrasts.append(dict(dataset=r['dataset'],fraction=r['fraction'],baseline=r['model'],noise_ratio=r['noise_ratio'],baseline_minus_tabpfn_loss_pp=r['loss_pp']-t['loss_pp']))
 per_task=aggregate(contrasts,['dataset','baseline'],['baseline_minus_tabpfn_loss_pp'])
 primary=aggregate(per_task,['baseline'],['baseline_minus_tabpfn_loss_pp'])
 for r in primary:
  vals=[x['baseline_minus_tabpfn_loss_pp'] for x in per_task if x['baseline']==r['baseline']]
  r.update(n_datasets=len(vals),min_dataset_contrast=min(vals),max_dataset_contrast=max(vals),positive_datasets=sum(x>0 for x in vals))
 out.mkdir(parents=True,exist_ok=False)
 for name,data in [('seed_cells',cells),('dataset_cells',tasks),('all_conditions',overall),('dataset_condition_contrasts',contrasts),('dataset_all_condition_contrasts',per_task),('all_condition_contrasts',primary)]:write_csv(out/f'{name}.csv',data)
 result={'verified_predictions':checked,'session_id':m['session_id'],'cohort':m['cohort'],'mode':m['mode'],'primary_accuracy_eligible':m['cohort']=='new' and m['mode']=='accuracy','primary_timing_eligible':m['cohort']=='original' and m['mode']=='timing','warnings':warnings,'interpretation':'Positive contrast means smaller TabPFN augmentation loss, not necessarily higher absolute accuracy. Timing ratio minus one times 100 is percent overhead. Summaries are descriptive, with equal dataset weights. No significance claims.'}
 (out/'validation.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);p.add_argument('--allow-test',action='store_true');a=p.parse_args();print(json.dumps(analyze(a.run,a.out,a.allow_test),indent=2))
