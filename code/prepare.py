"""Select datasets before running models; download failures stop, never silently replace."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2))
def prepare(out):
 import openml
 openml.config.set_root_cache_directory(str(ROOT.parent/"openml_extension_cache"))
 from sklearn.preprocessing import LabelEncoder
 from sklearn.model_selection import train_test_split
 import sys
 sys.path.insert(0,str(ROOT/'original'));from benchmark import nested_order
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((ROOT/'protocol.json').read_text());rules=cfg['eligibility']
 suite=openml.study.get_suite(cfg['suite_id'])
 tasks=sorted(suite.tasks);np.random.default_rng(cfg['selection_seed']).shuffle(tasks)
 save(out/'suite_snapshot.json',{'suite_id':cfg['suite_id'],'task_order':tasks})
 selected=[];log=[];seen=set();seen_names=set()
 try:
  for tid in tasks:
   task=openml.tasks.get_task(int(tid));did=int(task.dataset_id)
   entry={'task_id':int(tid),'dataset_id':did};log.append(entry)
   if did in seen or did in cfg['excluded_dataset_ids']:
    entry['status']='excluded_original_or_duplicate';continue
   seen.add(did)
   ds=openml.datasets.get_dataset(did,download_data=False);entry['name']=ds.name
   family='mfeat' if ds.name.lower().startswith('mfeat') else ds.name.lower()
   if ds.name.lower() in cfg['excluded_dataset_names']:entry['status']='excluded_original_name';continue
   if family in seen_names:entry['status']='duplicate_dataset_family';continue
   # Check metadata before fetching large tables.
   quality=ds.qualities or {};n=quality.get('NumberOfInstances');p=quality.get('NumberOfFeatures')
   if n is not None and not rules['min_rows']<=n<=rules['max_rows']:entry['status']='row_limit';continue
   if p is not None and p>rules['max_features']+1:entry['status']='feature_limit';continue
   X,y,categorical,names=ds.get_data(target=task.target_name,dataset_format='dataframe')
   if any(categorical):entry['status']='categorical_features';continue
   if y.isna().any():entry['status']='missing_labels';continue
   X=X.to_numpy(dtype=np.float64);y=LabelEncoder().fit_transform(y).astype(np.int64)
   counts=np.unique(y,return_counts=True)[1]
   if not (rules['min_rows']<=len(y)<=rules['max_rows'] and rules['min_features']<=X.shape[1]<=rules['max_features'] and 2<=len(counts)<=rules['max_classes'] and min(counts)>=rules['min_class_count']):
    entry['status']='actual_shape_or_class_limit';continue
   if np.isinf(X).any():entry['status']='infinite_features';continue
   name=f'openml_{did}';seen_names.add(family);(out/name).mkdir()
   np.savez_compressed(out/name/'data.npz',X=X,y=y)
   for seed in cfg['seeds']:
    pool,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
    order=pool[nested_order(y[pool],seed)]
    np.savez_compressed(out/name/f'split_{seed}.npz',ordered_training_pool=order,test=test)
    raw=hashlib.sha256(f'extension-noise-v1:{did}:{seed}'.encode()).digest()
    noise=np.random.default_rng(int.from_bytes(raw[:8],'little')).standard_normal(X.shape)
    np.savez_compressed(out/name/f'noise_{seed}.npz',noise=noise)
   entry.update(status='selected',key=name,rows=len(y),features=X.shape[1],class_counts=counts.tolist(),version=ds.version,family=family,feature_names=names,missing_by_feature=np.isnan(X).sum(axis=0).tolist(),source_url=f'https://www.openml.org/d/{did}')
   selected.append(dict(entry))
   if len(selected)==cfg['max_new_datasets']:break
  if not selected:raise RuntimeError('No eligible datasets. Do not broaden rules after seeing outcomes.')
  save(out/'selection.json',{'selected':selected,'screening':log,'remaining_tasks_not_screened':tasks[len(log):]})
  save(out/'freeze.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':digest(ROOT/'protocol.json'),'files':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException:
  save(out/'screening_incomplete.json',log);raise
 print(f'Frozen {len(selected)} datasets. No model performance was used.')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();prepare(a.out)
