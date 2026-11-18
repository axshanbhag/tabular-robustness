"""One process per run. Output directory must be new: timing runs never resume."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,json,hashlib,uuid,datetime,time,platform,subprocess,random,gc,importlib.metadata as md,sys,warnings
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'original'))

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def hardware():
 def cmd(args):
  try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
  except Exception as e:return type(e).__name__
 return {'platform':platform.platform(),'cpu':cmd(['lscpu']),'gpu':cmd(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),'threads':{k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
def model(name,seed):
 from sklearn.impute import SimpleImputer
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 if name=='LogisticRegression':
  from sklearn.linear_model import LogisticRegression
  return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=5000,random_state=seed))
 if name=='RandomForest':
  from sklearn.ensemble import RandomForestClassifier
  return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(n_estimators=300,max_features='sqrt',random_state=seed,n_jobs=2))
 if name=='CatBoost':
  from catboost import CatBoostClassifier
  return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),CatBoostClassifier(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3,random_seed=seed,thread_count=2,task_type='CPU',verbose=False,allow_writing_files=False))
 if name=='TabPFN':
  from tabpfn import TabPFNClassifier
  from tabpfn.constants import ModelVersion
  return TabPFNClassifier.create_default_for_version(ModelVersion.V2,n_estimators=4,random_state=seed,device='cuda')
 raise ValueError(name)
def load_data(cohort,key,seed,prepared):
 if cohort=='original':
  z=np.load(ROOT/'frozen/data_cache'/f'{key}.npz');sp=np.load(ROOT/'frozen/splits'/f'{key}_{seed}.npz');noise=np.load(ROOT/'frozen/noise_arrays'/f'{key}_{seed}.npz')['noise']
 else:
  b=Path(prepared)/key;z=np.load(b/'data.npz');sp=np.load(b/f'split_{seed}.npz');noise=np.load(b/f'noise_{seed}.npz')['noise']
 return z['X'],z['y'],sp['ordered_training_pool'],sp['test'],noise

def run(a):
 from sklearn.metrics import balanced_accuracy_score
 from threadpoolctl import threadpool_limits,threadpool_info
 cfg=json.loads((ROOT/'protocol.json').read_text());out=Path(a.out)
 if out.resolve().is_relative_to(ROOT):raise ValueError('Keep outputs outside the source package')
 out.mkdir(parents=True,exist_ok=False) # No resume, including after failures.
 (out/'predictions').mkdir();session=uuid.uuid4().hex
 if a.cohort=='new':
  freeze=json.loads((Path(a.prepared)/'freeze.json').read_text());assert freeze['protocol_sha256']==sha(ROOT/'protocol.json')
  for p,h in freeze['files'].items():assert sha(Path(a.prepared)/p)==h,p
  datasets=[x['key'] for x in json.loads((Path(a.prepared)/'selection.json').read_text())['selected']]
 else:datasets=sorted(p.stem for p in (ROOT/'frozen/data_cache').glob('*.npz'))
 models=cfg['models'];seeds=cfg['seeds'];fracs=cfg['fractions'];reps=cfg['timing_repeats'] if a.mode=='timing' else 1
 if a.mode=='pilot':datasets=datasets[:1];seeds=seeds[:1];fracs=[.5];reps=2
 if a.mode=='smoke':models=['LogisticRegression','RandomForest','CatBoost'];datasets=datasets[:1];seeds=seeds[:1];fracs=[.5];reps=1
 torch=None
 if 'TabPFN' in models:
  import torch
  if not torch.cuda.is_available():raise RuntimeError('CUDA required. Use a Colab GPU runtime.')
  torch.set_num_threads(1)
 blocks=[(d,s,f,m) for d in datasets for s in seeds for f in fracs for m in models]
 random.Random(cfg['selection_seed']).shuffle(blocks)
 manifest={'session_id':session,'started':now(),'cohort':a.cohort,'mode':a.mode,'datasets':datasets,'models':models,'seeds':seeds,'fractions':fracs,'ratios':cfg['ratios'],'repeats':reps,'expected':len(blocks)*3*reps,'versions':{x:md.version(x) for x in ['numpy','scipy','scikit-learn','threadpoolctl']},'python':platform.python_version(),'hardware':hardware(),'pid':os.getpid(),'process_started_monotonic':time.monotonic(),'protocol':cfg,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.npz']},'block_order':blocks}
 for name in ['torch','tabpfn','catboost']:
  try:manifest['versions'][name]=md.version(name)
  except md.PackageNotFoundError:pass
 save(out/'manifest.json',manifest)
 (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 configs={name:{k:repr(v) for k,v in model(name,11).get_params(deep=True).items()} for name in models}
 save(out/'estimator_parameters.json',configs)
 # Preflight checkpoint download/loading happens outside the measured run.
 if torch is not None:
  probe=model('TabPFN',11);probe.fit(np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.]]),np.array([0,1,0,1]));probe.predict(np.array([[.5,.5]]))
  cache=Path(os.environ.get('TABPFN_MODEL_CACHE_DIR',str(Path.home()/'.cache/tabpfn')))
  weights={str(p):sha(p) for p in cache.rglob('*') if p.is_file() and p.suffix in ['.ckpt','.pt','.pth','.safetensors']}
  if not weights:raise RuntimeError('No checkpoint files found in TABPFN_MODEL_CACHE_DIR; set it before importing tabpfn and use a new output directory.')
  save(out/'checkpoint_hashes.json',weights);del probe;gc.collect()
 completed=0;failed=False
 try:
  with threadpool_limits(limits=2,user_api="blas"):
   save(out/'threadpools.json',{'pools':threadpool_info(),'torch_intraop_threads':torch.get_num_threads() if torch is not None else None,'torch_interop_threads':torch.get_num_interop_threads() if torch is not None else None})
   for bi,(d,seed,f,m) in enumerate(blocks):
    X,y,order,test,noise=load_data(a.cohort,d,seed,a.prepared);n=max(len(np.unique(y)),int(len(order)*f));train=order[:n]
    assert not set(train)&set(test)
    assert set(y[train])==set(y)
    arrays={q:(X if q==0 else np.column_stack([X,noise[:,:max(1,int(np.floor(X.shape[1]*q+.5)))]])) for q in cfg['ratios']}
    def evaluate(q,rep,measured):
     A=arrays[q];xtr,xte=A[train],A[test]
     gc.collect()
     if torch is not None:torch.cuda.empty_cache();torch.cuda.synchronize()
     def clock(fn):
      if torch is not None:torch.cuda.synchronize()
      t=time.perf_counter();v=fn()
      if torch is not None:torch.cuda.synchronize()
      return v,time.perf_counter()-t
     start=now()
     with warnings.catch_warnings(record=True) as ws:
      warnings.simplefilter('always');est,construct=clock(lambda:model(m,seed));_,fit=clock(lambda:est.fit(xtr,y[train]));pred,predict=clock(lambda:np.asarray(est.predict(xte)).reshape(-1));pred2,predict2=clock(lambda:np.asarray(est.predict(xte)).reshape(-1))
     r={'session_id':session,'block':bi,'dataset':d,'seed':seed,'fraction':f,'model':m,'noise_ratio':q,'repeat':rep,'start_utc':start,'monotonic_end':time.monotonic(),'end_utc':now(),'n_train':n,'n_test':len(test),'construct_seconds':construct,'fit_seconds':fit,'first_predict_seconds':predict,'repeat_predict_seconds':predict2,'fit_first_predict_seconds':fit+predict,'balanced_accuracy':float(balanced_accuracy_score(y[test],pred)),'warnings':[str(x.message) for x in ws]}
     if measured:
      stem=f'{d}_{seed}_{f}_{m}_{q}_{rep}';np.savez_compressed(out/'predictions'/f'{stem}.npz',row_index=test,true=y[test],predicted=pred,second_prediction=pred2)
      with (out/'records.jsonl').open('a') as h:h.write(json.dumps(r,allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
     del est
    if a.mode in ['timing','pilot']:
     for q in cfg['ratios']:evaluate(q,-1,False)
    for rep in range(reps):
     qs=cfg['ratios'].copy();random.Random(f'{cfg["selection_seed"]}:{bi}:{rep}').shuffle(qs)
     for q in qs:evaluate(q,rep,True);completed+=1
    print(f'{bi+1}/{len(blocks)} blocks; {completed} measurements',flush=True)
 except BaseException as e:
  failed=True;save(out/'failure.json',{'type':type(e).__name__,'message':str(e),'time':now()});raise
 finally:
  save(out/'completion.json',{'complete':not failed and completed==manifest['expected'],'completed':completed,'expected':manifest['expected'],'session_id':session,'ended':now(),'hardware_end':hardware()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','pilot','accuracy','timing'],required=True);p.add_argument('--cohort',choices=['original','new'],default='original');p.add_argument('--prepared');p.add_argument('--out',required=True);a=p.parse_args()
 if a.cohort=='new' and not a.prepared:p.error('--prepared required for new cohort')
 run(a)
