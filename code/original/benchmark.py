"""Controlled small-data benchmark. See README before interpreting outputs."""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASETS = ['iris', 'wine', 'breast_cancer', 'sonar', 'ionosphere', 'diabetes']
MODELS = ['LogisticRegression', 'RandomForest', 'TabPFN']
FRACTIONS = [.10, .25, .50, .75, 1.0]
SEEDS = [11, 22, 33, 44, 55]
OPENML = {'sonar': (40, 'sonar', (208, 60)),
          'ionosphere': (59, 'ionosphere', (351, 34)),
          'diabetes': (37, 'diabetes', (768, 8))}


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def versions():
    out = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tabpfn']:
        try: out[name] = metadata.version(name)
        except metadata.PackageNotFoundError: out[name] = 'not installed'
    return out


def get_data(name, cache):
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f'{name}.npz'
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return z['X'], z['y'], json.loads(str(z['info']))
    local = {'iris': load_iris, 'wine': load_wine, 'breast_cancer': load_breast_cancer}
    if name in local:
        b = local[name]()
        X, raw_y = b.data, b.target
        source = f'sklearn.datasets.load_{name}'
        source_name = name
    else:
        data_id, expected_name, shape = OPENML[name]
        b = fetch_openml(data_id=data_id, as_frame=True)
        source_name = b.details['name']
        if source_name.lower() != expected_name or b.data.shape != shape:
            raise ValueError(f'Unexpected OpenML dataset: {source_name}, {b.data.shape}')
        X = b.data.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        raw_y = b.target
        if pd.isna(raw_y).any(): raise ValueError('Missing targets; inspect dataset.')
        source = f'https://www.openml.org/d/{data_id}'
    X = np.asarray(X, dtype=np.float64)
    if np.isinf(X).any(): raise ValueError('Infinite feature values.')
    y = LabelEncoder().fit_transform(raw_y).astype(np.int64)
    # Pima zeros in these clinical measurements encode missingness, not valid zero values.
    if name == 'diabetes':
        for j in [1, 2, 3, 4, 5]: X[X[:, j] == 0, j] = np.nan
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    info = {'name': name, 'source': source, 'source_name': source_name,
            'rows': len(y), 'features': X.shape[1], 'classes': len(np.unique(y)),
            'array_sha256': digest, 'missing_entries': int(np.isnan(X).sum())}
    np.savez_compressed(p, X=X, y=y, info=json.dumps(info))
    return X, y, info


def nested_order(y, seed):
    """Class-balanced prefixes, with random within-class order; no feature/test access."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    queues = [rng.permutation(np.flatnonzero(y == c)).tolist() for c in classes]
    proportions = counts / len(y)
    used = np.zeros(len(classes), dtype=int)
    order = []
    # Every class appears at the beginning, so all training fractions include all labels.
    for j in rng.permutation(len(classes)):
        order.append(queues[j][0]); used[j] += 1
    while len(order) < len(y):
        deficit = (len(order) + 1) * proportions - used
        deficit[used == counts] = -np.inf
        j = int(np.argmax(deficit))
        order.append(queues[j][used[j]]); used[j] += 1
    return np.asarray(order)


def make_model(name, seed, device):
    if name == 'LogisticRegression':
        model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             StandardScaler(), model)
    if name == 'RandomForest':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                                    random_state=seed, n_jobs=-1))
    # Pin the model family explicitly; do not rely on a changing package default.
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device=device, n_estimators=4, random_state=seed)


def sync_gpu(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()


def summarize(out, datasets, seeds, models):
    rows = [json.loads(p.read_text()) for p in sorted((out / 'records').glob('*.json'))]
    if not rows: return
    d = pd.DataFrame(rows)
    d.to_csv(out / 'raw_results.csv', index=False)
    expected = len(datasets) * len(seeds) * len(FRACTIONS) * len(models)
    status = {'completed': len(d), 'expected': expected, 'complete': len(d) == expected}
    atomic_json(out / 'completion.json', status)
    mean = d.groupby(['dataset', 'model', 'fraction']).agg(
        balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        balanced_accuracy_seed_sd=('balanced_accuracy', 'std'),
        accuracy_mean=('accuracy', 'mean'), n_seeds=('seed', 'nunique')).reset_index()
    mean.to_csv(out / 'learning_curve_summary.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for model, g in mean[mean.dataset == dataset].groupby('model'):
            g = g.sort_values('fraction')
            ax.plot(g.fraction * 100, g.balanced_accuracy_mean * 100, marker='o', label=model)
        ax.set(title=dataset, xlabel='Training pool used (%)', ylabel='Balanced accuracy (%)')
        ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7)
    fig.suptitle('Controlled learning curves' + ('' if status['complete'] else ' (INCOMPLETE / PILOT)'))
    fig.savefig(out / 'learning_curves.png', dpi=200); plt.close(fig)
    # Paired comparisons: same dataset, seed, fraction and test examples.
    wide = d.pivot(index=['dataset', 'seed', 'fraction'], columns='model', values='balanced_accuracy')
    gaps = []
    if 'TabPFN' in wide:
        for baseline in ['LogisticRegression', 'RandomForest']:
            if baseline not in wide: continue
            pair = wide[['TabPFN', baseline]].dropna().reset_index()
            pair['gap_pp'] = 100 * (pair.TabPFN - pair[baseline])
            pair['baseline'] = baseline
            gaps.append(pair)
    if gaps:
        paired = pd.concat(gaps, ignore_index=True)
        paired.to_csv(out / 'paired_gaps.csv', index=False)
        per_dataset = paired.groupby(['dataset', 'fraction', 'baseline']).agg(
            mean_gap_pp=('gap_pp', 'mean'), n_seeds=('seed', 'nunique')).reset_index()
        per_dataset.to_csv(out / 'dataset_mean_gaps.csv', index=False)
        # Only complete results enter the exploratory across-dataset bootstrap.
        if status['complete']:
            rng = np.random.default_rng(2026)
            overall = []
            for (fraction, baseline), g in per_dataset.groupby(['fraction', 'baseline']):
                values = g.mean_gap_pp.to_numpy()
                boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
                overall.append({'fraction': fraction, 'baseline': baseline,
                    'datasets': len(values), 'equal_dataset_mean_gap_pp': values.mean(),
                    'bootstrap_low_pp': np.quantile(boot, .025),
                    'bootstrap_high_pp': np.quantile(boot, .975)})
            pd.DataFrame(overall).to_csv(out / 'exploratory_across_dataset_gaps.csv', index=False)
            # Descriptive matching threshold; NOT a statistical equivalence test.
            matches = []
            for dataset in datasets:
                sub = mean[mean.dataset == dataset]
                tf = sub[sub.model == 'TabPFN'].sort_values('fraction')
                for baseline in ['LogisticRegression', 'RandomForest']:
                    base = sub[(sub.model == baseline) & (sub.fraction == 1)]
                    if base.empty: continue
                    score = float(base.balanced_accuracy_mean.iloc[0])
                    good = tf[tf.balanced_accuracy_mean >= score - .01]
                    matches.append({'dataset': dataset, 'baseline': baseline,
                        'baseline_full_balanced_accuracy': score,
                        'smallest_observed_matching_fraction': None if good.empty else float(good.fraction.iloc[0]),
                        'definition': 'TabPFN mean >= full-pool baseline mean minus 1 percentage point'})
            pd.DataFrame(matches).to_csv(out / 'descriptive_data_efficiency.csv', index=False)
    (out / 'READ_RESULTS_FIRST.txt').write_text(
        f'Completed {len(d)}/{expected} evaluations. Complete={status["complete"]}.\n'
        'Do not use incomplete/pilot outputs as the full benchmark.\n'
        'Metrics are proportions unless a column ends in _pp (percentage points).\n'
        'Seed SD describes repetition variability; it is not a confidence interval.\n'
        'Across-dataset bootstrap intervals are exploratory: six convenience-selected tasks are few,\n'
        'not a representative sample. Seeds share data and are not independent datasets.\n'
        'The 1-point matching threshold is descriptive, not proof of equivalence.\n'
        'No hyperparameter tuning; conclusions apply to these configurations.\n'
        'Classic datasets may overlap foundation-model development/evaluation; audit model report.\n'
        'Runtime includes preprocessing/inference but excludes initial checkpoint loading/download;\n'
        'RF uses CPU, TabPFN may use GPU, so runtime is not a hardware-controlled comparison.\n')


def run(out='benchmark_results', pilot=True, device='auto', baselines_only=False):
    out = Path(out)
    datasets = ['wine'] if pilot else DATASETS
    seeds = [11] if pilot else SEEDS
    models = MODELS[:2] if baselines_only else MODELS
    if not baselines_only:
        import torch
        if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Enable a Colab GPU or explicitly use CPU.')
        print('TabPFN device:', device, flush=True)
        # Load the checkpoint before timing or any baseline runs: fail early on access problems.
        probe = make_model('TabPFN', seeds[0], device)
        del probe; gc.collect()
    config = {'protocol': 'v1', 'datasets': datasets, 'seeds': seeds,
              'fractions': FRACTIONS, 'models': models, 'device': device,
              'checkpoint': 'TabPFN V2', 'test_fraction': .2,
              'versions': versions(), 'pilot': pilot, 'baselines_only': baselines_only,
              'rf_trees': 300, 'lr_C': 1.0, 'tabpfn_estimators': 4}
    source_path = Path(__file__) if '__file__' in globals() else None
    if source_path and source_path.is_file():
        config['script_sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Configuration/version changed. Use a NEW output folder; do not mix runs.')
    atomic_json(manifest, config)
    for folder in ['records', 'splits', 'predictions', 'data_cache']:
        (out / folder).mkdir(exist_ok=True)
    infos = []
    try:
        for dataset in datasets:
            X, y, info = get_data(dataset, out / 'data_cache')
            infos.append(info); atomic_json(out / 'datasets.json', infos)
            for seed in seeds:
                pool, test = train_test_split(np.arange(len(y)), test_size=.2,
                                             stratify=y, random_state=seed)
                order = pool[nested_order(y[pool], seed)]
                np.savez_compressed(out / 'splits' / f'{dataset}_{seed}.npz',
                                    ordered_training_pool=order, test=test)
                assert not set(order).intersection(test)
                for fraction in FRACTIONS:
                    n = max(len(np.unique(y)), int(np.floor(len(order) * fraction)))
                    train = order[:n]
                    for name in models:
                        key = f'{dataset}_{seed}_{fraction:.2f}_{name}'
                        record = out / 'records' / f'{key}.json'
                        if record.exists(): continue
                        print(f'{dataset} seed={seed} fraction={fraction} n={n} {name}', flush=True)
                        model = make_model(name, seed, device)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            sync_gpu(device)
                            start = time.perf_counter()
                            model.fit(X[train], y[train])
                            pred = model.predict(X[test])
                            sync_gpu(device)
                            elapsed = time.perf_counter() - start
                        pd.DataFrame({'row_index': test, 'true': y[test], 'predicted': pred}).to_csv(
                            out / 'predictions' / f'{key}.csv', index=False)
                        atomic_json(record, {'dataset': dataset, 'seed': seed, 'fraction': fraction,
                            'model': name, 'n_train': n, 'n_test': len(test),
                            'accuracy': float(accuracy_score(y[test], pred)),
                            'balanced_accuracy': float(balanced_accuracy_score(y[test], pred)),
                            'fit_predict_seconds': elapsed,
                            'warnings': [str(w.message) for w in caught]})
                        del model; gc.collect()
                        if device == 'cuda':
                            import torch
                            torch.cuda.empty_cache()
    finally:
        summarize(out, datasets, seeds, models)
    print('Finished. Read', out / 'READ_RESULTS_FIRST.txt', flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--full', action='store_true', help='Full 450 evaluations; default is pilot (15).')
    p.add_argument('--out', default='pilot_results')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--baselines-only', action='store_true', help='Setup testing only; not the proposed study.')
    a = p.parse_args()
    run(a.out, pilot=not a.full, device=a.device, baselines_only=a.baselines_only)
