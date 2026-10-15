from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

from .common import (ROOT, Blocked, array_hash, digest, immutable_json, project_path,
                     read_json, sha256, write_json)

LIMITATIONS = {
    'positioning': 'TabReD-derived temporal benchmark with inherited preprocessing limitations',
    'inherited_preprocessing': 'inherited/transductive preprocessing limitation',
    'upstream_full_data_statistics': ['missing rate >95% column deletion', 'constant and string-cardinality >50 column deletion', 'OrdinalEncoder(min_frequency=0.01) fitted before split and subsampling', 'min/max dtype downcasting'],
    'upstream_target_leakage': 'No explicit target-in-feature or target encoding found in pinned source; raw-table as-of availability not independently verified.',
    'artifact_generation_lineage': 'not independently verified; source legacy names differ from current archive documentation; documented limitations conservatively inherited',
    'rare_category_recovery': 'A-only re-encoding cannot undo upstream rare-category merging',
    'target': 'Dataset target event (source describes loan default); exact horizon NOT VERIFIED',
    'probability_name': 'dataset target-event probability / PD proxy; NOT one-year PD',
    'label_maturity': 'NOT VERIFIED; not a tradeable historical backtest',
    'entity_id': 'case_id absent in documented artifact; index_in_full is source row index, not customer ID; customer-level independence NOT VERIFIED',
    'ead_lgd_cashflows': 'No verified EAD, realized recovery, default time or contractual cashflows; amount fields are not automatically EAD',
    'portfolio': 'credit-event loss proxy = observed event label times assumed LGD; EAD=1 normalized exposure unit',
    'calendar_features': ['day_of_week', 'day_of_month', 'day_of_year'],
    'public_row_level_outputs': False,
}


def find_file(folder, options, required=True):
    for name in options:
        if (Path(folder) / name).is_file():
            return Path(folder) / name
    if required:
        raise Blocked(f'Missing dataset file: one of {options}')
    return None


def acquire(cfg, run_dir):
    folder = project_path(cfg['data_dir'])
    if (folder / 'info.json').is_file():
        return folder
    receipt = Path(run_dir) / 'download_status.json'
    # Presence only: never read or log tokens.
    config_dir = Path(os.environ.get('KAGGLE_CONFIG_DIR', Path.home() / '.kaggle'))
    credential_present = any(os.environ.get(k) for k in ['KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY']) or any((config_dir / f).exists() for f in ['kaggle.json', 'access_token', 'oauth.json', 'credentials.json'])
    # One bounded downloader call per explicit run invocation. A prior presence
    # check cannot prove that OAuth, refreshed credentials or access stayed unchanged.
    reason = 'BLOCKED_AUTH: official downloader requires Kaggle authentication / existing competition access; data absent.'
    if cfg['download']:
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        if not config_dir.exists():
            env['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
        executable = Path(sys.executable).parent / ('tabred.exe' if os.name == 'nt' else 'tabred')
        command = [str(executable), 'download', 'homecredit-default', '--output-path', str(folder.parent), '--max-workers', '1']
        log = Path(run_dir) / 'data_download.log'
        with log.open('w', encoding='utf-8') as output:
            try:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env, timeout=600, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code, reason = None, 'BLOCKED_NETWORK: official download timed out after 600 seconds; no unbounded retries.'
        if (folder / 'info.json').is_file():
            write_json(receipt, {'status': 'FILES_PRESENT_PENDING_AUDIT', 'command': command, 'exit_code': code, 'credential_present': bool(credential_present)})
            return folder
        log_text = log.read_text(encoding='utf-8', errors='replace')
        if 'Authentication required' not in log_text and 'accept the rules' not in log_text and code is not None:
            reason = f'BLOCKED_DOWNLOAD: official downloader exit={code}; no dataset files. See data_download.log.'
        write_json(receipt, {'status':'BLOCKED', 'reason':reason, 'command':command, 'exit_code':code, 'credential_present':bool(credential_present), 'success_requires_files':True})
    raise Blocked(reason)


def split_four(dates, train, validation, test):
    dates = np.asarray(dates)
    unique = np.unique(dates[validation])
    if len(unique) < 2:
        raise Blocked('Official validation has fewer than two complete dates.')
    boundary = unique[len(unique) // 2]
    splits = {'A':np.asarray(train, dtype=np.int64),
              'B':np.asarray(validation[dates[validation] < boundary], dtype=np.int64),
              'C':np.asarray(validation[dates[validation] >= boundary], dtype=np.int64),
              'D':np.asarray(test, dtype=np.int64)}
    all_ids = np.concatenate(list(splits.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise Blocked('Duplicate or overlapping split row indices.')
    if any(len(x) == 0 for x in splits.values()):
        raise Blocked('Empty temporal partition.')
    for left, right in zip('ABC', 'BCD'):
        if dates[splits[left]].max() >= dates[splits[right]].min():
            raise Blocked(f'Temporal order violation: {left} is not strictly before {right}; no random replacement.')
    return splits


@dataclass
class DevelopmentData:
    num_A: np.ndarray
    cat_A: np.ndarray
    y_A: np.ndarray
    num_B: np.ndarray
    cat_B: np.ndarray
    y_B: np.ndarray


class Dataset:
    """Only explicit A/B/C label access; D requires a frozen access receipt."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = read_json(self.folder / 'info.json')
        # The official 0.1.2 archive has only {"task":{"type":"binclass",
        # "score":"roc-auc"}}. Its identity is provided by the named directory
        # extracted by the official downloader after its registry hash check.
        dataset_name = self.info.get('name', self.folder.name)
        task_type = self.info.get('task_type', self.info.get('task', {}).get('type'))
        if dataset_name!='homecredit-default' or task_type not in ['binclass','binary','binary_classification']:
            raise Blocked('Artifact is not identified as homecredit-default binary task; source audit required.')
        self.paths = {'info':self.folder / 'info.json'}
        for key in ['num', 'cat', 'bin', 'meta']:
            p = find_file(folder, [f'x_{key}.npy', f'X_{key}.npy'], required=key in ['num','meta'])
            if p:
                self.paths[key] = p
        self.paths['y'] = find_file(folder, ['y.npy', 'Y.npy'])
        self.arrays = {k:np.load(p, mmap_mode='r', allow_pickle=False) for k,p in self.paths.items() if k != 'info'}
        self.n = len(self.arrays['num'])
        if any(len(a) != self.n for a in self.arrays.values()):
            raise Blocked('Array row counts disagree.')
        self.meta = self.arrays['meta']
        if self.meta.ndim != 2 or self.meta.shape[1] != 2:
            raise Blocked('x_meta schema differs from pinned [date_decision,index_in_full]; manual source audit required.')
        raw_dates = self.meta[:, 0]
        if not np.issubdtype(raw_dates.dtype, np.integer):
            raise Blocked('Date storage differs from pinned Polars Date epoch-day encoding.')
        self.dates = pd.to_datetime(raw_dates, unit='D', origin='unix').to_numpy(dtype='datetime64[D]')
        if np.isnat(self.dates).any():
            raise Blocked('Missing decision dates.')
        self.row_id = np.arange(self.n, dtype=np.int64)
        official = {}
        for part in ['train','val','test']:
            p = find_file(folder, [f'splits/default/{part}.npy', f'split-default/{part}_idx.npy', f'splits/default/{part}_idx.npy'])
            self.paths[part] = p
            idx = np.load(p, allow_pickle=False)
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(idx)==0 or idx.min()<0 or idx.max()>=self.n:
                raise Blocked('Invalid official split indices.')
            official[part] = idx
        self.splits = split_four(self.dates, official['train'], official['val'], official['test'])

    def features(self, ids):
        # Whitelist feature matrices. Never concatenate x_meta, row IDs or labels.
        num = [np.asarray(self.arrays['num'][ids], dtype=np.float32)]
        if 'bin' in self.arrays:
            num.append(np.asarray(self.arrays['bin'][ids], dtype=np.float32))
        cat = np.asarray(self.arrays['cat'][ids]) if 'cat' in self.arrays else np.empty((len(ids),0),dtype=np.int32)
        return np.concatenate(num,axis=1), cat

    def labels(self, part):
        if part not in ['A','B','C']:
            raise Blocked('D labels available only through final_labels after freeze.')
        values=np.asarray(self.arrays['y'][self.splits[part]]).reshape(-1)
        if not np.isin(values,[0,1]).all():
            raise Blocked(f'Nonbinary or invalid labels in {part}')
        return values.astype(np.int64)

    def final_labels(self, run_dir):
        from .common import verify_freeze
        run_dir = Path(run_dir)
        verify_freeze(run_dir / 'FROZEN_MANIFEST.json')
        receipt = read_json(run_dir / 'TEST_ACCESS_STARTED.json')
        if receipt['manifest_sha256'] != sha256(run_dir / 'FROZEN_MANIFEST.json'):
            raise Blocked('Final-label receipt belongs to a different freeze.')
        values=np.asarray(self.arrays['y'][self.splits['D']]).reshape(-1)
        if not np.isin(values,[0,1]).all():
            raise Blocked('Nonbinary or invalid labels in D; seen-test failure retained')
        return values.astype(np.int64)

    def audit(self, run_dir):
        run_dir = Path(run_dir)
        split_meta = {}
        for key, ids in self.splits.items():
            path = run_dir / 'splits' / (key + '.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if not np.array_equal(np.load(path), ids):
                    raise Blocked('Saved split changed.')
            else:
                np.save(path, ids)
            split_meta[key] = {'n':len(ids),'start':str(self.dates[ids].min()),'end':str(self.dates[ids].max()),'unique_dates':len(np.unique(self.dates[ids])),'index_sha256':sha256(path)}
        # Duplicate-feature analysis is label-free and bounded to one matrix at a time.
        row_hash = np.zeros(self.n,dtype=np.uint64)
        for key in ['num','cat','bin']:
            if key in self.arrays:
                row_hash ^= pd.util.hash_pandas_object(pd.DataFrame(self.arrays[key]),index=False).to_numpy()
        duplicates = self.n - len(np.unique(row_hash))
        overlap = {a+b:int(len(np.intersect1d(row_hash[self.splits[a]],row_hash[self.splits[b]]))) for a,b in [('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]}
        feature_names={}
        for key in ['num','cat','bin']:
            if key not in self.arrays:
                continue
            header=find_file(self.folder,[f'csv/x_{key}.csv',f'csv/X_{key}.csv'],required=False)
            names=list(pd.read_csv(header,nrows=0).columns) if header else []
            if len(names)!=self.arrays[key].shape[1]:
                names=[f'{key}_{i:04d}' for i in range(self.arrays[key].shape[1])]
            if any(name.lower() in ['target','y','case_id','date_decision','index_in_full','row_id'] for name in names):
                raise Blocked('Forbidden target/identifier appears in model feature header')
            feature_names[key]={'names':names,'source':'artifact CSV header only' if header else 'anonymous column indices'}
        immutable_json(run_dir/'feature_manifest.json',feature_names)
        audit = {'status':'DATA_AUDITED', **LIMITATIONS,
                 'sample_unit':'loan application/case in source; artifact row in this benchmark',
                 'dimensions':{k:list(a.shape) for k,a in self.arrays.items()},
                 'n_rows':self.n,'feature_count':sum(self.arrays[k].shape[1] for k in ['num','cat','bin'] if k in self.arrays),
                 'files':{str(p):sha256(p) for p in self.paths.values()}, 'info':self.info,
                 'meta_fields':['date_decision','index_in_full'], 'date_encoding':'integer days since 1970-01-01 (pinned upstream Polars Date cast)',
                 'raw_full_source_time_range':'NOT VERIFIED; artifact range reported separately',
                 'artifact_time_range':[str(self.dates.min()),str(self.dates.max())],
                 'splits':split_meta, 'split_intersections':0,
                 'duplicate_source_row_ids':self.n-len(np.unique(self.meta[:,1])),
                 'duplicate_feature_rows_hash_based':duplicates,'cross_split_duplicate_feature_hashes':overlap,
                 'feature_names':feature_names,
                 'D_label_distribution':'SEALED_NOT_COMPUTED', 'adaptation':'Official train=A; whole validation dates split at unique-date midpoint into B/C; official test unchanged.'}
        immutable_json(run_dir / 'data_audit.json',audit)
        immutable_json(run_dir / 'split_manifest.json',split_meta)
        return audit


class Preprocessor:
    """Common A-only information set: finite median imputation, constants, vocab.

    Same numerical information for both models. TabM receives A-only standard
    scaling; LightGBM can use the same scale (tree ordering is preserved).
    Unknown=0, missing=1; observed A categories start at 2.
    """
    @staticmethod
    def _key(value):
        if pd.isna(value):
            return None
        return str(value)

    def fit(self, num, cat):
        x = np.asarray(num, dtype=np.float64).copy()
        x[~np.isfinite(x)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            median = np.nanmedian(x,axis=0)
        self.median = np.nan_to_num(median,nan=0.0)
        x = np.where(np.isnan(x),self.median,x)
        self.keep = np.ptp(x,axis=0)>0
        if not self.keep.any():
            raise Blocked('No nonconstant numeric features after A-only preprocessing.')
        x = x[:,self.keep]
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale==0] = 1
        self.vocabs = []
        for j in range(cat.shape[1]):
            values = sorted({self._key(v) for v in cat[:,j] if self._key(v) is not None})
            self.vocabs.append({v:i+2 for i,v in enumerate(values)})
        self.cardinalities = [len(v)+2 for v in self.vocabs]
        self.fit_hash = digest({'num':array_hash(num),'cat':array_hash(cat)})
        self.input_dimensions = {'num_before':num.shape[1], 'num_after':int(self.keep.sum()),'cat':cat.shape[1], 'tabm_embedded_width':int(self.keep.sum())*16+sum(self.cardinalities)}
        return self

    def transform(self,num,cat):
        x = np.asarray(num,dtype=np.float64).copy()
        x = np.where(np.isfinite(x),x,self.median)
        x = ((x[:,self.keep]-self.mean)/self.scale).astype(np.float32)
        if not np.isfinite(x).all():
            raise Blocked('Nonfinite numeric transform; no silent clipping.')
        c = np.empty(cat.shape,dtype=np.int64)
        for j,vocab in enumerate(self.vocabs):
            c[:,j] = [1 if self._key(v) is None else vocab.get(self._key(v),0) for v in cat[:,j]]
        return x,c


def development(dataset, preprocessor):
    a_num,a_cat = dataset.features(dataset.splits['A'])
    b_num,b_cat = dataset.features(dataset.splits['B'])
    a_num,a_cat = preprocessor.transform(a_num,a_cat)
    b_num,b_cat = preprocessor.transform(b_num,b_cat)
    return DevelopmentData(a_num,a_cat,dataset.labels('A'),b_num,b_cat,dataset.labels('B'))
