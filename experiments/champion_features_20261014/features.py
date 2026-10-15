"""Execute the author's feature expressions on aligned A/B application records.

Public classes/functions are extracted unchanged from pinned source via AST.
Top-level model loading/submission code is never executed. Downcasting based on
all rows and full-data column selection are replaced by explicit A-only fitting.
"""
from __future__ import annotations
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import ast
import gc
import time
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import psutil
from common import HERE, PROJECT, ORIGINAL, RAW, DERIVED, read, write, sha, now, protected_hashes

LGB_GROUPS = [
    ('static_cb', 'train_static_cb_0.parquet', 0), ('static', 'train_static_0_*.parquet', 0),
    ('applprev', 'train_applprev_1_*.parquet', 1), ('bureau_a1', 'train_credit_bureau_a_1_*.parquet', 1),
    ('bureau_b1', 'train_credit_bureau_b_1.parquet', 1), ('deposit', 'train_deposit_1.parquet', 1),
    ('debitcard', 'train_debitcard_1.parquet', 1), ('tax_a', 'train_tax_registry_a_1.parquet', 1),
    ('tax_b', 'train_tax_registry_b_1.parquet', 1), ('tax_c', 'train_tax_registry_c_1.parquet', 1),
    ('person', 'train_person_1.parquet', 1), ('other', 'train_other_1.parquet', 1),
    ('bureau_a2', 'train_credit_bureau_a_2_*.parquet', 2), ('bureau_b2', 'train_credit_bureau_b_2.parquet', 2),
    ('person2', 'train_person_2.parquet', 2),
]
CAT_GROUPS = [
    LGB_GROUPS[0], LGB_GROUPS[1], LGB_GROUPS[2], LGB_GROUPS[7], LGB_GROUPS[8], LGB_GROUPS[9],
    LGB_GROUPS[3], LGB_GROUPS[4], LGB_GROUPS[11], LGB_GROUPS[10], LGB_GROUPS[5], LGB_GROUPS[6],
    LGB_GROUPS[13], LGB_GROUPS[12], ('applprev2', 'train_applprev_2.parquet', 2), LGB_GROUPS[14],
]
EXCLUDED = {'case_id', 'target', 'WEEK_NUM', 'MONTH', 'date_decision', 'row_id', 'index_in_full', 'split'}


def load_author(variant):
    manifest = read(HERE / 'SOURCE_MANIFEST.json')
    record = next(x for x in manifest['kernels'] if Path(x['file']).parts[1] == variant)
    path = HERE / record['file']
    assert sha(path) == record['sha256']
    wanted = {'Pipeline', 'Aggregator'} | ({'agg_by_case'} if variant == 'lgb' else set())
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = {'pl': pl, 'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def aggregate(frame, variant, filename, depth):
    ns = load_author(variant)
    assert 'target' not in frame.columns
    frame = ns['Pipeline'].set_table_dtypes(frame)
    if depth:
        if variant == 'lgb':
            frame = ns['agg_by_case'](filename, frame)
        else:
            if depth == 1:
                frame = frame.sort('num_group1')
            frame = frame.group_by('case_id').agg(ns['Aggregator'].get_exprs(frame))
    return frame


def make_mapping():
    path = DERIVED / 'mapping_AB.parquet'
    if path.exists():
        assert sha(path) == read(DERIVED / 'mapping_receipt.json')['mapping_sha256']
        return pl.read_parquet(path)
    a, b = [np.load(ORIGINAL / f'splits/{s}.npy') for s in ('A', 'B')]
    rows = np.r_[a, b]
    meta = np.load(PROJECT / 'data/homecredit-default/x_meta.npy', mmap_mode='r')[rows]
    base = pl.read_parquet(RAW / 'train_base.parquet', columns=['case_id', 'date_decision', 'WEEK_NUM', 'MONTH'])
    mapped = base[meta[:, 1].astype(int)].with_columns(pl.col('date_decision').cast(pl.Date))
    assert np.array_equal(mapped['date_decision'].cast(pl.Int32).to_numpy(), meta[:, 0])
    assert mapped['case_id'].n_unique() == len(rows)
    # Only mapped development records' target values enter memory; C/D excluded by filter.
    ry = pl.scan_parquet(RAW / 'train_base.parquet').filter(pl.col('case_id').is_in(mapped['case_id'])).select(['case_id', 'target']).collect()
    y = mapped.select('case_id').join(ry, on='case_id', how='left')['target'].to_numpy()
    assert np.array_equal(y, np.load(PROJECT / 'data/homecredit-default/y.npy', mmap_mode='r')[rows])
    mapped = mapped.with_columns(pl.Series('row_id', rows), pl.Series('split', ['A']*len(a)+['B']*len(b)))
    mapped.write_parquet(path)
    np.save(DERIVED / 'y_A.npy', y[:len(a)])
    np.save(DERIVED / 'y_B.npy', y[len(a):])
    write(DERIVED / 'mapping_receipt.json', dict(status='EXACT_DATE_AND_AB_LABEL_MATCH', n_A=len(a), n_B=len(b),
          mapping_sha256=sha(path), raw_base_sha256=sha(RAW / 'train_base.parquet'),
          original_meta_sha256=sha(PROJECT / 'data/homecredit-default/x_meta.npy'),
          case_id_meaning='Application identifier; not proof of customer identity or customer independence',
          C_D_labels_accessed=False, test_predictions_generated=False))
    return mapped


def aggregate_all(variant, mapping):
    groups = LGB_GROUPS if variant == 'lgb' else CAT_GROUPS
    folder = DERIVED / variant / 'tables'
    folder.mkdir(parents=True, exist_ok=True)
    receipts = []
    ids = mapping['case_id']
    for group, pattern, depth in groups:
        output, receipt = folder / f'{group}.parquet', folder / f'{group}.json'
        if receipt.exists():
            item = read(receipt)
            assert sha(output) == item['output_sha256']
            receipts.append(item)
            continue
        start = time.monotonic()
        pieces, input_rows, raw_hashes, empty_shards = [], 0, {}, []
        paths = sorted(RAW.glob(pattern))
        assert paths, pattern
        for raw in paths:
            frame = pl.scan_parquet(raw).filter(pl.col('case_id').is_in(ids)).collect(streaming=True)
            input_rows += len(frame)
            raw_hashes[raw.name] = sha(raw)
            if not len(frame):
                empty_shards.append(raw.name)
                continue
            pieces.append(aggregate(frame, variant, raw.name, depth))
            del frame
            gc.collect()
        combined = pl.concat(pieces, how='vertical_relaxed') if pieces else pl.DataFrame({'case_id':pl.Series([],dtype=pl.Int64)})
        duplicates = len(combined) - combined['case_id'].n_unique()
        # Upstream read_files aggregates each shard before unique(case_id). No
        # silent repair: a cross-shard case would make its original behavior ambiguous.
        if duplicates:
            write(receipt.with_name(group+'_conflict.json'), dict(status='BLOCKED_SHARD_DUPLICATION', count=duplicates))
            raise ValueError(f'Author per-shard aggregation drops duplicate cases in {variant}/{group}; review required')
        combined.write_parquet(output)
        item = dict(group=group, variant=variant, depth=depth, input_rows=input_rows,
                    output_rows=len(combined), output_columns=len(combined.columns)-1,
                    cross_shard_duplicate_cases=duplicates, seconds=time.monotonic()-start,
                    empty_AB_shards_skipped=empty_shards,
                    input_sha256=raw_hashes, output_sha256=sha(output),
                    peak_rss_snapshot_bytes=psutil.Process().memory_info().rss)
        write(receipt, item)
        receipts.append(item)
        print(f'{variant}/{group}: {input_rows:,} -> {len(combined):,} cases, {len(combined.columns)-1} features', flush=True)
        del pieces, combined
        gc.collect()
    return receipts


def fit_filter(series, a_mask, variant):
    train = series.filter(a_mask)
    if train.null_count() / len(train) > .95:
        return 'A_missing_gt_95pct'
    if series.dtype == pl.String and (train.n_unique() == 1 or train.n_unique() > 50):
        return 'A_string_cardinality_not_in_2_to_50'
    c = series.name
    if variant == 'lgb' and c[-1] not in ['P', 'A', 'L', 'M'] and ('month_' in c or 'year_' in c):
        return 'author_calendar_name_filter'
    return None


def join_applications(base, source):
    # The two author pipelines use different integer widths for this key.
    # Rechunk also avoids a Polars 0.20.31 multi-chunk gather panic.
    base = base.with_columns(pl.col('case_id').cast(pl.Int64)).rechunk()
    source = source.with_columns(pl.col('case_id').cast(pl.Int64)).rechunk()
    out = base.join(source, on='case_id', how='left', coalesce=True)
    assert out['case_id'].equals(base['case_id'])
    return out


def align_and_filter(variant, mapping):
    path = DERIVED / variant / 'schema.json'
    if path.exists():
        info = read(path)
        for item in info['feature_tables']:
            assert sha(HERE / item['path']) == item['sha256']
        return info
    a_mask = mapping['split'] == 'A'
    groups = LGB_GROUPS if variant == 'lgb' else CAT_GROUPS
    folder = DERIVED / variant / 'aligned'
    folder.mkdir(exist_ok=True)
    seen, numeric, categorical, rejected, tables = set(), [], [], {}, []
    base = mapping.select('case_id', 'date_decision').rechunk()
    calendar = mapping.select(
        pl.col('date_decision').dt.month().alias('decision_month'),
        pl.col('date_decision').dt.weekday().alias('decision_weekday'))
    all_tables = [('calendar', calendar)] + [(g, None) for g, _, _ in groups]
    for table_i, (group, calendar_frame) in enumerate(all_tables):
        if calendar_frame is not None:
            frame = calendar_frame
        else:
            source = pl.read_parquet(DERIVED / variant / 'tables' / f'{group}.parquet').rechunk()
            assert 'target' not in source.columns
            frame = join_applications(base, source)
            for c in source.columns:
                if c == 'case_id':
                    continue
                if c.endswith('D') and not (variant == 'lgb' and 'count' in c):
                    frame = frame.with_columns((pl.col(c)-pl.col('date_decision')).dt.total_days().alias(c))
            frame = frame.drop('case_id', 'date_decision')
        keep, rename = [], {}
        for c in frame.columns:
            assert c not in EXCLUDED
            # Prefix identifies the original table, removing join-suffix ambiguity.
            new = group + '__' + c
            assert new not in seen
            reason = fit_filter(frame[c], a_mask, variant)
            if reason:
                rejected[new] = reason
                continue
            seen.add(new)
            keep.append(c)
            rename[c] = new
            if frame[c].dtype == pl.String or frame[c].dtype == pl.Categorical:
                categorical.append(new)
            elif frame[c].dtype.is_numeric() or frame[c].dtype == pl.Boolean:
                numeric.append(new)
            else:
                raise TypeError(f'Unconverted output: {new}: {frame[c].dtype}')
        if keep:
            out = folder / f'{group}.parquet'
            frame.select(keep).rename(rename).write_parquet(out)
            tables.append(dict(path=str(out.relative_to(HERE)), sha256=sha(out)))
        del frame
        gc.collect()
    info = dict(variant=variant, n_A=int(a_mask.sum()), n_B=int((~a_mask).sum()),
                numeric=numeric, categorical=categorical, rejected=rejected, feature_tables=tables,
                excluded=list(sorted(EXCLUDED)), fit_rows='A only',
                adaptation='Author aggregations and date deltas; A-only column filters; table-prefixed names; shared train-only median/unknown/scaling/missing flags in models. Author sentinel fill and data-driven dtype reduction not applied.',
                not_exact_champion_feature_mask='Original final column list comes from absent upstream fitted models; no assertion of exactly 644/661 inputs',
                chronology_limit='Source record as-of availability and target maturity unverified; not strict point-in-time')
    write(path, info)
    return info


def main():
    DERIVED.mkdir(exist_ok=True)
    before = protected_hashes()
    mapping = make_mapping()
    for variant in ['lgb', 'cat']:
        aggregate_all(variant, mapping)
        info = align_and_filter(variant, mapping)
        print(variant, len(info['numeric']), 'numeric', len(info['categorical']), 'categorical', flush=True)
    assert protected_hashes() == before
    receipt = dict(status='AUTHOR_AGGREGATIONS_COMPLETE_AB',
          n_A=int((mapping['split']=='A').sum()), n_B=int((mapping['split']=='B').sum()),
          original_protected_hashes=before, C_D_labels_accessed=False, official_competition_test_accessed=False,
          feature_schema_hashes={v:sha(DERIVED/v/'schema.json') for v in ['lgb','cat']})
    path=HERE/'feature_extraction_receipt.json'
    old=read(path) if path.exists() else None
    if old and 'feature_schema_hashes' in old:
        assert {k:v for k,v in old.items() if k!='completed_utc'}==receipt
    else:
        assert not (HERE/'PROTOCOL_LOCK.json').exists(),'Cannot revise feature receipt after freeze'
        write(path,dict(receipt,completed_utc=now()))


if __name__ == '__main__':
    main()
