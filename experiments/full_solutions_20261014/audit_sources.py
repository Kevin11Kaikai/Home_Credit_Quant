"""Evidence audit, no training and no downloads."""
import shutil,importlib.metadata
from common import *

def main():
    cache=Path(r'D:\Year3_Mao_Projects\Tabular_Transformer\benchmark\experiments\architecture_benchmark_20261010')
    entries=[e for e in read(cache/'sources.json') if e['architecture']=='excel']
    for e in entries:
        if 'sha256' in e:
            p=cache/'vendor/original/excel'/e['file'];assert sha(p)==e['sha256']
            out=HERE/'sources/excel_official'/e['file'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    upstream=(HERE/'sources/excel_official/bin/excel_former.py').read_text(encoding='utf-8')
    local=(HERE/'vendor/excel.py').read_text(encoding='utf-8')
    # Whitespace is normalized; the only architecture-file change is lib import.
    assert upstream.replace('import lib','from . import excel_lib as lib').strip()==local.strip()
    write(HERE/'sources/excel_official/SOURCE.json',dict(entries=entries,core_identical_except_import=True,local_patch='Runtime attention kernel adapter only; forward/gradient equivalence tested with original finite causal mask.',vendor_files={p.name:sha(p) for p in (HERE/'vendor').glob('*.py')}))
    old=read(PREVIOUS/'SOURCE_MANIFEST.json')
    manifest=dict(created_utc=now(),champion_public_notebooks=old['kernels']+[dict(url='https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-lightautoml-inference',version=12,kernel_id=55312659,sha256=sha(HERE/'sources/champion_dnn/fork-of-home-credit-lightautoml-inference.py'))],champion_writeup='https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy',fraud=read(HERE/'sources/fraud/SOURCE.json'),s6e7=read(HERE/'sources/s6e7/SOURCE.json'),excel_official=entries,packages={name:dict(version=importlib.metadata.version(name),license=importlib.metadata.metadata(name).get('License'),home_page=importlib.metadata.metadata(name).get('Home-page')) for name in ['catboost','lightautoml','masamlp','catstat','lightgbm']},data_policy='Existing official 2024 competition training cache only; no new dataset, uploads or submissions; previous A/B mappings unchanged.')
    write(HERE/'SOURCE_MANIFEST.json',manifest)
    audit=dict(status='BLOCKED_DEPENDENCY',scope='Exact full 2024 Home Credit champion reproduction',checked_utc=now(),source_artifacts=['LightGBM inference v16','CatBoost inference v12','DenseLight/LightAutoML inference v12','author first-place writeup and model/weight diagram'],known_recipe=dict(models=['LightGBM','DenseLight','CatBoost'],weights=[.4,.24,.36],final_early_week_subtract=.03),missing=['Original fitted model files referenced by the inference notebooks','Exact final 644/661 feature masks stored inside those artifacts','Full training configuration and final ensemble training notebook'],access_checks={n:read(HERE/'sources'/f'{n}_access.json') for n in ['hclgb-models','hc-catboost-models','hc-lama-models']},interpretation='Guessed author/slug dataset references derived from mount names return 403; public notebook API exposes blank data-source identifiers. This does not prove that the actual datasets are private, nor identify their true owner. No bypass or unverified mirror used.',local_fallback='Separately labeled three-family reconstruction using author feature expressions, five A-only grouped folds, documented fixed training parameters, fixed published weights and score heuristic; not exact original competition reproduction.',private_test_score='Not reproducible locally: official private labels are not available; no Kaggle submission authorized or made.')
    write(HERE/'CHAMPION_REPRODUCTION_AUDIT.json',audit)
    print('SOURCE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
