"""Bounded official-only source/data acquisition for the requested FE study."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'
RAW = HERE / 'raw'
COMPETITION = 'home-credit-credit-risk-model-stability'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def pin_sources():
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest
    path = HERE / 'SOURCE_MANIFEST.json'
    if path.exists():
        obj = json.loads(path.read_text())
        for entry in obj['kernels']:
            assert sha(HERE / entry['file']) == entry['sha256']
        return
    api = KaggleApi()
    api.authenticate()
    sources = []
    for variant in ['lgb', 'cat']:
        meta = json.loads((UPSTREAM / variant / 'kernel-metadata.json').read_text())
        owner, slug = meta['id'].split('/')
        with api.build_kaggle_client() as client:
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = owner, slug
            response = client.kernels.kernels_api_client.get_kernel(request)
        source = UPSTREAM / variant / meta['code_file']
        assert source.read_text(encoding='utf-8') == response.blob.source.replace('\r\n', '\n')
        sources.append(dict(author=owner, kernel=meta['id'], kernel_id=meta['id_no'],
                            version=response.metadata.current_version_number,
                            url='https://www.kaggle.com/code/' + meta['id'],
                            file=str(source.relative_to(HERE)), sha256=sha(source),
                            upstream_docker_image=meta['docker_image'],
                            license='Not returned by official kernel API; retain locally with attribution, no redistribution assumed',
                            usage='Audited feature classes/functions only; no top-level inference, weights or submission'))
    save(path, dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), kernels=sources,
                   data_source='https://www.kaggle.com/competitions/' + COMPETITION + '/data',
                   access='User accepted competition terms and authorized OAuth in this conversation; no new terms accepted',
                   restriction='Competition data and row-level derivatives stay local; no upload or submission',
                   initial_network_failure='WinError 10013 in sandbox; official read/download escalation subsequently approved'))


def main():
    pin_sources()
    RAW.mkdir(exist_ok=True)
    files = json.loads((UPSTREAM / 'competition_files.json').read_text(encoding='utf-8-sig'))
    selected = [f for f in files if f['name'].startswith('parquet_files/train/') or f['name'] == 'feature_definitions.csv']
    assert sum(f['size'] for f in selected) < 2 * 2**30
    assert len(selected) == 33
    record_path = HERE / 'download_receipt.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else dict(files={}, started_utc=datetime.now(timezone.utc).isoformat())
    exe = Path(sys.executable).parent / 'kaggle.exe'
    # Base and field definitions first, so mapping can be audited while the rest downloads.
    selected.sort(key=lambda f: (f['name'] not in ['parquet_files/train/train_base.parquet', 'feature_definitions.csv'], f['name']))
    for i, entry in enumerate(selected):
        name = Path(entry['name']).name
        target = RAW / name
        previous = record['files'].get(name)
        if previous and previous.get('status') == 'VERIFIED':
            assert target.stat().st_size == entry['size'] and sha(target) == previous['sha256']
            continue
        start = time.monotonic()
        command = [str(exe), 'competitions', 'download', COMPETITION, '-f', entry['name'], '-p', str(RAW), '-q']
        if not target.exists():
            result = subprocess.run(command, capture_output=True, timeout=300)
            if result.returncode:
                record['files'][name] = dict(status='FAILED', exit_code=result.returncode, command=command,
                                            seconds=time.monotonic()-start,
                                            reason='Official downloader failed; no automatic retry or alternate source')
                save(record_path, record)
                raise RuntimeError(f'Official download failed for {name}; exit={result.returncode}')
            if not target.exists():
                archive = RAW / (name + '.zip')
                if not archive.exists():
                    raise RuntimeError(f'Expected downloaded file or single-file archive missing: {name}')
                with zipfile.ZipFile(archive) as z:
                    candidates = [m for m in z.infolist() if Path(m.filename).name == name and not m.is_dir()]
                    assert len(candidates) == 1 and candidates[0].file_size == entry['size']
                    with z.open(candidates[0]) as src, target.open('wb') as dst:
                        for block in iter(lambda: src.read(1 << 20), b''):
                            dst.write(block)
        assert target.stat().st_size == entry['size'], f'Size mismatch: {name}'
        record['files'][name] = dict(status='VERIFIED', size=target.stat().st_size, sha256=sha(target),
                                    seconds=time.monotonic()-start, command=command, official_name=entry['name'])
        save(record_path, record)
        print(f'{i+1}/{len(selected)} VERIFIED {name}', flush=True)
    record['status'] = 'COMPLETE_TRAIN_PARQUET_ONLY'
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save(record_path, record)


if __name__ == '__main__':
    main()
