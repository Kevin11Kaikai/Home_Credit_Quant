"""Restore hash-pinned optional source dependencies locally, not for redistribution.

No training, data download, submissions, login, or terms acceptance. The optional
cache root is a prior authorized project, never a mirror. Network paths fail
closed if the recorded bytes cannot be recovered.
"""
import argparse, ast, hashlib, json, shutil, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FE='experiments/champion_features_20261014'
FULL='experiments/full_solutions_20261014'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(b):return hashlib.sha256(b).hexdigest()

def matched_bytes(data, expected):
    if digest(data)==expected:return data
    text=data.decode('utf-8').replace('\r\n','\n')
    for candidate in [text.encode(),text.replace('\n','\r\n').encode()]:
        if digest(candidate)==expected:return candidate
    raise ValueError('Pinned SHA256 mismatch; source was not installed')

def obtain(relative, expected, url=None, cache=None, kernel=None):
    target=ROOT/relative
    if target.exists():return matched_bytes(target.read_bytes(),expected)
    if cache:
        data=(cache/relative).read_bytes()
    elif kernel:
        from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
        api=KaggleApi();api.authenticate()
        request=ApiGetKernelRequest()
        request.user_name,request.kernel_slug=kernel['kernel'].split('/')
        request.version_label=str(kernel['version'])
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.get_kernel(request)
        data=response.blob.source.encode('utf-8')
    else:
        assert url.startswith('https://raw.githubusercontent.com/')
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Home-Credit-Quant-reproduction'}),timeout=40) as response:
            data=response.read()
    data=matched_bytes(data,expected)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--champion',action='store_true')
    p.add_argument('--excel',action='store_true')
    p.add_argument('--cache-root',type=Path)
    args=p.parse_args()
    if not (args.champion or args.excel):p.error('Choose --champion and/or --excel explicitly')
    if args.champion:
        manifest=read(ROOT/FE/'SOURCE_MANIFEST.json')
        for record in manifest['kernels']:
            path=Path(record['file'].replace('\\','/'))
            obtain(str(Path(FE)/path),record['sha256'],cache=args.cache_root,kernel=record)
        for name in ['competition_files.json','lgb/kernel-metadata.json','cat/kernel-metadata.json']:
            source=ROOT/'evidence/feature_transfer/upstream'/name
            target=ROOT/FE/'upstream'/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        print('Champion public source dependencies restored and hash verified. No private models acquired.')
    if args.excel:
        manifest=read(ROOT/'evidence/full_solutions/sources/excel_official/SOURCE.json')
        sources={}
        for row in manifest['entries']:
            if 'sha256' not in row:continue
            relative=f'{FULL}/sources/excel_official/'+row['file']
            sources[row['file']]=obtain(relative,row['sha256'],row['url'],cache=args.cache_root).decode('utf-8').replace('\r\n','\n')
        dest=ROOT/FULL/'vendor';dest.mkdir(parents=True,exist_ok=True)
        core=sources['bin/excel_former.py'].replace('import lib','from . import excel_lib as lib')
        deep=sources['lib/deep.py'];tree=ast.parse(deep)
        wanted={'reglu','geglu','tanglu','get_activation_fn'}
        funcs=[ast.get_source_segment(deep,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        assert len(funcs)==4
        helper='import typing as ty\nimport torch\nfrom torch import Tensor\nimport torch.nn.functional as F\n'+'\n\n'.join(funcs)+'\n\n'+sources['lib/aug_utils.py']
        for name,text in [('excel.py',core),('excel_lib.py',helper),('__init__.py','')]:
            (dest/name).write_bytes(matched_bytes(text.encode(),manifest['vendor_files'][name]))
        print('ExcelFormer core and isolated helper restored to the recorded vendor hashes.')

if __name__=='__main__':main()
