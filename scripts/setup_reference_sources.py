"""Fetch pinned third-party sources separately from the authors' own source pack."""
from pathlib import Path
import argparse
import hashlib
import io
import urllib.request
import zipfile
import shutil
import json

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'astgcn':('guoshnBJTU/ASTGCN-2019-pytorch','2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296',
        '413b20d81645e4a3dd0d346a86fdd73bdeebb3f6b8bc20c3331a454789b48313'),
    'tslib':('thuml/Time-Series-Library','4e938a1767106324dd753b2a44832bf870a0252e',
        '8ad00c38d00baa6b4a17df2a688a96c945a0c005b6caceca0e932e6f5ac6edaa')}
p=argparse.ArgumentParser();p.add_argument('--source',choices=list(SOURCES),required=True);a=p.parse_args()
repo,commit,expected=SOURCES[a.source]
destination=ROOT/'vendor'/a.source
if destination.exists():
    print('Source directory already exists; inspect its SOURCE.json and do not overwrite it.')
    raise SystemExit(0)
payload=urllib.request.urlopen(f'https://codeload.github.com/{repo}/zip/{commit}',timeout=60).read()
if hashlib.sha256(payload).hexdigest()!=expected:raise RuntimeError('Pinned source archive SHA256 mismatch; stop and verify the upstream archive.')
destination.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        parts=Path(item.filename).parts[1:]
        if not parts or item.is_dir():continue
        target=destination.joinpath(*parts).resolve()
        if not target.is_relative_to(destination.resolve()):raise RuntimeError('Unsafe archive path')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(item))
(destination/'SOURCE.json').write_text(json.dumps(dict(repository='https://github.com/'+repo,
    commit=commit,archive_sha256=expected),indent=2),encoding='utf8')
print('Pinned reference source downloaded',a.source,commit)
