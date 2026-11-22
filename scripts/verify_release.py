"""Verify the public allowlist, checksums and absence of private artifact types."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
errors=[]
for entry in manifest['files']:
    path=root/entry['path']
    if not path.is_file():
        errors.append('Missing: '+entry['path']);continue
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        errors.append('Checksum mismatch: '+entry['path'])
allowed={x['path'] for x in manifest['files']}|{'release_manifest.json'}
for path in root.rglob('*'):
    if not path.is_file():continue
    rel=path.relative_to(root).as_posix()
    if rel.startswith('.git/') or '__pycache__/' in rel:continue
    if rel not in allowed:errors.append('Unlisted file: '+rel)
if errors:
    print('\n'.join(errors));sys.exit(1)
print('PASS: all published files match the allowlist and their SHA256 checksums.')
print('No original observations or fitted artifacts are included in this release.')
