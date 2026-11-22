"""Print the saved suite progress without opening a model or using the GPU."""
from pathlib import Path
from collections import Counter
import json
ROOT=Path(__file__).resolve().parents[1]
completed=Counter();active=[]
def read(path):
    try:return json.loads(path.read_text(encoding='utf8'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
for path in (ROOT/'results/checkpoints').glob('*/run_manifest.json'):
    manifest=read(path)
    if manifest.get('smoke'):continue
    if manifest.get('status')=='PASS':completed[manifest['group']]+=1
    else:active.append(dict(run=path.parent.name,status=manifest.get('status','UPDATING')))
records={};stage=read(ROOT/'manifests/current_stage.json')
for group,planned in dict(E01=24,E02=8,E03=40,E04=60,E05=50,E06=85,E08=108,LEGACY=48,LEGACY_ROLL=32).items():
    entries=list((ROOT/'experiments'/group).glob('*.json'));passed=0
    for entry in entries:
        record=read(entry)
        if record.get('run_dir') and read(Path(record['run_dir'])/'run_manifest.json').get('status')=='PASS':passed+=1
    records[group]=dict(completed_references=passed,planned_references=planned,
        group_status=read(ROOT/f'manifests/{group}_status.json').get('status','RUNNING' if stage.get('group')==group else 'PENDING'))
print(json.dumps(dict(current_stage=stage,
    unique_completed_formal_runs=dict(completed),references=records,active=active),ensure_ascii=False,indent=2))
