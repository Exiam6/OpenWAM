#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
root="$PWD/results/robustness-20260921"
python3 - <<'PY'
import hashlib,json,time
from pathlib import Path
p=Path('results/robustness-20260921')
deadline=time.monotonic()+3600
while not (p/'baseline-exit-code.txt').exists() and time.monotonic()<deadline:
 time.sleep(10)
assert (p/'baseline-exit-code.txt').read_text().strip()=='0'
assert (p/'adapter-exit-code.txt').read_text().strip()=='0'
selection=json.loads((p/'adapter-selection.json').read_text())
assert selection['selected'] is not None and selection['validation_only']
assert selection['candidates'][selection['selected']]['eligible']
record={'selected':selection['selected'],'seed':42,'unlocked_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'fixed_scenes_sha256':hashlib.sha256((p/'fixed-scenes.json').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((p/'adapter-selection.json').read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((p/'adapter-selected.pt').read_bytes()).hexdigest()}
(p/'confirmation-unlock.json').write_text(json.dumps(record,indent=2))
PY
bash scripts/run-fixed-robustness.sh identity-confirm
bash scripts/run-fixed-robustness.sh adapter-confirm
