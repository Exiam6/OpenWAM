#!/usr/bin/env bash
set -euo pipefail
base=/data02/zifanz4/openwam-experiments
out="$base/endtoend-20260923/compressors-multiview"
exec 9>"$base/gpu-cm001-6.lock"
flock -n 9
python3 - <<'PY'
import json,hashlib,subprocess,shutil,datetime
from pathlib import Path
o=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923/compressors-multiview');j=json.loads((o/'protocol.json').read_text());assert not (o/'start.json').exists()
assert hashlib.sha256(Path('/home/zifanz4/openwam-experiments/scripts/endtoend/fit_compressors.py').read_bytes()).hexdigest()==j['code_sha256']
for p in ['/home/zifanz4/.local/state/openwam-selfcheck/paused.json','/home/zifanz4/openwam-experiments/studies/endtoend-20260923/paused.json']:assert not Path(p).exists()
s=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip().split(',');assert s[0]==j['gpu_uuid'] and int(s[1])<100 and int(s[2])==0 and int(s[3])==0,s
assert j['gpu_uuid'] not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
assert shutil.disk_usage(o).free>40*2**30
(o/'start.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'gpu_check':s}))
PY
export CUDA_VISIBLE_DEVICES=GPU-91a9a93b-b386-82f0-3009-88cc9bb79d06
export PYTHONPATH="$base/results/layers-20260922/native-preflight/deps:$base/endtoend-20260923/OpenWAM"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
trap 'printf "%s\n" "$?" > "$out/exit-code.txt"' EXIT
timeout --signal=TERM --kill-after=30s 7200 "$base/policy-env/bin/python" -u /home/zifanz4/openwam-experiments/scripts/endtoend/fit_compressors.py > "$out/run.log" 2>&1
