#!/bin/bash
# Start second-slot post-workers on shenlong-gpu-01 (amendment 4). One per GPU, setsid+nohup.
set -u
R=/home/zifanz4/openwam-runtime/rae-factorial-20261010
PY=/home/zifanz4/openwam-runtime/policy-env/bin/python
test "$(hostname)" = "shenlong-gpu-01.cs.illinois.edu" || { echo "wrong host"; exit 2; }
test -f $R/resource-amendment-gpu01-second-slot.json || { echo "amendment 4 not applied"; exit 2; }
for gpu in 0 1 2 3 4 5 6 7; do
  f=$R/post-worker-gpu$gpu-slot1.json
  if [ -s $f ] && pid=$(python3 -c "import json;print(json.load(open('$f'))['pid'])" 2>/dev/null) && kill -0 $pid 2>/dev/null; then echo "gpu$gpu slot1 already running pid=$pid"; continue; fi
  setsid nohup $PY -u $R/post_worker.py --gpu $gpu --slot 1 >> $R/post-worker-gpu$gpu-slot1.out 2>&1 < /dev/null &
  echo "gpu$gpu slot1: started pid=$!"
done
