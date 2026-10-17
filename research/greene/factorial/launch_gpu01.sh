#!/bin/bash
# Start the factorial workers on shenlong-gpu-01 (setsid+nohup; no systemd user units here).
# Usage: launch_gpu01.sh train|post   -- one worker per GPU, PIDs recorded in *-services.json
set -u
R=/home/zifanz4/openwam-runtime/rae-factorial-20261010
PY=/home/zifanz4/openwam-runtime/policy-env/bin/python
kind=${1:?train|post}
case $kind in
  train) script=worker.py; out=worker; json=worker-services.json;;
  post)  script=post_worker.py; out=post-worker; json=post-worker-services.json;;
  *) echo "unknown kind $kind"; exit 2;;
esac
test "$(hostname)" = "shenlong-gpu-01.cs.illinois.edu" || { echo "wrong host"; exit 2; }
test -f $R/resource-amendment-gpu01.json || { echo "amendment not applied"; exit 2; }
echo "[" > $R/$json.tmp
for gpu in 0 1 2 3 4 5 6 7; do
  if [ -s $R/$out-gpu$gpu.json ] && pid=$(python3 -c "import json;print(json.load(open('$R/$out-gpu$gpu.json'))['pid'])" 2>/dev/null) && kill -0 $pid 2>/dev/null; then
    echo "gpu$gpu: $out already running pid=$pid"; continue
  fi
  setsid nohup $PY -u $R/$script --gpu $gpu >> $R/$out-gpu$gpu.out 2>&1 < /dev/null &
  pid=$!
  echo "  {\"gpu\": \"$gpu\", \"pid\": $pid, \"script\": \"$script\", \"started\": \"$(date -Is)\", \"host\": \"$(hostname)\"}," >> $R/$json.tmp
  echo "gpu$gpu: started $script pid=$pid"
done
sed -i '$ s/,$//' $R/$json.tmp; echo "]" >> $R/$json.tmp; mv $R/$json.tmp $R/$json
