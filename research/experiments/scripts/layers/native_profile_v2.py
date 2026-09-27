"""Bounded native trainer throughput instrumentation; no policy evaluation."""
import json,os,sys,time
from pathlib import Path
ROOT=Path('/data02/zifanz4/openwam-experiments');OUT=ROOT/'results/layers-20260922/native-preflight/trial2'
sys.path.insert(0,str(ROOT/'OpenWAM'))
sys.path.insert(0,str(ROOT/'OpenWAM/scripts'))
from openwam.train.openwam_trainer import OpenWAMTrainer
old=OpenWAMTrainer.log_step
def log(self,**kw):
    import torch
    row={k:kw[k] for k in ['global_step','opt_step','steps_per_sec','batch_size']}
    row.update(metrics={k:float(v) for k,v in kw['metrics'].items()},allocated_bytes=torch.cuda.max_memory_allocated(),wall_time=time.time(),rank=int(os.environ.get('RANK',0)))
    assert all(__import__('math').isfinite(x) for x in row['metrics'].values()),row
    with (OUT/f'profile-rank{row["rank"]}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return old(self,**kw)
OpenWAMTrainer.log_step=log
import train
if __name__=='__main__':train.main()
