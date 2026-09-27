#!/usr/bin/env python3
"""User-requested monitor; queue at most one self-check in the existing task."""
import datetime, fcntl, hashlib, json, os, sqlite3, subprocess, sys
from pathlib import Path
ROOT = Path('/home/zifanz4/openwam-experiments')
RUN = Path('/data02/zifanz4/openwam-experiments/results/norm-20260921')
STATE = Path('/home/zifanz4/.local/state/openwam-selfcheck')
THREAD = '01a0bbfe-bfd4-7110-b17d-6654af16113f'
MARKER = '[OPENWAM_SELFCHECK_3H]'
LEGACY_MARKER = '[OPENWAM_SELFCHECK_15M]'
STATE.mkdir(parents=True, exist_ok=True)
# Explicit user pause persists even if an old command is invoked manually.
if (STATE/'paused.json').exists(): sys.exit(0)
lock = (STATE/'monitor.lock').open('w')
try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError: sys.exit(0)
now = datetime.datetime.now().astimezone().isoformat()
def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {'exit_code':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    except Exception as e: return {'error':str(e)}
summary = json.loads((RUN/'summary.json').read_text()) if (RUN/'summary.json').exists() else None
snapshot = {'checked_at':now,'thread':THREAD,'summary_sha256':hashlib.sha256((RUN/'summary.json').read_bytes()).hexdigest() if summary else None,
 'primary':{k:{'successes':v['successes'],'episodes':v['episodes']} for k,v in summary['primary'].items()} if summary else {},
 'integrity_passed':bool(summary and summary['all_integrity_checks_passed']),
 'exit_codes':{p.name:p.read_text().strip() for p in RUN.glob('*exit-code.txt')},
 'gpu':command(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader']),
 'tmux':command(['tmux','list-sessions','-F','#{session_name}'])}
# Include the active diagnostic without treating an incomplete JSON write as failure.
def read_optional_json(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except (json.JSONDecodeError,OSError): return {'read_state':'not_yet_atomic_or_readable'}
diag=RUN.parent/'repeatability-20260921'
if diag.exists():
    diag_state={'launch':read_optional_json(diag/'launch.json'),'summary':read_optional_json(diag/'summary.json'),'exit_code':(diag/'exit-code.txt').read_text().strip() if (diag/'exit-code.txt').exists() else None,'generated_chunks':{}}
    for phase in ['A','B']:
        rows=read_optional_json(diag/f'process-{phase}'/'records.json')
        diag_state['generated_chunks'][phase]=len(rows) if isinstance(rows,list) else rows
    result=diag_state['summary']
    if isinstance(result,dict) and 'aggregates' in result:
        diag_state['summary']={k:result[k] for k in ['all_integrity_checks_passed','generated_chunks','unique_inputs','aggregates']}
    snapshot['repeatability']=diag_state
    closed=diag/'closedloop'
    if closed.exists():
        closed_state={'launch':read_optional_json(closed/'launch.json'),'exit_code':(closed/'exit-code.txt').read_text().strip() if (closed/'exit-code.txt').exists() else None,'groups':{}}
        for name in ['r0-clean','r0-noise','r1-clean','r1-noise']:
            group=read_optional_json(closed/name/'summary.json')
            closed_state['groups'][name]={k:group.get(k) for k in ['episodes','successes']} if isinstance(group,dict) else None
        result=read_optional_json(closed/'summary.json')
        closed_state['summary']={k:result.get(k) for k in ['all_integrity_checks_passed','episodes','requests','paired_trajectories']} if isinstance(result,dict) else None
        snapshot['closedloop_repeatability']=closed_state
probe=RUN.parent/'attribution-20260921'
if probe.exists():
    result=read_optional_json(probe/'summary.json')
    snapshot['attribution_probe']={'launch':read_optional_json(probe/'launch.json'),'exit_code':(probe/'exit-code.txt').read_text().strip() if (probe/'exit-code.txt').exists() else None,'rounds':{f'round{i}':read_optional_json(probe/f'round{i}'/'summary.json')for i in[0,1]},'summary':{k:result.get(k)for k in['all_integrity_checks_passed','traces','commands_executed','policy_inference_calls']}if isinstance(result,dict)else None}
# Current user-authorized study, separate from all frozen historical results.
active = read_optional_json(STATE/'active-study.json')
if isinstance(active, dict) and 'root' in active:
    study_root = Path(active['root'])
    snapshot['active_study'] = {'root': str(study_root), 'launch': read_optional_json(study_root/'launch.json'), 'progress': read_optional_json(study_root/'progress.json')}

# Read only the queue for this task; mutations always use the supported CLI.
c=sqlite3.connect('file:/home/zifanz4/.codex/queue_1.sqlite?mode=ro',uri=True)
pending=c.execute('select id,payload_json from queued_items where thread_id=?',(THREAD,)).fetchall();c.close()
own=[row[0] for row in pending if MARKER in row[1] or LEGACY_MARKER in row[1]]
if own: snapshot['wake']={'state':'already_queued','message_ids':own}
else:
    prompt=f'''{MARKER} 用户要求每3小时自检并思考下一步。定时触发：{now}。
先读取 /home/zifanz4/.local/state/openwam-selfcheck/latest.json、/home/zifanz4/openwam-experiments/STATUS.md 与 NEXT_STEPS.md，再核查实际日志、退出码、结果及资源。记录本次判断和下一步到 selfchecks.md；开始处理时写 /home/zifanz4/.local/state/openwam-selfcheck/last-reasoned.json（当前时间、状态、下一步）。
以 STATUS.md、NEXT_STEPS.md 和 active-study.json 中当前已授权研究为准，读取对应计划与 progress.json；原 studies/layers-20260922 窗口已结束，不得重启。遵守各研究固定deadline；不得因延迟消息延长预算。若存在paused.json立即保持暂停。按照既有授权推进 OpenWAM 表征/扰动评测；不要重复已完成80次评测，不启动新子代理，不终止其他任务，不改动冻结协议或挑选有利结果。新实验先固定方案、做检查，再在空闲资源上有界运行。每次需解释下一步解决哪个不确定性。
只在有实质进展、完成、失败或需要用户决策时通知用户；没有变化保持安静，保留本地自检记录。实质结果继续更新既有公网报告，发布失败要如实记录。不要重复创建定时器或承诺尚未验证的自动唤醒。'''
    snapshot['wake']=command(['/home/zifanz4/.local/bin/codex','queue','--thread',THREAD,'--message',prompt])
    snapshot['wake']['state']='queued_unverified_delivery' if snapshot['wake'].get('exit_code')==0 else 'queue_failed'
# A queued message is not evidence that an agent executed a reasoning turn.
ack=STATE/'last-reasoned.json'
snapshot['last_reasoned']=json.loads(ack.read_text()) if ack.exists() else None
payload=json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n'
tmp=STATE/'latest.json.tmp';tmp.write_text(payload);tmp.replace(STATE/'latest.json')
with (STATE/'checks.jsonl').open('a') as f:f.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
print(json.dumps({'checked_at':now,'integrity_passed':snapshot['integrity_passed'],'wake':snapshot['wake'],'last_reasoned':snapshot['last_reasoned']},ensure_ascii=False))
