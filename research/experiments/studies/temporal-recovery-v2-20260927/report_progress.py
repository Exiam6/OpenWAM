"""Read-only experiment inspection; write reporting snapshots only."""
import datetime,json
from pathlib import Path
S=Path(__file__).parent;plan=json.loads((S/'recovery-plan.json').read_text());O=Path(plan['output'])
def read(p):return json.loads(p.read_text()) if p.exists() else None
q={'time':datetime.datetime.now().astimezone().isoformat(),'deadline':plan['deadline'],'output':str(O),'training_completed':6,'rollouts_total':1080,'rollouts_complete':0,'technical_failures':0,'lanes':{},'groups':{},'audit':read(S/'results-audit.json'),'original_failed_starts_retained':True,'supersedes_failed_start':'temporal-recovery-20260927','source_manifest':str(S/'evaluation-code-manifest.json')}
for arm in ['mean','learned']:
 q['lanes'][arm]={k:read(O/'lanes'/arm/(k+'.json')) for k in ['started','complete','exit']}
 for seed in [42,43,44]:
  r=O/'evaluation'/f'{arm}-seed{seed}';server=(r/'server.log').read_text(errors='replace') if (r/'server.log').exists() else '';dev=(r/'development/run.log').read_text(errors='replace') if (r/'development/run.log').exists() else ''
  q['groups'][f'{arm}-{seed}']={'started':read(r/'runner-started.json') is not None,'child':read(r/'child.json'),'exit':read(r/'exit.json'),'server_ready':'ENDTOEND_SERVER_READY' in server,'multi_prompt_gate':'MULTIPROMPT_GATE_PASSED' in dev,'development_gate':read(r/'development/result.json'),'heldout_complete':read(r/'complete.json'),'heldout_groups_started':[str(p.relative_to(r)) for p in r.glob('*/*/started.json')]}
  for p in r.glob('*/*/seed-*/result.json'):
   x=read(p);q['rollouts_complete']+=x['status']=='complete';q['technical_failures']+=x['status']!='complete'
terminal=all(x['exit'] for x in q['lanes'].values());q['status']='terminal' if terminal else 'heldout_evaluation_running' if q['rollouts_complete'] or any(x['heldout_groups_started'] for x in q['groups'].values()) else 'deployment_gates_running'
(S/'progress.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({'time':q['time'],'status':q['status'],'rollouts_complete':q['rollouts_complete'],'technical_failures':q['technical_failures'],'groups':{k:{x:v[x] for x in ['started','server_ready','multi_prompt_gate','exit']} for k,v in q['groups'].items()}}))
