"""Read-only union of preserved original episodes and bounded recovery episodes."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import time

R = Path('/home/zifanz4/openwam-runtime')
ROOT = R / 'ex6-c_resourcerecovery_1020'
AUDIT_SOURCE = Path('/home/zifanz4/openwam-vae-control/research/greene/ex6-b_contributionaudit_1020/watch_transfer.py')
spec = importlib.util.spec_from_file_location('audit', AUDIT_SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def snapshot():
    p = audit.read(ROOT/'protocol.json')
    result = audit.snapshot(p['original_protocol_sha256'])
    terminal = True
    for task, arms in p['missing'].items():
        task_state = result['tasks'][task]
        task_state['original_worker_status'] = task_state['worker_status']
        claim = audit.read(R/'nightly-gpu/claims'/('ex6c-'+task.replace('_','-')+'.json'))
        status = claim['status'] if claim else 'pending'
        task_state['worker_status'] = status
        terminal &= status in ['complete','failed']
        cohort = audit.read(Path(p['original_root'])/'tasks'/task/'code/scenes'/task/'manifest.json')
        selected = {r['seed'] for r in cohort['accepted']}
        for arm, missing in arms.items():
            old = task_state['arms'][arm]
            retained = [r for r in old['rows'] if r['status']=='verified']
            assert not (set(missing) & {r['scene_seed'] for r in retained})
            recovery = []
            for scene in missing:
                path = ROOT/'tasks'/task/arm/'episodes'/f'seed-{scene}'/'result.json'
                if path.exists():
                    try:
                        recovery.append(audit.validate_episode(path,task,arm,selected))
                    except Exception as exc:
                        recovery.append({'status':'audit_failure','source':str(path),'error':repr(exc)})
            old['original_noncomplete_rows'] = [r for r in old['rows'] if r['status']!='verified']
            old['rows'] = retained + recovery
            verified = [r for r in old['rows'] if r['status']=='verified']
            old.update(verified=len(verified), successes=sum(r['success'] for r in verified), missing_or_unverified=5-len(verified))
        task_state['complete_pairs'] = 0
        for scene in selected:
            pair = [next((r for r in a['rows'] if r.get('scene_seed')==scene and r['status']=='verified'),None)
                    for a in task_state['arms'].values()]
            if all(pair):
                assert len({json.dumps(r['pairing']) for r in pair})==1
                task_state['complete_pairs'] += 1
    result['verified_episodes'] = sum(a['verified'] for t in result['tasks'].values() for a in t['arms'].values())
    result['complete'] = result['verified_episodes']==140 and all(t['complete_pairs']==5 for t in result['tasks'].values())
    result['all_jobs_terminal'] = terminal
    result['recovery_protocol_sha256'] = audit.sha(ROOT/'protocol.json')
    result['conclusion'] = ('Complete exploratory transfer pilot; one training seed, no population superiority claim.' if result['complete']
                            else 'Incomplete exploratory transfer pilot; resource interruptions are not policy failures.')
    return result


if __name__ == '__main__':
    audit.OUT = ROOT
    while True:
        result = snapshot()
        audit.publish(result)
        if result['all_jobs_terminal'] or result['expired']:
            break
        time.sleep(30)
