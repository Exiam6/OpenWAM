"""Archive bounded contract-check outcome without replaying model computations."""
import datetime
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

R = Path('/home/zifanz4/openwam-experiments')
S = R / 'studies/goalaux-20260922'
L = R / 'studies/layers-20260922'
REPORT = R / 'reports/layers-20260922'
STATE = Path('/home/zifanz4/.local/state/openwam-selfcheck')
now = datetime.datetime.now().astimezone().isoformat()
result = json.loads((S / 'attempt2/contract-result.json').read_text())
assert result['passed'] and (S / 'attempt2/contract-exit-code.txt').read_text().strip() == '0'
next_step = ('Freeze a separate bounded GPU BF16 parity and real-shape throughput preflight, then check current idle resources. '
             'Only after those checks, freeze the full matched L12 reconstruction-only versus goal-auxiliary protocol, '
             'including joint goal-improvement/state-retention criteria, before any new compressor fitting or fresh-scene collection. '
             'The previous60confirmation episodes are exposed development for this follow-up; no winner/weight sweep or completed-evaluation replay. '
             'Original deadline2026-09-23T01:15:12-05:00 remains unchanged.')
publication = {'attempted_at': now, 'operation': 'Sites get_site', 'result': 'NOT_FOUND', 'http_code': 404,
               'deployment_attempted': False, 'existing_public_version_unchanged': True,
               'reason': 'Existing project unavailable through current Sites connection', 'local_report_updated': True}
(S / 'publication-attempt.json').write_text(json.dumps(publication, indent=2) + '\n')
p = L / 'progress.json'
progress = json.loads(p.read_text())
progress.setdefault('superseded_progress_fields', []).append({
    'recorded_at': now, 'confirmation': progress.get('confirmation'),
    'secondary_initial_goal': progress.get('secondary_initial_goal')})
progress.update(updated_at=now, status='completed_layer_study_goalaux_cpu_contract_passed', next_step=next_step,
                confirmation='COMPLETE exit0;60fresh scenes; primary acceptance failed; independent audit passed',
                secondary_initial_goal='COMPLETE exit0; same60fresh scenes; independent270prediction/6interval audit passed; mixed task effects',
                goal_aux_cpu_contract={'passed': True, 'path': str(S / 'attempt2/contract-result.json'),
                                       'scope': 'one training frame;CPU fp32; no model fit or benefit measurement',
                                       'attempt1_exit': 1, 'attempt2_exit': 0},
                goal_aux_model_training_started=False)
progress['public_report_update'] = {'status': 'blocked', 'last_checked_at': now, 'reason': 'Sites get_site404', 'local_report': str(REPORT/'index.html')}
p.write_text(json.dumps(progress, indent=2) + '\n')
record = json.loads((STATE / 'last-reasoned.json').read_text())
record.update(updated_at=now, status=progress['status'], next_step=next_step, deadline=progress['deadline'],
              notification='Meaningful new CPU implementation check; no new scientific gain result',
              timer_evidence='The13:30queued message reached this task at~15:51CDT. Delivery observed with delay; timely15min idle wake not established.')
(STATE / 'last-reasoned.json').write_text(json.dumps(record, indent=2) + '\n')
entry = f'''# Selfcheck — auxiliary CPU contract passed — {now}

Completed layer-study results are unchanged; all final scoring exit codes are0.
New bounded CPU check: attempt1 exited1 before any parameter update because the
G2 fixture's relative data path differed on G1; attempt2 fixed only this path,
kept raw image hash verification and passed in6.23s with2threads and noGPU.
For one training frame/seed42 checkpoint, lambda0 preserves native loss, RNG,
80parameter gradients and one AdamW update exactly. Auxiliary loss gives finite
nonzero encoder/head gradients; decoder gradients are absent. Changing later
features leaves the initial mean exact; gradients to future input are zero.
This is CPU FP32 engineering evidence, not improved representations or policy.
No new compressor checkpoint or held-out scoring; no completed experiment replay.
Original deadline01:15:12CDT Sep23 and all resource caps remain unchanged.

Next: {next_step}

Sites get_site again404; local report updated, publicversion26 remains stale.
Single existing15mincron retained. The delayed13:30message was observed at15:51;
this verifies delayed delivery, not reliable15min autonomous reasoning cadence.
See studies/goalaux-20260922/attempt2/contract-result.json and PATH_AMENDMENT.md.
'''
for filename in ['STATUS.md', 'NEXT_STEPS.md']:
    file = R / filename
    file.write_text(entry + '\n---\n\n' + file.read_text())
with (R / 'selfchecks.md').open('a') as f:
    f.write('\n\n' + entry)
proposal = L / 'PROPOSED_GOAL_AUX_FOLLOWUP.md'
text = proposal.read_text()
text = text.replace('These checks have not yet been executed for the new auxiliary implementation.\nThey must be logged separately from the already passed primary-study audits.',
                    'Update: a separate bounded CPU FP32 contract check passed on one existing\ntraining frame and the seed42 checkpoint. Zero-weight native loss/RNG/gradient/\nAdamW update equality, nonzero auxiliary encoder gradients, and zero future-input\ngradients passed. See ../goalaux-20260922/attempt2/contract-result.json. This is\nnot a full data-alignment audit or a GPU BF16/throughput check; those remain\npending. No new compressor has been fitted and no benefit has been measured.')
proposal.write_text(text)
shutil.copyfile(S/'attempt2/contract-result.json', REPORT/'goalaux-cpu-contract.json')
section = '''<section id="goalaux-contract"><h2>下一步：目标辅助损失的实现检查</h2>
<p>2026-09-22：仅在一个原训练样本与固定检查点上做 CPU FP32 检查。
关闭辅助损失时，原损失、随机数状态、80 组参数梯度及一次 AdamW 更新完全一致。
辅助目标能向编码器传递梯度，后续帧变化不影响初始表征，未来输入梯度为零。</p>
<p>首次检查因跨机器数据路径不同而失败，修正路径且保持原图像哈希检查后通过。
没有保存新训练检查点，没有测量收益。GPU BF16 一致性、训练成本和新场景验证仍待执行。
下一轮须同时验证目标信息改善与状态信息保留。</p>
<p><a href="goalaux-cpu-contract.json">完整检查结果与限制</a>。公网发布再次遇到 Sites 404，当前更新仅在本地报告。</p></section>'''
p = REPORT/'index.html'
html = p.read_text()
assert 'id="goalaux-contract"' not in html
assert '</main>' in html
p.write_text(html.replace('</main>', section+'\n</main>'))
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.targets=[]
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src') and value and not value.startswith(('#','https:','http:','data:','mailto:')):
                self.targets.append(value.split('#')[0])
parser=Links();parser.feed(p.read_text())
for target in parser.targets: assert (REPORT/target).is_file(),target
for path in REPORT.iterdir():
    if path.suffix in {'.html','.json','.py','.patch'}:
        data=path.read_text()
        for identifier in ['/home/zifanz4','/data02/','/vol13/','.csl.illinois.edu']:
            assert identifier not in data, (path,identifier)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(REPORT.iterdir()) if p.is_file() and p.name!='artifact-sha256.json'}
(REPORT/'artifact-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'recorded_at': now, 'local_links_checked':len(parser.targets), 'files_hashed':len(manifest), 'next_step':next_step},indent=2))
