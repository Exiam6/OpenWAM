"""Record new engineering evidence and mirrored job state; never rerun a fit."""
import datetime,hashlib,json,re,shutil
from pathlib import Path
from html.parser import HTMLParser
R=Path('/home/zifanz4/openwam-experiments');S=R/'studies/goalaux-20260922';L=R/'studies/layers-20260922';REPORT=R/'reports/layers-20260922';STATE=Path('/home/zifanz4/.local/state/openwam-selfcheck')
now=datetime.datetime.now().astimezone().isoformat();result=json.loads((S/'gpu-preflight-v2/result.json').read_text());assert result['passed'];assert (S/'gpu-preflight-v2/exit-code.txt').read_text().strip()=='0'
train=S/'matched-training';complete=json.loads((train/'training-complete.json').read_text()) if (train/'training-complete.json').exists() else None
progress=json.loads((train/'training-progress.json').read_text()) if (train/'training-progress.json').exists() else {};count=len(progress.get('completed',[]));exitcode=(train/'training-exit-code.txt').read_text().strip() if (train/'training-exit-code.txt').exists() else None
finished=complete is not None and exitcode=='0';status='six_matched_goal_aux_fits_complete_pending_new_confirmation' if finished else 'goal_aux_matched_training_running' if exitcode is None else 'goal_aux_matched_training_failed'
next_step=('Six final checkpoint/paired-sampler audits already passed: do not rerun them. Prepare/freeze/check fresh-collection code and data-identity gate for the already fixed800000/801000/802000ranges. Fit/readout hashes must be frozen before new confirmation scoring. No completed evaluation replay, no loss/seed sweep; the previous60scenes are exposed development. Original study deadline01:15:12CDT Sep23 unchanged.' if finished else 'Await the bounded six-fit launcher exit code, then verify all checkpoint hashes and paired sample sequences. Next uncertainty is whether goal supervision improves new-scene readout without state regression; only the fixed prospective cohort can answer it. Never restart completed fits or score the old60as fresh. Original deadline01:15:12CDT Sep23 unchanged.')
publication={'recorded_at':now,'lookup_observed_during':'16:00 selfcheck processing','result':'Sites get_site NOT_FOUND404','deployment_attempted':False,'existing_public_version_unchanged':True,'local_report_updated':True}
(S/'publication-gpu-training.json').write_text(json.dumps(publication,indent=2)+'\n')
p=L/'progress.json';data=json.loads(p.read_text());data.update(updated_at=now,status=status,next_step=next_step,goal_aux_model_training_started=True,goal_aux_training={'models_completed':count,'planned':6,'exit_code':exitcode,'fixed_additional_updates_per_model':2000,'state':status,'fresh_confirmation_scored':False,'protocol_sha256':hashlib.sha256((S/'matched-protocol.json').read_bytes()).hexdigest()},goal_aux_gpu_preflight={'passed':True,'original_exit':1,'continuation_exit':0,'zero_weight_parity':'exact and not replayed','pooling_fix':'deterministic block mean; equivalents checked','timing_scope':'training-resident3episodecompute only','path':str(S/'gpu-preflight-v2/result.json')});data['public_report_update']={'status':'blocked','last_checked_at':now,'reason':'Sites get_site404','local_report':str(REPORT/'index.html')};p.write_text(json.dumps(data,indent=2)+'\n')
record=json.loads((STATE/'last-reasoned.json').read_text());record.update(updated_at=now,status=status,next_step=next_step,deadline=data['deadline'],actual_training_start=json.loads((train/'training-started.json').read_text())['started_at']);(STATE/'last-reasoned.json').write_text(json.dumps(record,indent=2)+'\n')
base_ms=result['profiles']['reconstruction_only']['median_seconds_per_update']*1000;aux_ms=result['profiles']['goal_auxiliary']['median_seconds_per_update']*1000
audit_passed=(train/'training-audit.json').exists() and json.loads((train/'training-audit.json').read_text())['passed']
if finished:assert audit_passed
data['goal_aux_training']['independent_artifact_audit_passed']=audit_passed
p=L/'progress.json';p.write_text(json.dumps(data,indent=2)+'\n')
entry=f'''# Selfcheck — GPU contract passed; matched intervention {count}/6 fits — {now}

No pause marker; fixed study deadline2026-09-23T01:15:12-05:00 unchanged.
Completed layer-study results and old80policy evaluations are unchanged.
GPU BF16 native/zero-aux loss,RNG,80gradients and oneAdamW update were exact.
The first actual GPU run then failed at nondeterministic adaptive-pool backward.
A recorded continuation replaced only that operation with equivalent fixed-grid
means, passed CPU forward/gradient and GPU forward/deterministic backward checks,
and completed within the original10minute preflight window. Originalfailure
retained; the already passed zero-weight check was not replayed. Model-compute
medians{base_ms:.2f}ms baseline,{aux_ms:.2f}ms auxiliary; this excludes full data IO.
Occupied and ECC-error devices were excluded before execution; no task terminated.

Separate prospective protocol MATCHED_PROTOCOL.md frozen before any retained fit:
L12 baseline continuation versus goalaux0.1;3matched seeds42/43/44;2000updateseach;
training-only105episodes; auxiliary normalization matches the frozen readout.
Its aligned BF16 gradient check passed before fitting. Currentcompleted{count}/6,
launcherexit={exitcode}; independentcheckpoint/sampling audit={audit_passed}; singleidleL40S,4threads,30mincap. No performance benefit
or new confirmation score has been observed. Goal improvement and state retention
must both pass the frozen joint criteria on NEW scenes, not the old60episodes.

Next: {next_step}

Public Sites lookup still404; local report updated; originalpublicversion26 stale.
Existing single15mincron unchanged; no new timer or claim of timely idle wakeup.
'''
for name in ['STATUS.md','NEXT_STEPS.md']:
 p=R/name;p.write_text(entry+'\n---\n\n'+p.read_text())
with (R/'selfchecks.md').open('a') as f:f.write('\n\n'+entry)
shutil.copyfile(S/'gpu-preflight-v2/result.json',REPORT/'goalaux-gpu-preflight.json');shutil.copyfile(S/'MATCHED_PROTOCOL.md',REPORT/'goalaux-matched-protocol.md')
public_status={'as_of':now,'models_completed':count,'planned_models':6,'updates_per_model':2000,'training_exit_code':exitcode,'state':status,'independent_training_artifact_audit_passed':audit_passed,'fresh_confirmation_scored':False,'policy_benefit_established':False,'deadline':'2026-09-23T01:15:12-05:00','public_deployment':'blocked404;localonly'}
(REPORT/'goalaux-training-status.json').write_text(json.dumps(public_status,indent=2)+'\n')
section=f'''<section id="goalaux-gpu-training"><h2>GPU 检查与等预算训练对照</h2>
<p>更新：{now}。GPU BF16 下关闭辅助项的损失、随机数、80 组梯度和一次更新完全一致。
修复不支持确定性反传的自适应池化后，等价分块平均通过数值与梯度检查。原失败保留。</p>
<p>单卡预检每步约 {base_ms:.1f} ms（原目标）与 {aux_ms:.1f} ms（辅助目标），仅为驻留特征下的模型计算成本。</p>
<p><strong>新对照已启动：{count}/6 个模型完成</strong>，两组 × 三个相同种子，各固定续训 2,000 步。独立检查点与采样一致性审计：{audit_passed}。
辅助头使用与评测相同的表征归一化。训练使用原 105 个训练场景，尚未评分新的确认场景。
没有新增收益结论；必须同时满足目标读出改善和状态读出保持。原截止时间不变。</p>
<p><a href="goalaux-gpu-preflight.json">GPU 检查详情</a> · <a href="goalaux-matched-protocol.md">冻结的科学协议</a> · <a href="goalaux-training-status.json">训练状态</a></p>
<p>Sites 返回 404，当前更新仅在本地报告，原公网版本未更新。</p></section>'''
p=REPORT/'index.html';html=p.read_text()
if 'id="goalaux-gpu-training"' in html:html=re.sub(r'<section id="goalaux-gpu-training">.*?</section>',section,html,flags=re.S)
else:html=html.replace('</main>',section+'\n</main>')
p.write_text(html)
class Links(HTMLParser):
 def __init__(self):super().__init__();self.paths=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v and not v.startswith(('#','https:','http:','data:','mailto:')):self.paths.append(v.split('#')[0])
parser=Links();parser.feed(html)
for path in parser.paths:assert (REPORT/path).is_file(),path
for f in REPORT.iterdir():
 if f.suffix in ['.html','.json','.md','.py','.patch']:
  text=f.read_text()
  for forbidden in ['/home/zifanz4','/data02/','/vol13/','.csl.illinois.edu','GPU-dee01971']:
   assert forbidden not in text,(f,forbidden)
(REPORT/'artifact-sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(REPORT.iterdir()) if f.is_file() and f.name!='artifact-sha256.json'},indent=2)+'\n')
print(json.dumps({'status':status,'completed_models':count,'recorded_at':now,'local_links_checked':len(parser.paths)},indent=2))
