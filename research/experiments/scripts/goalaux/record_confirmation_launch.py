"""Archive verified progress and portable public report; no experiment replay."""
import datetime,json,hashlib,re
from pathlib import Path
from html.parser import HTMLParser
R=Path('/home/zifanz4/openwam-experiments');S=R/'studies/goalaux-20260922';L=R/'studies/layers-20260922';P=R/'reports/layers-20260922';STATE=Path('/home/zifanz4/.local/state/openwam-selfcheck')
now=datetime.datetime.now().astimezone().isoformat();check=json.loads((S/'current-check.json').read_text());audit=json.loads((S/'readouts/audit.json').read_text());native=json.loads((S/'native-preflight/result.json').read_text());freeze=json.loads((S/'confirmation/freeze.json').read_text());assert audit['passed'] and native['passed'];assert not check['scored'];counts=check['accepted_counts'];total=sum(counts.values());assert check['collection_exit'] is None and check['confirmation_exit'] is None;assert 'wam-goalaux-collect-v2' in check['jobs'] and 'wam-goalaux-confirm' in check['jobs']
next_step='Read G2 results/goalaux-20260922/fresh-v2 and confirmation actual logs/exit codes. Bounded dependent job already waits for all60 scenes, then identity/parity/extraction/scoring/independent CPU audit; do NOT create a duplicate launcher. Next uncertainty: whether goal auxiliary improves noisy initial-waypoint information while preserving both clean/noisy state information on independent scenes. Joint ridge criteria fixed; no benefit claim until complete audited result. Collection deadline19:15:55CDT Sep22 and study deadline01:15:12CDT Sep23 unchanged. Any further infrastructure failure stops; preserve artifacts and repair only with recorded evidence, never replay completed work.'
status='new_goalaux_confirmation_collecting_and_bounded_scoring_queued'
publication={'checked_at':now,'operation':'Sites get_site(existing project)','result':'SitesConnectorError: Sites project not found','deployment_attempted':False,'public_version_updated':False,'previous_known_public_version':26,'local_report_updated':True}
(S/'publication-confirmation-launch.json').write_text(json.dumps(publication,indent=2)+'\n')
readout={'completed':True,'exit_code':0,'training_episodes':105,'training_clips':1260,'ridge_endpoint_fits':36,'ridge_groups':54,'mlp_fits':36,'independent_saved_artifact_audit_passed':True,'new_confirmation_scored':False}
fresh={'as_of':check['as_of'],'accepted_counts':counts,'accepted_total':total,'distinct_attempts_completed':check['attempts'],'planned_accepted_total':60,'original_failed_setups':1,'initial_failure':'missing old-root collision asset paths caused planner fallback, stopped before any actions/trajectory','documented_repair':'only two copied YAML asset prefixes; same native geometry, physics, planner, cameras, seed sequence, originaldeadline','current_actual_planner':'CuroboPlanner; checked in both arms per scene','collection_deadline':check['collection_deadline'],'collection_exit_code':check['collection_exit'],'confirmation_job':'running dependency wait without GPU; frozen hashes and 69 readout interface checks passed','confirmation_exit_code':check['confirmation_exit'],'new_scores_observed':False,'frozen_confirmation_files':len(freeze['sha256'])}
progress=json.loads((L/'progress.json').read_text());progress.update(updated_at=now,status=status,next_step=next_step,goal_aux_readouts=readout,goal_aux_confirmation=fresh);progress['goal_aux_training']['fresh_confirmation_scored']=False;progress['public_report_update']={'status':'blocked','last_checked_at':now,'reason':publication['result'],'local_report':str(P/'index.html')};(L/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');(S/'progress.json').write_text(json.dumps({'updated_at':now,'status':status,'readouts':readout,'confirmation':fresh,'next_step':next_step,'study_deadline':progress['deadline']},indent=2)+'\n')
record=json.loads((STATE/'last-reasoned.json').read_text());record.update(updated_at=now,status=status,next_step=next_step,deadline=progress['deadline'],actual_job_snapshot_time=check['as_of']);(STATE/'last-reasoned.json').write_text(json.dumps(record,indent=2)+'\n')
entry=f'''# Selfcheck — new matched confirmation collecting ({total}/60) — {now}

Snapshot {check['as_of']}. Resumed; no pause markers. Original study window unchanged:
13:15:12CDT Sep22 to01:15:12CDT Sep23. Delayed16:15/16:45triggers coalesced in this
active turn; no new timer, subagent, completed experiment replay, or task termination.
Old80policy evaluations, completed layer results, and six matched codec fits unchanged.

New readout fits exit0:105original train episodes/1260clips,54ridge groups and36MLPs.
Independent saved-data/statistics/folds/normal-equation/NumPy-forward audit passed;
maximum relative ridge residual {audit['max_ridge_normal_equation_relative_residual']:.3g}.
Training-only interface checks cover69new/anchor state and goal model paths.

Copied isolated native environment to Group2 because no idle ECC-clean Group1L40S.
273native source/config hashes verified. Empty-scene native rendering/CUDA binding
passed exit0. First seed800000 then stopped before actions: copied robot YAML still
referred to Group1 collision assets and triggered Mplib fallback. Actual-backend guard
prevented collecting altered data. Initial failure/exit1 retained. Permitted recorded
failed-work repair changed only two copied YAML absolute path prefixes, verifying
URDF/collision geometry hashes; no completed expert trajectory or fit repeated.
COLLECTION_PATH_REPAIR.md records the continuation before it ran. Same seed800000
initialization repaired under fresh-v2; real CuroboPlanner and render PCI pass each scene.

As of snapshot: {counts}; completed distinct candidates{check['attempts']},
current collector exitNone. One L40S used,4threads; GPU snapshot {check['gpu']}.
Collector originaldeadline {check['collection_deadline']} remains unchanged.
Bounded confirmation job is already queued, no GPU during dependency wait;58files
frozen before scores. It requires all60scenes, content identity against210previous
scenes, and exact encoder/codec zero-noise parity. One-hour post-allocation cap,
original studydeadline; no refits. No new confirmation metrics have been observed.

Next: {next_step}

Publication retry: existing Sites project stillnotfound. Local HTML and public-safe
artifacts updated; publicversion26 NOTupdated. Existing Sites account clarification
pending; no replacementsite or duplicate timer created.
'''
for filename in ['STATUS.md','NEXT_STEPS.md']:
 p=R/filename;p.write_text(entry+'\n---\n\n'+p.read_text())
with (R/'selfchecks.md').open('a') as f:f.write('\n\n'+entry)
public={'as_of':check['as_of'],'state':status,'models_completed':6,'training_episodes':105,'readout_fit_and_independent_audit_passed':True,'accepted_counts':counts,'accepted_total':total,'planned_scenes':60,'current_collector_exit_code':None,'original_setup_failure_retained':True,'repair_only_copied_asset_path_prefixes':True,'source_files_verified':273,'native_render_and_actual_planner_checks_passed':True,'collection_deadline':check['collection_deadline'],'study_deadline':progress['deadline'],'confirmation_job':'bounded dependency wait; no GPU allocated while waiting','confirmation_files_frozen':len(freeze['sha256']),'joint_primary':'noisy goal gain plus clean/noisy state and clean goal preservation; all tasks reported','new_confirmation_scored':False,'policy_benefit_established':False,'public_publication':'existing project lookup failed; this update local only'}
(P/'goalaux-confirmation-status.json').write_text(json.dumps(public,indent=2)+'\n');(P/'goalaux-readout-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(P/'goalaux-collection-repair.md').write_text('''# Native collector portability repair

An isolated copy of the native environment passed empty-scene rendering and GPU
binding checks. At the first candidate, two robot YAML files still referenced the
old filesystem root. Missing collision files activated the wrapper's fallback
planner. The added actual-backend assertion stopped initialization before any
expert action, saved trajectory, or confirmation score. Original failure retained.

The prospective scientific protocol permits documented repairs of failed work.
The continuation replaces only the two copied absolute asset path prefixes;
original robot URDF and collision geometry hashes match. Physics, camera, planner,
rendering, seeds, acceptance rules and collection deadline remain unchanged.
The interrupted initialization is repaired at the same candidate; no completed
expert-feasibility attempt is repeated or discarded. Both actual planner classes
and rendering device are checked on every new scene. This is an environment
portability finding, not evidence that the representation improves control.
''')
section=f'''<section id="goalaux-confirmation-pending"><h2>新实验：目标辅助监督能否同时保住两类信息</h2>
<p>状态核查：{check['as_of']}。两组各三个相同种子的模型已训练完成；新增读出器只使用原来的105个训练回合，保存系数与推理一致性独立检查通过。</p>
<p><strong>新场景采集：{total}/60</strong>。每任务20个，按冻结种子顺序取专家可行场景。当前尚无新确认分数，也没有策略收益结论。</p>
<p>第一个场景曾因复制后的资产路径触发规划器回退，在任何动作前被检查拦住。已保留失败，只修复两份配置的路径；实际 Curobo 规划器和渲染设备检查现已通过。采集截止仍为19:15:55，研究总截止仍为次日01:15:12（CDT）。</p>
<p>评分程序已冻结并等待完整数据。它会先与之前210个回合核对数据身份，再做零噪声一致性检查；以场景为独立单位比较目标读出和状态读出。必须同时满足预设收益与保持标准。</p>
<p><a href="goalaux-confirmation-status.json">当前状态</a> · <a href="goalaux-readout-audit.json">读出器独立检查</a> · <a href="goalaux-collection-repair.md">采集失败及修复记录</a> · <a href="goalaux-matched-protocol.md">冻结科学协议</a></p>
<p class="note">本轮 Sites 查询仍返回项目不存在；当前更新仅在本地网页，原公网版本尚未更新。</p></section>'''
p=P/'index.html';html=p.read_text()
if 'id="goalaux-confirmation-pending"' in html:html=re.sub(r'<section id="goalaux-confirmation-pending">.*?</section>',section,html,flags=re.S)
else:
 pos=html.index('<div class="banner">');html=html[:pos]+section+'\n'+html[pos:]
html=re.sub(r'本地报告更新：[^。]+。',f'本地报告更新：{now}。',html,count=1);p.write_text(html)
train=json.loads((P/'goalaux-training-status.json').read_text());train.update(as_of=check['as_of'],state=status,confirmation_scenes_completed=total,readouts_independently_audited=True);(P/'goalaux-training-status.json').write_text(json.dumps(train,indent=2)+'\n')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.paths=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v and not v.startswith(('#','https:','http:','data:','mailto:')):self.paths.append(v.split('#')[0])
links=Links();links.feed(html)
for q in links.paths:assert (P/q).is_file(),q
for f in P.iterdir():
 if f.suffix in ['.html','.json','.md','.py','.patch']:
  text=f.read_text()
  for forbidden in ['/home/zifanz4','/data02/','/vol13/','.csl.illinois.edu','GPU-dee01971']:assert forbidden not in text,(f,forbidden)
(P/'artifact-sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.iterdir()) if f.is_file() and f.name!='artifact-sha256.json'},indent=2)+'\n');print(json.dumps({'status':status,'fresh_scenes':total,'local_report_links_checked':len(links.paths),'published':False},indent=2))
