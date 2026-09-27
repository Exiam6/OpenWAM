"""Archive completed bounded run; report stored results, no rescoring/inference."""
from pathlib import Path
import datetime,json,hashlib,shutil,re
P=Path('/home/zifanz4/openwam-experiments');R=Path('/data02/zifanz4/openwam-experiments/results/confirmation-20260923');S=P/'studies/confirmation-20260923';H=P/'reports/layers-20260922';now=datetime.datetime.now().astimezone().isoformat();q=json.loads((R/'result.json').read_text());window=json.loads((R/'window.json').read_text())
assert datetime.datetime.fromisoformat(q['completed_at'])<datetime.datetime.fromisoformat(window['deadline']);assert json.loads((R/'features/exit.json').read_text())['exit_code']==0
assert q['episodes']==60 and q['readout_refits']==0
for name in ['result.json','identity-audit.json','sample-manifest.json','runtime-compatibility.json','resource-wait-exit.json']:
 shutil.copy2(R/name,S/name)
for sub in ['features','feature-attempt-l40s','feature-attempt-rainier-startup']+[f'fresh/{t}' for t in q['task_order']]:
 (S/sub).mkdir(parents=True,exist_ok=True)
 for name in ['launch.json','started.json','progress.json','manifest.json','hardware-parity.json','exit.json','run.log']:
  if (R/sub/name).exists():shutil.copy2(R/sub/name,S/sub/name)
counts={};attempts={}
for t in q['task_order']:
 m=json.loads((R/'fresh'/t/'manifest.json').read_text());assert m['complete'];counts[t]=len(m['accepted']);attempts[t]=len(m['attempts']);assert json.loads((R/'fresh'/t/'exit.json').read_text())['exit_code']==0
next_step='New4hwindow closed15:38:51CDT with all60scenes and frozen scoring completed12:20:42. No new experiment, refit, inference, rescore or automaticdeadlineextension. Preserve old56unscored/old80unchanged and allnewfailures. Independent numerical audit scripts/confirmation/audit.py was prepared but notexecuted beforedeadline because toolapproval service exhaustedusage; its original deadline guard is preserved. Next required validation is a separately authorized bounded CPUartifact audit of rawlabels,135savedpredictions and99pairedintervals; then package minimal representation-baseline and component-reporting contribution. Current completed-run numbers remain provisional pendingthat audit; no fullRAE/closedloop claims.'
progress=json.loads((S/'progress.json').read_text());progress.update(status='window_closed_scoring_complete_independent_audit_pending',updated_at=now,accepted_counts=counts,attempt_counts=attempts,feature_extracted_scenes=60,feature_host='rainier',feature_gpu_index=4,hardware_parity_passed=True,feature_exit_code=0,completed_at=q['completed_at'],all_collectors_exit_zero=True,study_gpu_workers_remaining=0,resource_waiter='finished afterdispatch; no longeractive',next_step=next_step,joint_primary_passed_by_frozen_scorer=q['joint_primary_passed'],independent_numerical_audit_completed=False,closure_integrity=json.loads((S/'closure-integrity.json').read_text()))
(S/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
primary=q['co_primary'];g=primary['goal'];st=primary['state'];gr=st['gripper'];rawgr=q['paired']['state']['noise0.10']['Wan48_native_scale']['DINO_raw768']['gripper']
findings=f'''# New independent representation confirmation — completed, numerical audit pending

The frozen scorer completed {q['completed_at']}, exit0, inside the unchanged
15:38:51CDT deadline. Three collectors ended12:11:44/12:14:25/12:20:06, all exit0.
60expert-feasible scenes:20per task,78consecutive attempts (20/22/36),18expert-infeasible
attempts retained.12fixed state frames/scene,5040clean/noisy feature rows. No readout
refit, heldout recalibration, winner change, old80rerun or expired56/60 reuse.

## Stored primary outcomes — not yet independently numerically audited

PCA48 versus native Wan48, same192pooled visual dimensions and frozen readouts
trained using matched noise augmentation. Primary condition sigma0.10. Paired
scene bootstrap2000draws,20scenes/task, seed20260922; repeated frames/noise/SVAEseeds
are not independent observations.

| Endpoint | Relative change | Paired95% interval |
|---|---:|---:|
| Initial first-close XY | {g['relative_change_percent']:.2f}% | [{g['paired95'][0]:.2f}, {g['paired95'][1]:.2f}]% |
| Short-state E | {st['relative_change_percent']:.2f}% | [{st['paired95'][0]:.2f}, {st['paired95'][1]:.2f}]% |
| Translation component | {st['translation']['relative_change_percent']:.2f}% | [{st['translation']['paired95'][0]:.2f}, {st['translation']['paired95'][1]:.2f}]% |
| Gripper component | {gr['relative_change_percent']:.2f}% | [{gr['paired95'][0]:.2f}, {gr['paired95'][1]:.2f}]% |

The frozen scorer flags both co-primary absolute-difference intervals below0.
This is a stored-run result, not a completed independent audit. Gripper's interval
crosses0: no gripper improvement or noninferiority claim. Alltask goal/state point
changes favorPCA, butstate change varies substantially (-9.96/-58.74/-42.70%).
RawDINO grippererror increases{rawgr['relative_change_percent']:.2f}%
[{rawgr['paired95'][0]:.2f},{rawgr['paired95'][1]:.2f}]%, despite better totalstateE.
Both compact routes retain that tradeoff information; no uniform bestrepresentation
or causal pretraining explanation is established.

## Full matrix from saved outputs

'''
for kind,cons in q['matrix'].items():
 findings+='\n### '+kind+'\n\n| Route | Clean E | Noise.04 E | Noise.10 E |\n|---|---:|---:|---:|\n'
 for name in cons['clean']:
  findings+=f"| {name} | {cons['clean'][name]['E']:.6f} | {cons['noise0.04'][name]['E']:.6f} | {cons['noise0.10'][name]['E']:.6f} |\n"
findings+='''
## Provenance and remaining checks

Pipeline identity gate passed:60episodes/720frames,0recordedpriorencodedhashoverlap.
All370frozen input/code hashes were verified unchanged after the window. This is
byte integrity, not numerical validation. Independent audit.py exists but was not
executed beforedeadline; no audit.json exists. Do not remove its deadline guard or
silently relabel this as an audited confirmation. Toolautomaticapproval failed due
to usage exhaustion during the12:08report update; the action didnotexecute. Tools
becameavailable at15:49. Backgroundboundedexperiments completed normally at12:20.

L40S trainingfixture: WanRMS.10603failedfixed.05tolerance; DINOexact. A40dispatch
onrainier initially failedbeforePython becausehostglibc2.31wasincompatible. Original
Python/package/weights were retained; aprivatecm001glibc2.35loader/librarybundle
allowedstartup, without changing systemlibraries. A40trainingparity thenpassed:
Wanexact, DINOmaxRMS.045968andmaxcoordinate.224641withinoriginal.05/.5bounds.
Both failures retained, zero newscene inference beforeparitypass. Only3collectors+
1featureGPU concurrent. No studyGPUworkers remain at15:49.

The implication, if the independent audit confirms these saved outputs, is that
compact pretrained visual features are a useful low-cost offline control-readout
baseline against pixelVAE latents under the matched training/noise protocol. It is
not fullRAE (no trainedpixeldecoder), not closedloopcontrol, not truecontact, not
unseen-task or unseen-noise generalization. Architecture/objective/pretraining
remainconfounded. RawDINO has3072pooledinputs versus192forcompact/Wan controls.

Next: authorize a bounded CPUartifact audit, then contribute the smallest useful
reproducible baseline comparison with train-only normalization, scene provenance,
and separate translation/gripper reporting. Do not change the defaultpolicy or
claim RAE/policygain on the basis of these offline metrics.
'''
(S/'FINDINGS.md').write_text(findings)
header=f'''# Independent60scenes completed; window closed; audit pending — {now}

All60newscenes collected (20/20/20;78attempts); all3collectors exit0. Frozen feature/scoring job completed12:20:42 exit0, wellbefore unchanged15:38:51CDTdeadline. No ownedstudyGPUworker remains; cm009GPU1/2/3 andrainierGPU4 checked1MiB/0%/ECC0at15:49. No newexperiment afterdeadline.

Stored scorerflags co-primarypass: PCA48vsnativeWan noise.10goalerror-73.05% (paired95[-79.60,-65.72]),stateE-39.11%([-48.70,-27.30]). Gripper-12.33%([-26.12,+5.91])remainsuncertain; rawDINOgripper+79.65%([42.65,126.73]). Fullmatrix andtaskdifferences retained. These are pending independent numericalaudit, not finalauditedbenefit or fullRAE/closedloopclaims. 370frozenfilehashesverifiedunchanged; pipelineidentity/paritypassed.

RecoveredA40featurepipeline12:07usingprivateglibccompatibility; originalfailedL40Sparity andpre-Pythonrainierstartupattempt retained. Automaticapprovalservice usagefailure prevented12:08records/reportwrite andsubsequentsupervision until15:49. Backgroundjobscompletednormally. audit.py was preparedbutnotexecutedbeforedeadline; itsguardunchanged. See studies/confirmation-20260923/FINDINGS.md.

Next: {next_step}

PublicSiteslookup15:51stillfailed404projectnotfound. LocalHTMLandartifactsupdated; no newsiteor publicsuccessclaim.

---

'''
for name in ['STATUS.md','NEXT_STEPS.md']:
 f=P/name;old=f.read_text();f.write_text(header+old.split('\n---\n\n',1)[1])
with (P/'selfchecks.md').open('a') as f:f.write('\n'+header.split('\n---\n')[0].replace('# Independent','## Independent')+'\n\nProcessed delayed12:15trigger at15:49; latestmonitor15:45. Requiredold/newstatefilesread andno pause. Newwindowclosurehonored; norescoring/inference/refit. Restored last-reasonedandarchivedoutputs afterautomaticreviewservice recovered.\n')
state=Path('/home/zifanz4/.local/state/openwam-selfcheck');(state/'last-reasoned.json').write_text(json.dumps({'time':now,'status':progress['status'],'trigger':'delayed2026-09-23T12:15:01.777102-05:00','next_step':next_step,'independent_audit_completed':False},indent=2)+'\n')
(S/'WINDOW_CLOSURE.md').write_text(header.split('\n---\n')[0]+'\n')
(S/'publication.json').write_text(json.dumps({'time':now,'project_id':'appgprj_6ab06a7c072481919a12cc100f8e9fa3','published':False,'error':'SitesConnectorError: Sites project not found','error_code':404,'local_report_updated':True},indent=2)+'\n')
# User-readable report uses saved values only.
table=''
for kind,cons in q['matrix'].items():
 for name,x in cons['clean'].items():table+=f"<tr><td>{kind}</td><td>{name}</td><td>{x['E']:.4f}</td><td>{cons['noise0.04'][name]['E']:.4f}</td><td>{cons['noise0.10'][name]['E']:.4f}</td></tr>"
section=f'''<section id="confirmation-20260923"><h2>新60场景评分完成 · 独立审计待完成</h2><p>计算于9月23日12:20完成，早于15:38截止时间。三任务各20个有效场景，共保留78次尝试；采集与评分均退出0。四张卡已释放。此处是已保存评分结果，<strong>独立数值审计尚未执行</strong>，不能标记为已审计结论。</p><table><tr><th>PCA48 相对 Wan48（噪声0.10）</th><th>误差变化</th><th>场景配对95%区间</th></tr><tr><td>初始目标XY</td><td>−73.05%</td><td>[−79.60%, −65.72%]</td></tr><tr><td>短期状态E</td><td>−39.11%</td><td>[−48.70%, −27.30%]</td></tr><tr><td>平移分量</td><td>−40.13%</td><td>[−49.87%, −27.76%]</td></tr><tr><td>夹爪分量</td><td>−12.33%</td><td>[−26.12%, +5.91%]</td></tr></table><p>冻结评分程序标记两个预设主指标通过，但夹爪区间跨零，不能声称可靠改善或非劣。原始DINO的夹爪误差反而增加79.65%（区间[42.65%,126.73%]），尽管总体状态误差更低。PCA、SVAE与Wan均为192维池化读出输入，原始DINO为3072维。</p><img src="confirmation-20260923-intervals.png" alt="待独立审计的保存结果：PCA目标与状态误差区间低于零，夹爪区间跨零，原始DINO夹爪误差升高。"><details><summary>全部路线、干净与噪声条件</summary><div class="scroll"><table><tr><th>端点</th><th>路线</th><th>干净E</th><th>噪声.04 E</th><th>噪声.10 E</th></tr>{table}</table></div></details><p><strong>检查与限制：</strong>原生规划器、场景来源和原定硬件容差检查通过；370个冻结文件校验一致。L40S上的Wan数值偏差、rainier的Python启动失败均已保留。私有兼容库修复后，A40上的Wan与参考完全相同，DINO最大标准化RMS0.04597满足原定0.05容差。模型、读出与实验规则未改。</p><p>12:08的记录写入因自动审批服务额度耗尽而未执行，15:49访问恢复；后台计算已按界限完成。未在截止后重启实验或跳过审计脚本的时间限制。下一步是有界CPU独立审计，再整理最小可复现贡献。尚无完整RAE、真实接触或闭环策略收益结论。</p><p><strong>网页状态：</strong>15:51公网服务仍返回项目不存在（404）；本地报告已更新。</p><p><a href="confirmation-20260923-result.json">全部保存结果</a> · <a href="confirmation-20260923-predictions.npz">保存预测</a> · <a href="confirmation-20260923-findings.md">结论与限制</a> · <a href="confirmation-20260923-protocol.json">冻结方案</a> · <a href="confirmation-20260923-progress.json">终态</a> · <a href="confirmation-20260923-ampere-parity.json">A40一致性</a></p></section>'''
h=H/'index.html';html,n=re.subn(r'<section id="confirmation-20260923">.*?</section>',section,h.read_text(),count=1,flags=re.S);assert n==1;h.write_text(html)
for src,dst in [(S/'progress.json','progress.json'),(R/'result.json','result.json'),(R/'predictions.npz','predictions.npz'),(S/'FINDINGS.md','findings.md'),(R/'features/hardware-parity.json','ampere-parity.json')]:shutil.copy2(src,H/('confirmation-20260923-'+dst))
print('Archived completedrun,closedfixedwindow,independentauditpending;public404recorded')
