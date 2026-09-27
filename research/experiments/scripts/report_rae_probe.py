"""Record every matched-baseline outcome and update the existing local report."""
from pathlib import Path
import datetime,hashlib,html,json,re,shutil
P=Path('/home/zifanz4/openwam-experiments');S=P/'studies/rae-baseline-20260922';R=Path('/data02/zifanz4/openwam-experiments/results/rae-baseline-20260922');O=P/'reports/layers-20260922'
now=datetime.datetime.now().astimezone().isoformat();d=json.loads((R/'probe-result.json').read_text());audit=json.loads((R/'probe-audit.json').read_text());assert audit['passed'];assert json.loads((R/'probe-exit.json').read_text())['exit_code']==0
for f in ['probe-result.json','probe-audit.json','probe-fit-complete.json','probe-launch.json','probe-exit.json','wan-selection.json','shift-diagnostic.json','shift-diagnostic-plan.json']:
 shutil.copy2(R/f,S/f)
next_step='Before original01:15:12CDTdeadline, prospectively fix a bounded training-noise control for the initial-image goal endpoint: same105training scenes, identical clean/noise mixtures and episode-CV across Wan native/LN and cached-DINO-derived raw/PCA/SVAE routes. Estimate all normalization/augmentation/calibration from training data only. Reuse existing old60 evaluation feature vectors and retain their exploratory label; do not use heldout means to correct scores. This distinguishes robustly accessible information from the current clean-trained linear-readout distribution shift. No new run is launched by this report; freeze inputs/code/resources first. Expired56/60 and old80 remain untouched.'
findings='''# Matched Wan baseline — exploratory result

Completed 2026-09-22 23:52 CDT. One A40, 135.6 seconds for the matched probe,
18 new ridge readouts, 105 training scenes and the entire old exposed 60-scene
layer-study cohort. Two Wan normalizations, both endpoints and all three noise
conditions retained. Existing DINO outputs were reused without inference or refits.
This does not complete the independent goalaux56/60 cohort, replicate full RAE,
isolate pretraining causally, or demonstrate closed-loop policy improvement.

The interface preflight passed on eight training frames from one episode:
strict official Wan weights, expected48x24x20 current-frame latent, exact causal
first-frame and cache-reset equality, finite pixel decoding. Nine-frame encoding
peak allocated memory1.964GB. The matched probe exited0 and released its GPU.
Separate saved-artifact audit checked36prediction conditions and18CV choices;
expanded-affine prediction residual3.55e-15, episode-metric residual1.82e-12.

## Observed error (lower is better)

| Representation | State clean | State noise.10 | Goal clean | Goal noise.10 |
|---|---:|---:|---:|---:|
| Wan48 native scale |0.2944|37.3384|0.2384|182.6310|
| Wan48 per-token LN |0.3007|26.5377|0.2371|314.5404|
| DINO raw768 |0.2169|0.7943|0.1197|0.3832|
| DINO PCA48 |0.1857|0.3682|0.1351|0.6421|
| DINO SVAE48, three-seed mean |0.1821|0.3736|0.1760|0.6014|

Full noise.04 and per-task results, both Wan normalization contrasts, and
unadjusted descriptive bootstrap intervals are retained in probe-result.json.
They are not confirmatory intervals on an independently collected new cohort.
The goal is first-close expert waypoint XY, not object pose or actual contact.
The state score weights translation and gripper groups equally.

## Interpretation and remaining confound

DINO-derived representations support markedly more robust clean-trained ridge
readouts in this setup. PCA48 versus Wan48 keeps the pooled visual dimension at192;
rawDINO uses3072 and is a different capacity comparison. PCA48 and SVAE48 have
similar short-state noise errors; rawDINO retains lower goal error than either
compressor. This suggests a compression tradeoff, not proof that SVAE is required.

The enormous Wan noise degradation requires an explanation before promoting any
99-percent relative reduction as a representation benefit. A separately declared
post-hoc diagnostic used only saved vectors and fixed coefficients, with no
correction, retraining or model inference. Wan noisy feature RMS in training
standard-deviation units is5.6–8.0 for state and9.6–26.5 for goal, versus roughly1
for clean. For rawDINO the corresponding ranges are1.40–1.42 and2.10–2.56.
Across both Wan normalizations, tasks and the noise.10 condition,96.7–99.93percent
of noise-induced output-shift energy is the task-wide mean component. This is
energy of noisy-minus-clean predictions, NOT the fraction of total prediction
error, and does not establish that information is irreversibly lost.

Thus current evidence supports a strong noise distribution shift/readout bias
mechanism. It does not establish RAE superiority under equally noise-adapted
training. A fair next control trains every route with identical training-only
noise augmentation and retains all clean/noisy outcomes. Heldout average offsets
must not be used to calibrate the readouts. Full RAE additionally needs a trained
pixel decoder; training only that decoder cannot improve an unchanged frozen
encoder's features. No original protocol or deadline was modified.
'''
(S/'FINDINGS.md').write_text(findings+'\n## Next uncertainty\n\n'+next_step+'\n')
for source,target in [('probe-result.json','rae-matched-probe.json'),('probe-audit.json','rae-probe-audit.json'),('matched-probe-protocol.json','rae-probe-protocol.json'),('shift-diagnostic.json','rae-shift-diagnostic.json')]:shutil.copy2(R/source,O/target)
shutil.copy2(S/'FINDINGS.md',O/'rae-findings.md')
pre=json.loads((R/'result.json').read_text());pre.pop('training_rows');pre['scope']='Training-only8frames from one episode; geometry/causality/resource preflight, no task benefit claim.';(O/'rae-interface-preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
labels={'Wan48_native_scale':'Wan VAE48 · 原生尺度','Wan48_per_token_layernorm':'Wan VAE48 · 逐 token LN','DINO_raw768':'DINO 原始768维','DINO_PCA48':'DINO + PCA48','DINO_SVAE48_seed_mean':'DINO + SVAE48（三种子平均）'}
rows=''
for name,label in labels.items():
 values=[d['matrix'][ep][condition][name]['E'] for ep in ['state','goal'] for condition in ['clean','noise0.04','noise0.10']]
 rows+='<tr><td>'+label+'</td>'+''.join('<td>'+format(v,'.4f')+'</td>' for v in values)+'</tr>'
section='''<section id="rae-matched-baseline"><h2>新增：像素 VAE 与 DINO 表征的配对读出对照</h2>
<p>完成于 23:52 CDT。105 个训练场景、原有 60 个评测场景；新增 18 个 Wan 线性读出器，DINO 结果直接复用。全部场景、噪声条件及两种 Wan 归一化均保留。独立数值审计通过，GPU 作业已结束。</p>
<p class="banner">这是在已看过的 60 个场景上追加的探索性对照。不是新的独立确认、完整 RAE 复现或闭环策略提升。误差越小越好。</p>
<div class="scroll"><table><thead><tr><th>表征</th><th>状态·干净</th><th>状态·噪声.04</th><th>状态·噪声.10</th><th>目标·干净</th><th>目标·噪声.04</th><th>目标·噪声.10</th></tr></thead><tbody>'''+rows+'''</tbody></table></div>
<p>DINO 路线在当前读出设置下更抗噪。PCA48 与 SVAE48 的短时状态误差接近，原始 DINO 的目标读出更好，说明压缩有任务相关的取舍。</p>
<p><strong>暂不把巨大相对差距宣传为 RAE 收益。</strong>已保存特征的事后诊断发现：Wan 加噪后偏离训练分布，噪声造成的输出变化中约 96.7–99.93% 的能量表现为任务内共同偏移；这不是总预测误差的解释比例，也不证明信息不可恢复。下一步须补相同训练噪声增强的读出控制，区分信息保留与读出器分布偏移，所有校准只使用训练集。</p>
<p>DINO 路径目前没有像素解码器，不能称为完整 RAE。Wan 接口检查已验证尺寸、首帧因果性与像素解码；没有重新运行旧80次策略评测，也没有评分未完成的56/60场景。</p>
<p><a href="rae-findings.md">解释与限制</a> · <a href="rae-matched-probe.json">全部结果及描述性区间</a> · <a href="rae-probe-protocol.json">预先固定方案</a> · <a href="rae-probe-audit.json">独立数值检查</a> · <a href="rae-shift-diagnostic.json">全部偏移诊断</a> · <a href="rae-interface-preflight.json">接口预检</a></p>
<p class="note">公网发布仍失败：Sites project not found。此更新目前只在本地网页。</p></section>'''
p=O/'index.html';s=p.read_text();assert 'id="rae-matched-baseline"' not in s;s=s.replace('<section id="goalaux-confirmation-pending">',section+'\n<section id="goalaux-confirmation-pending">',1);s=s.replace('<h1>目标位置读出改善，短期状态抗噪退化</h1>','<h1>新增像素 VAE 对照：先解释噪声下的差距</h1>');s=s.replace('本地报告更新：2026-09-22T19:21:18.931171-05:00','本地报告更新：'+now);p.write_text(s)
for name in ['rae-findings.md','rae-matched-probe.json','rae-probe-protocol.json','rae-probe-audit.json','rae-shift-diagnostic.json','rae-interface-preflight.json']:
 t=(O/name).read_text();assert not any(x in t for x in ['/home/zifanz4','/data02/','/vol13/','.csl.illinois.edu','GPU-d13929da'])
for link in re.findall(r'href="([^"#]+)"',s):
 if '://' not in link:assert (O/link).exists(),link
hashes=json.loads((O/'artifact-sha256.json').read_text())
for f in O.iterdir():
 if f.is_file() and f.name!='artifact-sha256.json':hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(O/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
header='# Matched Wan baseline completed — '+now+'\n\n'+findings.split('## Observed error')[0].split('\n\n',1)[1]+'\nNew result: DINO representations have much lower noisy clean-trained ridge error than Wan; very large Wan distribution shift and shared output bias explain why this is not yet a fair noise-adapted/fullRAE benefit claim. All conditions retained, independent arithmetic audit passed. See studies/rae-baseline-20260922/FINDINGS.md.\n\nNext: '+next_step+'\n\nPublic report remains unpublished: existing Sites project lookup returned project not found. Local HTML updated, no replacement site.\n\n---\n\n'
for f in [P/'STATUS.md',P/'NEXT_STEPS.md']:f.write_text(header+f.read_text())
f=P/'studies/layers-20260922/progress.json';p=json.loads(f.read_text());p['rae_followup_scope'].update(status='matched_probe_completed_audited_distribution_shift_control_next',matched_probe_result=str(S/'probe-result.json'),matched_probe_audit=str(S/'probe-audit.json'),probe_exit_code=0,full_RAE_reproduced=False,policy_gain_measured=False);p['last_selfcheck'].update(checked_at=now,scientific_state_changed=True,evidence=str(S/'FINDINGS.md'),owned_jobs_running=False);p['next_step']=next_step;f.write_text(json.dumps(p,indent=2)+'\n')
f=Path('/home/zifanz4/.local/state/openwam-selfcheck/last-reasoned.json');p=json.loads(f.read_text());p.update(updated_at=now,status='matched_wan_probe_complete_audited',next_step=next_step,evidence=str(S/'FINDINGS.md'),study_deadline='2026-09-23T01:15:12-05:00');f.write_text(json.dumps(p,indent=2)+'\n')
with (P/'selfchecks.md').open('a') as f:f.write('\n### Completion '+now+'\n\nMatched probe completed23:52:09 exit0;18newreadouts and36predictionconditions audited. All old60scenes retained; noise.04/.10 and clean, both Wan norms, originalcachedDINOcontrols. Preflight and probe total GPU jobs bounded, released. Large Wan noisy error traced descriptively to strong training-standardized input shift and task-wide output drift; no test correction applied. Full results/limits in studies/rae-baseline-20260922/FINDINGS.md. Next: '+next_step+'\nLocal report, links/hashes and public-artifact path scan passed. Existing Sites lookup failed: project not found; no deployment.\n')
print(json.dumps({'local_report':str(O/'index.html'),'artifacts':len(hashes),'next_step':next_step,'recorded_at':now}))
