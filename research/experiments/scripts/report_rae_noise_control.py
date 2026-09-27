from pathlib import Path
import datetime,hashlib,json,re,shutil
P=Path('/home/zifanz4/openwam-experiments');R=Path('/data02/zifanz4/openwam-experiments/results/rae-noise-control-20260922');S=P/'studies/rae-noise-control-20260922';O=P/'reports/layers-20260922';now=datetime.datetime.now().astimezone().isoformat()
d=json.loads((R/'result.json').read_text());a=json.loads((R/'audit.json').read_text());assert a['passed'] and json.loads((R/'exit.json').read_text())['exit_code']==0
for name in ['result.json','audit.json','launch.json','exit.json','asset-check.json','hardware-parity.json','feature-complete.json','fit-complete.json','selection.json']:
 shutil.copy2(R/name,S/name)
next_step='Within the unchanged01:15:12CDTdeadline, prospectively freeze the short-horizon state counterpart of the equal-noise control: identical105training episodes and12fixed current frames per scene, all frozen representations, noise0.04/.10 draws, episode-disjoint CV and alpha_eff=7*alpha relative to the original12-frame-per-scene fit (not84*alpha). Reuse already extracted initial-frame noisy vectors from this run and all original clean/evaluation vectors; only encode missing training-frame variants. Retain all translation/gripper, clean/noisy, task and seed results. This resolves whether the goal-information advantage also preserves short-state information. No world/action-policy gain is claimed; no expired56/60 or old80 replays. No launch follows automatically from this report; freeze and validate code/resources first.'
findings='''# Equal-noise goal-readout control

Completed 2026-09-23 00:05:20 CDT, exit0; one previously idle A40 released.
Training105episodes with7variants each: cached clean plus3independent draws at
sigma.04 and3at sigma.10. All21newtask-specific ridge fits used identical
scene-disjoint fivefold CV, training-only statistics and alpha grid. Ridge
penalty was scaled by7to preserve regularization per independent scene; a
synthetic repeated-clean fixture verified unchanged predictions (2.44e-13).
No encoder/reducer updates. Existing60scene evaluation vectors were reused,
with all tasks, clean/noisy conditions, both Wan normalizations and3SVAEseeds.
The epoch/sample count is not735independent scenes.

## Results: expert initial-image first-close waypoint XY NMSE

| Route | Clean-trained clean | Noise-adapted clean | Clean-trained noise.10 | Noise-adapted noise.10 | Noise-adapted noise.04 |
|---|---:|---:|---:|---:|---:|
| Wan48 native |0.2384|0.3685|182.6310|0.4148|0.3159|
| Wan48 per-token LN |0.2371|0.2762|314.5404|0.4323|0.3228|
| DINO raw768 |0.1197|0.0742|0.3832|0.1139|0.0893|
| DINO PCA48 |0.1351|0.0898|0.6421|0.1107|0.0931|
| DINO SVAE48 seed mean |0.1760|0.1030|0.6014|0.1160|0.0974|

Wan's previously enormous noisy error largely disappears after fair training
adaptation. Its original182.63versusDINO0.38gap cannot be described as intrinsic
representation information loss. The new intervention changes the training
mixture, training normalization and readout jointly; it does not separately
identify each one's causal contribution. Clean Wan performance regresses
(native+54.6percent, LN+16.5percent point estimates); those regressions remain.

A substantial DINO-associated advantage remains under equal augmentation.
At noise.10, PCA48 versus nativeWan48 changes error by−73.31percent, descriptive
paired95percent interval[−82.47,−62.32]; versus WanLN it is−74.40percent,
[−83.27,−63.87]. These are both192-dimensional pooled visual controls.
RawDINO andSVAE48 also have lower errors than both Wan routes; every one of
these comparisons has the same direction on each of the three tasks. Report all
contrasts, not only the best. RawDINO's pooled dimension3072is not matched to192.
PCA andSVAE errors are close; this run does not establish that a learned SVAE is
necessary or that PCA is superior in full world/action training.

## Validation and limits

Saved-artifact audit passed all21fits and63prediction conditions. Training
normalization matches exactly; ridge normal-equation residual1.02e-14;
expanded-affine prediction residual1.01e-15; episode-metric residual8.53e-14.
All7variants of a scene stay in one CV fold. No evaluation vectors or outcome
labels were accessed by the fitting stage. The complete run took47.25seconds,
including training-only encoding and cached-vector evaluation.

Three fixed training-only hardware checks passed predeclared tolerances.
Wan cached vectors were bit-identical; DINO/reducer features differed across
devices: max task RMS0.04597and max coordinate0.22464in clean-training standard
deviation units (limits0.05and0.5). This is approximate parity, not bit identity,
and must remain a numerical limitation. Only those three frames were checked.

The60evaluation scenes and noise severities were already exposed. All intervals
are descriptive, not independent confirmatory inference. This measures only
initial-image waypoint information, not actual contact, object pose, short-state
noninferiority or closed-loop policy success. Architecture/pretraining/objective
differences are confounded. DINO has no trained pixel decoder in this path; full
RAE reproduction remains incomplete. No56/60partial cohort or old80policyreplay.
The original Sep23 01:15:12CDTdeadline remains fixed.
'''
(S/'FINDINGS.md').write_text(findings+'\n## Next uncertainty\n\n'+next_step+'\n')
for src,dst in [('result.json','rae-noise-control.json'),('audit.json','rae-noise-control-audit.json'),('hardware-parity.json','rae-noise-hardware-parity.json')]:shutil.copy2(R/src,O/dst)
shutil.copy2(S/'FINDINGS.md',O/'rae-noise-findings.md');protocol=json.loads((R/'protocol.json').read_text());protocol['resource']='One idle NVIDIA A40, memory<100MiB,util0,ECC0,no compute jobs; cooperative lock; fixed deadline and1200second bound; no other task signals';(O/'rae-noise-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
labels={'Wan48_native_scale':'Wan VAE48 · 原生尺度','Wan48_per_token_layernorm':'Wan VAE48 · 逐 token LN','DINO_raw768':'DINO 原始768维','DINO_PCA48':'DINO + PCA48','DINO_SVAE48_seed_mean':'DINO + SVAE48（三种子平均）'}
rows=''
for name,label in labels.items():
 vals=[]
 for cond in ['clean','noise0.04','noise0.10']:
  x=d['matrix'][cond][name];vals.append(f"{x['original_clean_trained_E']:.4f} → {x['noise_adapted_E']:.4f}")
 rows+='<tr><td>'+label+'</td>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>'
section='''<section id="rae-equal-noise-control"><h2>相同噪声训练后，DINO 的目标读出优势仍存在</h2>
<p>00:05 CDT 完成：相同105个训练场景、相同噪声、按场景划分的交叉验证，冻结编码器与压缩器，只训练21个线性读出器。735份训练输入来自105个场景，不算735个独立样本。按原场景数校正正则化强度，避免重复样本数改变比较。</p>
<p class="banner">以下是初始图像到专家首次闭合抓手位置的 XY 误差，越小越好。箭头：仅干净训练 → 加入相同噪声训练。使用原有60个已暴露场景，属于探索性结果；不是完整 RAE 或闭环机器人提升。</p>
<div class="scroll"><table><thead><tr><th>表征</th><th>干净评测</th><th>噪声 .04</th><th>噪声 .10</th></tr></thead><tbody>'''+rows+'''</tbody></table></div>
<p>Wan 的强噪声误差从182.63降至0.4148，说明之前的巨大差距不能直接解释为信息丢失。干净性能同时出现退化，已完整保留。公平增强之后，DINO 三条路线的强噪声误差仍约0.111–0.116，三个任务方向一致。</p>
<p>在同为192维的读出输入下，PCA48相对原生Wan48误差降低73.3%，描述性配对95%区间为62.3%–82.5%；逐token LN的Wan对照也保留。PCA与SVAE数值接近，目前没有证明学习式压缩必不可少。</p>
<p>21组拟合和63组预测通过独立数值核查。跨设备训练帧检查中，DINO存在最大约0.046个训练标准差的均方根差异，低于预设门槛，但不是逐位相同。所有归一化、增强与模型选择仅使用训练数据。</p>
<p><strong>下一步：</strong>补同样的短时状态读出控制，检验目标信息优势能否同时保留动作相关信息。目前不能声称状态保持、接触建模或策略成功率提升。</p>
<p><a href="rae-noise-findings.md">解释与限制</a> · <a href="rae-noise-control.json">全部任务、条件与区间</a> · <a href="rae-noise-protocol.json">预先固定方案</a> · <a href="rae-noise-control-audit.json">独立数值核查</a> · <a href="rae-noise-hardware-parity.json">跨设备检查</a></p>
<p class="note">公网更新仍失败：Sites project not found。此版本仅在本地报告。</p></section>'''
f=O/'index.html';s=f.read_text();assert 'id="rae-equal-noise-control"' not in s;s=s.replace('<section id="rae-matched-baseline">',section+'\n<section id="rae-matched-baseline">',1);s=s.replace('<h1>新增像素 VAE 对照：先解释噪声下的差距</h1>','<h1>相同噪声训练后，哪些表征仍保留目标信息？</h1>');s=s.replace('同一 DINOv3 骨干、相同压缩预算，比较末层 L12、中间层 L6 与末四层等权平均 K4。主候选始终是 K4-S-VAE；结果不好也不换成其他候选。','当前新增像素VAE对照、分布偏移诊断与相同噪声训练控制。全部正负结果与较早的层级比较均保留，离线读出不等于闭环策略收益。');s=re.sub(r'本地报告更新：[^。]+。','本地报告更新：'+now+'。',s,count=1);f.write_text(s)
for n in ['rae-noise-control.json','rae-noise-control-audit.json','rae-noise-hardware-parity.json','rae-noise-findings.md','rae-noise-protocol.json']:
 t=(O/n).read_text();assert not any(x in t for x in ['/home/zifanz4','/data02/','/vol13/','.csl.illinois.edu','GPU-d13929da'])
for link in re.findall(r'href="([^"#]+)"',s):
 if '://' not in link:assert (O/link).exists(),link
hashes=json.loads((O/'artifact-sha256.json').read_text())
for f in O.iterdir():
 if f.is_file() and f.name!='artifact-sha256.json':hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(O/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
summary='Equal-noise goal control completed00:05:20 exit0;21newridgefits/63predictionconditions independently audited.105training scenes,7matched variants, per-scene regularization, cached exposed60eval. Wan noise.10goalE182.631→0.41479(native),314.540→0.43234(LN); DINOraw/PCA48/SVAE48afteradaptation0.11391/0.11069/0.11603. Wan cleanEworsens0.23838→0.36845and0.23708→0.27619; retained. DINOadvantage survives same augmentation and matched192D PCA control; not proof of fullRAE, causalpretraining, shortstate or policygain. Across-device DINO parity approximate (maxRMS0.04597trainSD), within predeclared bound. GPUreleased. See studies/rae-noise-control-20260922/FINDINGS.md. Public publication again failed: Sites project not found; local HTML updated.'
header='# Equal-noise goal control completed — '+now+'\n\n'+summary+'\n\nNext: '+next_step+'\n\n---\n\n'
for f in [P/'STATUS.md',P/'NEXT_STEPS.md']:f.write_text(header+f.read_text())
f=P/'studies/layers-20260922/progress.json';p=json.loads(f.read_text());p['rae_noise_control']={'status':'completed_audited','result':str(S/'result.json'),'audit':str(S/'audit.json'),'exit_code':0,'closed_loop_evaluated':False,'next_stage':'short_state_equal_noise_control_not_launched'};p['last_selfcheck']={'checked_at':now,'trigger':'2026-09-22T23:45:01.494044-05:00','scientific_state_changed':True,'owned_jobs_running':False,'evidence':str(S/'FINDINGS.md')};p['next_step']=next_step;f.write_text(json.dumps(p,indent=2)+'\n')
evidence={'checked_at':now,'trigger':'2026-09-22T23:45:01.494044-05:00','actual_start':'2026-09-22T23:57:41.959221-05:00','summary':summary,'next_step':next_step,'deadline':'2026-09-23T01:15:12-05:00','old_goalaux':'terminalexit1/incomplete56unscored, no confirmationv3','gpu_released':True,'resource_after':'A40 0MiB,0percent,0ECC, no process on selected UUID','no_pause_markers':True};(S/'selfcheck-2345.json').write_text(json.dumps(evidence,indent=2)+'\n')
f=Path('/home/zifanz4/.local/state/openwam-selfcheck/last-reasoned.json');p=json.loads(f.read_text());p.update(updated_at=now,status='equal_noise_goal_control_completed_audited',next_step=next_step,evidence=str(S/'FINDINGS.md'));f.write_text(json.dumps(p,indent=2)+'\n')
with (P/'selfchecks.md').open('a') as f:f.write('\n## '+now+' — delayed23:45trigger handled23:57; equal-noise goal control completed\n\n'+summary+'\n\n'+next_step+'\nOriginaldeadline unchanged. No old80/incomplete56replay, newtimer, agent, other-task signal or change to older frozen protocols. Newprotocol51443e9604b7edd1aaa3861dc6a75831b529bbcde42f506825aaf97d7b17d483 frozen before execution. Local report links, artifacts and private-path scan passed; no successful public deployment claimed.\n')
f=P/'contribution/REPRESENTATION_EVIDENCE.md';f.write_text('# Update: equal-noise representation controls — '+now+'\n\n'+summary+'\n\nPotential small contribution: an opt-in encoder/readout benchmark with paired perturbations, scene-grouped CV, per-scene regularization, PCA dimension control, full positive/negative reporting and validated native preprocessing. No default policy/encoder change or fullRAE claim is justified yet.\n\n---\n\n'+f.read_text())
print(json.dumps({'report':str(O/'index.html'),'artifact_count':len(hashes),'updated_at':now,'next_step':next_step}))
