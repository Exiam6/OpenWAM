"""Archive and report the completed, independently audited state control."""
from pathlib import Path
import datetime,hashlib,json,re,shutil
P=Path('/home/zifanz4/openwam-experiments');R=Path('/data02/zifanz4/openwam-experiments/results/rae-state-control-20260923');S=P/'studies/rae-state-control-20260923';O=P/'reports/layers-20260922';now=datetime.datetime.now().astimezone().isoformat()
d=json.loads((R/'result.json').read_text());a=json.loads((R/'audit.json').read_text());c=json.loads((R/'component-summary.json').read_text());assert a['passed'] and json.loads((R/'exit.json').read_text())['exit_code']==0
for f in R.iterdir():
 if f.is_file() and f.suffix in ['.json','.log']:shutil.copy2(f,S/f.name)
next_step='Archive the equal-noise goal/state findings and prepare an independent confirmation protocol with frozen preprocessing, all routes, both Wan norms, clean/noisy conditions, and separate translation/gripper endpoints. The remaining original window cannot accommodate the measured roughly94-minute native60scene collection plus scoring; do not launch a knowingly over-budget cohort or shrink it after observing results. No extension past01:15:12CDT, no expired56/60collection reuse, no old80replay. Independent sample validation and fullRAE/world-action training remain future work, not completed gains. Finish the small reproducibility/input-range documentation contribution and preserve the deadline.'
findings='''# Equal-noise short-state control: improvement with a component tradeoff

Completed 2026-09-23 00:24:31 CDT, exit 0. One A40, 386.32 seconds; GPU released
and checked at 0 MiB / 0% / 0 uncorrected volatile ECC errors, no compute process.
The original study deadline remains 2026-09-23 01:15:12 CDT.

## Controlled comparison

105 training scenes, 12 fixed frames per scene, 7 variants per frame (clean plus
3 draws each at pixel sigma .04/.10). 8,820 rows represent 105 independent scenes.
All 1,260 clean vectors and 630 already computed noisy initial-frame vectors were
reused. Only 6,930 missing noisy training images were encoded. All existing
60-scene evaluation vectors were reused without another encoder pass.
42 scene-grouped ridge fits (7 routes x 3 tasks x translation/gripper), matching
fivefold CV, training-only normalization and base-alpha grid. Effective alpha is
7x the base alpha relative to the original 12-frame clean fit, not 84x. Both
Wan normalizations, all tasks/conditions and all three SVAE seeds are retained.
No encoder or compression weights were changed. SVAE errors are averaged within
the same scene; seeds are not counted as independent scenes.

## Results: short-horizon state E (lower is better)

E is the equally weighted mean of training-variance-normalized translation and
gripper MSE, then equally averaged over scenes and tasks. It predicts future
end-effector deltas and gripper state, not contact forces or commanded actions.

| Route | Clean | Noise .04 | Noise .10 |
|---|---:|---:|---:|
| Wan48 native | .311144 | .301808 | .308657 |
| Wan48 per-token LN | .317889 | .302171 | .307048 |
| DINO raw768 | .138145 | .126476 | .137719 |
| DINO PCA48 | .174708 | .158600 | .173212 |
| DINO SVAE48 seed mean | .171989 | .160043 | .173965 |
| Cached proprio-only reference | .347911 | .347911 | .347911 |

At noise .10, PCA48 versus noise-adapted native Wan changes E by -43.88%,
descriptive scene-paired 95% interval [-52.42, -33.87]; versus Wan LN it changes
by -43.59% [-52.33, -33.03]. PCA48 and Wan each have 192 pooled visual inputs;
DINO raw has 3,072. The matched low-dimensional result is not attributable simply
to the raw DINO readout having more inputs. Architecture/objective/pretraining
still differ, so this does not isolate a causal effect of pretraining.

Noise adaptation also rescues Wan's clean-trained state error: 37.338 -> .3087
(native), 26.538 -> .3070 (LN) at sigma .10. Therefore the original massive noisy
gap cannot be interpreted as irreversible loss of control information. Wan's
clean E worsens by about 5.7% in both normalizations. PCA/SVAE clean improvement
versus their own clean-trained versions is about 5.9%/5.6%, with intervals crossing
zero. No uniformly beneficial training intervention is established.

## Do not hide the gripper tradeoff

Noise .10, averages over three tasks; each column is a normalized MSE group.

| Route | Translation | Gripper | Gripper change vs native Wan |
|---|---:|---:|---:|
| Wan48 native | .596071 | .021243 | reference |
| Wan48 per-token LN | .592837 | .021259 | +0.08% |
| DINO raw768 | .240619 | .034819 | +63.91% |
| DINO PCA48 | .327466 | .018959 | -10.75% |
| DINO SVAE48 seed mean | .329050 | .018879 | -11.13% |
| Cached proprio only | .675046 | .020776 | -2.20% |

Raw DINO has the best aggregate and translation error but worse gripper error
than Wan on ALL three tasks (+102.38%, +114.20%, +39.04%). More representation
dimensions are not automatically better for every target under this readout.
The role of finite-sample readout regularization versus information content is
unresolved; do not claim compression causally removes nuisance information.

PCA/SVAE have lower translation and gripper point errors than both Wan variants
on each task. However PCA's aggregated gripper change vs native Wan has a
post-hoc descriptive interval [-25.09%, +9.84%], and SVAE [-24.93%, +8.86%].
These intervals include worsening: a reliable gripper benefit or noninferiority
has NOT been established. This subgroup summary does not replace the frozen
aggregate endpoint or promote exploratory intervals to confirmatory inference.

The cached proprio-only reference was declared before the state results.
PCA's total E is 50.21% below proprio-only, indicating that the joint visual/state
readout adds predictive value in this setup. The gripper story remains mixed:
PCA/SVAE gripper errors on adjust_bottle are +3.56%/+2.29% above proprio-only;
their average gripper changes are -8.74%/-9.13%, with intervals crossing zero.
Raw DINO's average gripper error is +67.59% above proprio-only. The comparison
reuses the original frozen proprio-only fit, with no new fit or heldout calibration.

## Independent arithmetic verification and limitations

All 42 fits and 63 prediction conditions passed a separate saved-artifact audit:
training statistics match exactly; normalized ridge normal-equation residual
6.51e-13, expanded affine prediction difference 2.33e-15, per-episode group
metric difference 6.22e-15. All 630 reused initial noisy vectors are bit-identical.
Frozen inputs/code remained unchanged, folds are scene-disjoint, and no evaluation
vectors or labels were accessed during fitting. The efficient eigensolver matched
the original direct solve on a training-independent synthetic fixture to 3.02e-14.

The supplemental proprio summary first stopped on a bit-exact aggregate comparison.
Diagnosis found only 1.11e-16 to 4.44e-16 differences from clean versus triplicated
noise reductions. The corrected check permits 1e-14 absolute roundoff. Every raw
score and the failure diagnosis remain; no model, result or primary audit changed.

Across-device DINO parity remains approximate: prior three-training-frame check
had max RMS .04597 and max coordinate .22464 in training-SD units, below the
predeclared .05/.5 limits. Wan was exact. This is not a full hardware invariance
study. The 60 evaluation scenes and perturbation strengths were already exposed;
all intervals are descriptive. These are offline expert-trajectory readouts, not
learned-policy rollouts, independent confirmation, measured contact reasoning,
novel-camera robustness, causal evidence of pretraining, or complete RAE training.
DINO's frozen-feature path still lacks the pixel decoder of a complete RAE.

## What this can contribute

A paired representation benchmark with a native Wan baseline, strict image-range
and causal-first-frame checks, an equal-noise-training control, train-only scene CV,
PCA dimension control, cached proprio baseline, and separate gripper/translation
reporting. The biggest methodological correction is that clean-trained noise
collapse is not itself proof of information loss. The practical candidate for
further validation is compact pretrained features; learned SVAE necessity is not
demonstrated because PCA is similarly competitive. Do not change policy defaults.
'''
(S/'FINDINGS.md').write_text(findings+'\n## Next uncertainty\n\n'+next_step+'\n')
publication={'checked_at':now,'project_id':'appgprj_6ab06a7c072481919a12cc100f8e9fa3','status':'blocked','error':'SitesConnectorError: Sites project not found','lookup_observed_at':'2026-09-23T00:29:47-05:00','new_site_created':False,'deployed':False};(S/'publication-check.json').write_text(json.dumps(publication,indent=2)+'\n')
for src,dst in [('result.json','rae-state-control.json'),('audit.json','rae-state-control-audit.json'),('component-summary.json','rae-state-components.json'),('proprio-rounding-diagnostic.json','rae-state-rounding-diagnostic.json')]:shutil.copy2(R/src,O/dst)
shutil.copy2(S/'FINDINGS.md',O/'rae-state-findings.md');protocol=json.loads((R/'protocol.json').read_text());protocol['resource']='One idle A40; exclusive cooperative lock;1800second maximum and original fixed01:15:12CDTdeadline; no other tasks interrupted';(O/'rae-state-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
labels={'Wan48_native_scale':'Wan VAE48 · 原生尺度','Wan48_per_token_layernorm':'Wan VAE48 · 逐 token LN','DINO_raw768':'DINO 原始768维','DINO_PCA48':'DINO + PCA48','DINO_SVAE48_seed_mean':'DINO + SVAE48（三种子平均）'}
rows=''
for name,label in labels.items():
 vals=[d['matrix'][con][name]['noise_adapted_E'] for con in ['clean','noise0.04','noise0.10']];gr=c['conditions']['noise0.10']['routes'][name]['versus_Wan_native']['group_change_percent'][1];rows+='<tr><td>'+label+'</td>'+''.join(f'<td>{v:.4f}</td>' for v in vals)+f'<td>{gr:+.1f}%</td></tr>'
rows+='<tr><td>仅机器人自身状态（缓存对照）</td><td>0.3479</td><td>0.3479</td><td>0.3479</td><td>−2.2%</td></tr>'
section='''<section id="rae-state-control"><h2>短时状态验证：压缩语义表征有优势，但抓手收益尚不确定</h2>
<p>00:24 CDT 完成：105个训练场景，每场景12帧，所有表征使用相同干净／噪声混合训练。42组线性拟合、63组预测通过独立核查；复用已有评测特征，仅新增6,930张训练噪声图像的编码。</p>
<p class="banner">短时状态 E 是位移与抓手两组归一化均方误差的平均，越低越好。以下均为相同噪声训练后的结果；60个评测场景已暴露，属于探索性证据。没有完成完整 RAE 或证明闭环策略收益。</p>
<div class="scroll"><table><thead><tr><th>表征</th><th>干净 E</th><th>噪声 .04 E</th><th>噪声 .10 E</th><th>.10 抓手误差相对原生 Wan</th></tr></thead><tbody>'''+rows+'''</tbody></table></div>
<p>同为192维池化视觉输入，PCA48在强噪声下的总状态误差比原生Wan低43.9%，比仅用机器人自身状态低50.2%。它与SVAE的状态、目标读出均有较好结果，目前没有证据表明必须使用学习式压缩。</p>
<p><strong>平均分会掩盖取舍：</strong>原始DINO的位移和总误差最低，抓手误差却比Wan高63.9%，三个任务都退化。PCA/SVAE的抓手点估计较好，但描述性95%区间仍跨过零；PCA相对Wan为−25.1%至+9.8%。因此不能声称抓手稳定改善或各控制分量全面提升。相对仅自身状态，PCA在adjust_bottle抓手上仍退化3.6%。</p>
<p>Wan经过噪声训练也从37.34降至0.3087，干净误差则增加约5.7%。保留这些负结果，避免把读出器分布偏移误认为不可恢复的信息损失。</p>
<p><strong>下一步解决的疑问：</strong>紧凑语义特征的位移优势、抓手取舍能否在新场景重现？应固定新场景协议后独立验证，再考虑匹配的世界／动作模型训练。当前剩余窗口不足以完成实测约94分钟的60场景采集，本轮不缩小样本凑结论，也不延长01:15截止时间。</p>
<p><a href="rae-state-findings.md">完整解释与限制</a> · <a href="rae-state-control.json">所有任务与条件</a> · <a href="rae-state-components.json">位移／抓手与自身状态对照</a> · <a href="rae-state-protocol.json">固定方案</a> · <a href="rae-state-control-audit.json">独立核查</a> · <a href="rae-state-rounding-diagnostic.json">补充汇总的舍入诊断</a></p>
<p class="note">公网发布仍被 Sites project not found 阻断；此更新只保存为本地网页，未声称上线。</p></section>'''
f=O/'index.html';s=f.read_text();assert 'id="rae-state-control"' not in s;s=s.replace('<section id="rae-equal-noise-control">',section+'\n<section id="rae-equal-noise-control">',1);s=s.replace('<h1>相同噪声训练后，哪些表征仍保留目标信息？</h1>','<h1>什么表征更适合控制：收益与抓手取舍</h1>');s=re.sub(r'本地报告更新：[^。]+。','本地报告更新：'+now+'。',s,count=1);s=s.replace('<strong>下一步：</strong>补同样的短时状态读出控制，检验目标信息优势能否同时保留动作相关信息。目前不能声称状态保持、接触建模或策略成功率提升。','<strong>后续更新：</strong>短时状态控制已完成，见页面顶部。抓手收益仍不确定；尚无接触建模或策略成功率提升证据。');f.write_text(s)
for n in ['rae-state-control.json','rae-state-control-audit.json','rae-state-components.json','rae-state-findings.md','rae-state-protocol.json','rae-state-rounding-diagnostic.json']:
 assert not any(x in (O/n).read_text() for x in ['/home/zifanz4','/data02/','/vol13/','.csl.illinois.edu','GPU-d13929da']),n
for link in re.findall(r'href="([^"#]+)"',s):
 if '://' not in link:assert (O/link).exists(),link
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file() and f.name!='artifact-sha256.json'};(O/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
summary='Equal-noise short-state control completed00:24:31 exit0,42fits/63prediction conditions independently audited;105train/exposed60eval. Noise.10stateE: Wan native0.308657/LN0.307048, DINOraw0.137719/PCA480.173212/SVAE480.173965, cachedproprio0.347911. PCAvsnativeWan -43.88% descriptive95[-52.42,-33.87]; same pooled192D visual input. RawDINO gripper+63.91% vsWan despite besttotalE. CompactPCA/SVAE gripper pointgains~11%, intervals crosszero; adjust_bottlegripper+3.56%/+2.29%vsproprio. Wan cleanregression~5.7%retained. No independent/fullRAE/closedloopgain. GPUreleased; no pause; originaldeadline unchanged. PublicSites lookup again failed projectnotfound; portableHTML updated.'
header='# Equal-noise short-state control completed — '+now+'\n\n'+summary+'\n\nNext: '+next_step+'\n\n---\n\n'
for f in [P/'STATUS.md',P/'NEXT_STEPS.md']:f.write_text(header+f.read_text())
f=P/'studies/layers-20260922/progress.json';p=json.loads(f.read_text());p['updated_at']=now;p['rae_state_control'].update(status='completed_audited',result=str(S/'result.json'),audit=str(S/'audit.json'),component_summary=str(S/'component-summary.json'),exit_code=0,closed_loop_evaluated=False,full_RAE_reproduced=False,gpu_released=True);p['rae_noise_control']['next_stage']='short_state_equal_noise_control_completed_audited';p['public_report_update']=dict(publication,local_report=str(O/'index.html'));p['last_selfcheck']={'checked_at':now,'trigger':'2026-09-23T00:00:01.504356-05:00','status':'equal_noise_state_completed_audited','scientific_state_changed':True,'owned_jobs_running':False,'evidence':str(S/'FINDINGS.md')};p['next_step']=next_step;f.write_text(json.dumps(p,indent=2)+'\n')
evidence={'checked_at':now,'trigger':'2026-09-23T00:00:01.504356-05:00','summary':summary,'next_step':next_step,'deadline':'2026-09-23T01:15:12-05:00','gpu_released':True,'resource_after':'A40 0MiB,0percent,0ECC, no compute process on selected GPU','no_pause_markers':True,'protocol_sha256':d['protocol_sha256']};(S/'selfcheck-0000.json').write_text(json.dumps(evidence,indent=2)+'\n')
f=Path('/home/zifanz4/.local/state/openwam-selfcheck/last-reasoned.json');p=json.loads(f.read_text());p.update(updated_at=now,status='equal_noise_short_state_completed_audited',next_step=next_step,evidence=str(S/'FINDINGS.md'));f.write_text(json.dumps(p,indent=2)+'\n')
with (P/'selfchecks.md').open('a') as f:f.write('\n## '+now+' — delayed00:00trigger; short-state control completed and audited\n\n'+summary+'\n\n'+next_step+'\nNo newtimer/agent, other-task signal, old80replay or expired56confirmation. Original protocols unchanged. Supplemental roundoff failure retained; correction only1e-14aggregatecheck, no score modification.\n')
f=P/'contribution/REPRESENTATION_EVIDENCE.md';f.write_text('# Update: equal-noise state/goal evidence — '+now+'\n\n'+summary+'\n\nSmall contribution: paired native-Wan/pretrained-feature offline benchmark, including matched noise adaptation, PCA dimensionality control, component-wise gripper checks and cached proprio reference. Do not replace encoder/policy defaults. A separate documentation-only patch corrects WanVideoVAE.batch_encode input range to[-1,1], matching native preprocessing; source remains frozen/unmodified.\n\n---\n\n'+f.read_text())
print(json.dumps({'report':str(O/'index.html'),'artifact_count':len(hashes),'updated_at':now,'audit_passed':a['passed'],'public_deployed':False}))
