#!/usr/bin/env python3
"""Publishable snapshot; final claims require audited summary.json."""
import html,json,os,re,shutil,tarfile,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;RUN=Path('/data02/zifanz4/openwam-experiments/results/norm-20260921')
SITE=Path(os.environ.get('NORM_SITE_DIR','/home/zifanz4/research-reports/dist'));A=SITE/'assets/norm-20260921';A.mkdir(parents=True,exist_ok=True)
ARMS=['identity','renorm','adapter','adapter_renorm'];LABELS={'identity':'原策略','renorm':'原策略再归一化','adapter':'旧适配器','adapter_renorm':'适配器后再归一化'}
summary=json.loads((RUN/'summary.json').read_text()) if (RUN/'summary.json').exists() else None
if summary:assert summary['all_integrity_checks_passed']
partial=json.loads((RUN/'partial-audit.json').read_text()) if (RUN/'partial-audit.json').exists() else None
evidence=summary or partial
rows=[];complete=0
for arm in ARMS:
 for condition in ['clean','noise']:
  d=RUN/f'{arm}-{condition}';records=[]
  if (d/'episodes.json').exists():
   try:records=json.loads((d/'episodes.json').read_text())
   except json.JSONDecodeError:pass
  complete+=len(records)
  row=[LABELS[arm], '干净' if condition=='clean' else '噪声 σ=0.20', f'{len(records)}/10',sum(x['success'] for x in records)]
  if summary:
   stats=summary['primary'][f'{arm}-{condition}'];row+=[f"{100*stats['success_rate']:.0f}%",f"{100*stats['wilson_95ci'][0]:.1f}–{100*stats['wilson_95ci'][1]:.1f}%"]
  rows.append(row)
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
offline=json.loads((RUN/'offline.json').read_text());stats=offline['arms']
fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
x=range(4);colors=['#84998f','#b9c4b6','#d8ad60','#26694c']
axs[0].bar(x,[stats[k]['noise_restoration_mse'] for k in ARMS],color=colors);axs[0].set_ylabel('MSE to original clean latent');axs[0].set_title('Validation noise sigma = 0.10')
axs[1].bar(x,[stats[k]['statistics']['noise']['mean_token_rms'] for k in ARMS],color=colors);axs[1].axhline(1,color='#223b31',ls='--',lw=1);axs[1].set_ylim(.9,1.01);axs[1].set_ylabel('Mean token RMS');axs[1].set_title('Output scale, noise input')
for ax in axs:ax.set_xticks(x,['Original','Original\n+ norm','Adapter','Adapter\n+ norm'],fontsize=9);ax.spines[['top','right']].set_visible(False)
fig.savefig(A/'offline.svg');plt.close(fig)
files=['protocol.md','source-freeze.json','analysis-freeze.json','offline.json','intervention-cpu-check.json','replay-cpu-check.json','historical-scene.json','historical-repeat-note.json','fresh-scenes.json','summary.json','partial-audit.json','resource-resume-amendment.md','resume-source-freeze.json','resume-start.json','completion-after-resume.json']
for f in files:
 if (RUN/f).exists():shutil.copyfile(RUN/f,A/f)
for f in RUN.glob('*-hook.json'):shutil.copyfile(f,A/f.name)
with tarfile.open(A/'reproduction-scripts.tar.gz','w:gz') as t:
 for f in ['NORM_PROTOCOL.md','REPRODUCE_NORM.md','scripts/norm_common.py','scripts/norm_client.py','scripts/serve_norm.py','scripts/norm_freeze_manifest.py','scripts/norm_offline.py','scripts/run-norm-study.sh','scripts/check_norm_intervention.py','scripts/check_norm_replay.py','scripts/summarize_norm.py','scripts/robustness_common.py','scripts/robustness_client.py','scripts/resume-norm-last-arm.sh','scripts/norm_resume_preflight.py','scripts/finish_resumed_norm.py','policy-requirements-lock.txt','benchmark-requirements-lock.txt']:
  t.add(ROOT/f,arcname=f)
 for f in ['fresh-scenes.json','historical-scene.json','resource-resume-amendment.md','resume-source-freeze.json']:
  if (RUN/f).exists():t.add(RUN/f,arcname='results/norm-20260921/'+f)
interpretation = ''
if summary:
 interpretation = '<div class="callout caution"><strong>结论：再归一化没有显示净收益。</strong>冻结适配器由干净 9/10、强噪声 9/10，变成 10/10、8/10：挽回一次干净失败，同时新增一次噪声失败。两项预定配对比较的双侧精确 p 均为 1.0。原策略再归一化的噪声结果为 9/10，对原策略 7/10 的配对 p=0.5；十个场景不足以确认稳定提升。不能据此声称 RAE 优于 VAE，或接触阶段更稳健。</div>'
if summary:status='已完成，配对与来源核验通过'
elif (RUN/'resume-exit-code.txt').exists() and (RUN/'resume-exit-code.txt').read_text().strip()!='0':status='续跑已停止：请查看运行记录，尚无最终结论'
elif (RUN/'resume-launcher.log').exists() and not (RUN/'resume-exit-code.txt').exists():status='最后一组已恢复运行；前三组已核验'
elif (RUN/'study-exit-code.txt').exists() and (RUN/'study-exit-code.txt').read_text().strip()!='0':status='任务已停止：尚无最终结论'
else:status='运行中：固定方案，尚无最终结论'
resource_note='<p>运行记录：前三组完成后，16:27 的完全空闲预检被一个小型图形上下文触发，最后一组没有启动。停机日志已保留，17:28 只恢复未开始的最后一组，未重跑或删除先前三组。恢复时图形进程已自行退出，实际预检为 6 MiB、0% 利用率、无其他进程。仍使用同一张 GPU，模型与场景设置不变。<a href="assets/norm-20260921/resource-resume-amendment.md">续跑说明</a>。</p>' if (RUN/'resource-resume-amendment.md').exists() else ''
lead=f'80 次新场景配对评测已完成。' if summary else f'已完成 {complete}/80 次主要评测。本页是发布时快照，不是实时监控。'
comparison=''
if evidence:
 comparison=table(['候选 − 对照','条件','新增成功','新增失败','相同结局','双侧精确 p'],[[key.split('-')[0], '干净' if key.endswith('-clean') else '噪声',v['wins'],v['losses'],v['ties'],f"{v['two_sided_exact_p']:.4f}"] for key,v in evidence['paired_comparisons'].items()])
if partial and not summary:
 comparison+='<p>以上比较仅覆盖已完整核验的前三组；两个噪声比较均为新增成功 2、新增失败 0，双侧配对 p=0.5，尚不足以证明稳定提升。第四组和两项预定主要比较仍待完成。</p>'
if summary:
 comparison=table(['候选 − 对照','条件','新增成功','新增失败','相同结局','双侧精确 p'],[[key.split('-')[0], '干净' if key.endswith('-clean') else '噪声',v['wins'],v['losses'],v['ties'],f"{v['two_sided_exact_p']:.4f}"] for key,v in summary['paired_comparisons'].items()])
 historical=table(['模型','Seed 300003','动作数'],[[LABELS[k],'成功' if v['records'][0]['success'] else '失败',v['records'][0]['actions']] for k,v in summary['historical'].items()])
else:
 hist_rows=[]
 for arm in ARMS:
  path=RUN/f'{arm}-historical-clean/summary.json'
  data=json.loads(path.read_text()) if path.exists() else None
  hist_rows.append([LABELS[arm],('成功' if data['records'][0]['success'] else '失败') if data else '等待／运行中',data['records'][0]['actions'] if data else '—'])
 historical=table(['组别','Seed 300003','动作数'],hist_rows)
ref_path=RUN/'reference-clean/summary.json';ref_completed=10 if ref_path.exists() else 0
if not summary:lead+=f' 十个场景的可行性参考已完成 {ref_completed}/10；参考成功率不并入主要结果。'
(A/'progress.json').write_text(json.dumps({'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'primary_completed':complete,'primary_planned':80,'reference_completed':ref_completed,'final':bool(summary),'status':status},indent=2))
videos=[]
if evidence:
 groups=[(k,v,'主要评测') for k,v in evidence['primary'].items()]+[(k+'-historical-clean',v,'历史诊断') for k,v in evidence['historical'].items()]+[('reference-clean',evidence['reference'],'可行性参考：不计入主要成功率')]
 for name,data,role in groups:
  dest=A/name;dest.mkdir(exist_ok=True);blocks=[]
  for i,row in enumerate(data['records']):
   paths=list((RUN/name/'runtime/eval_result').rglob(f'episode{i}.mp4'));assert len(paths)==1
   shutil.copyfile(paths[0],dest/f'episode{i}.mp4')
   blocks.append(f'<details><summary>Seed {row["seed"]} · {"成功" if row["success"] else "失败"} · {row["actions"]} 步</summary><p>{html.escape(row["instruction"])}</p><video controls playsinline preload="none" src="assets/norm-20260921/{name}/episode{i}.mp4" style="width:100%;max-width:640px"></video></details>')
  first=data['records'][0]['seed'];shutil.copyfile(RUN/name/f'seed-{first}-policy.png',dest/'policy-input.png')
  videos.append(f'<details><summary>{html.escape(name)} · {role} · {data["successes"]}/{data["episodes"]}</summary><img src="assets/norm-20260921/{name}/policy-input.png" style="width:320px;max-width:100%" alt="实际初始策略输入">'+''.join(blocks)+'</details>')
historical_note='<div class="callout caution">'+'后续重放更新：在新一轮单场景诊断中，原策略在 seed 300003 也于 400 步失败；初始主相机图像、状态与指令一致。此次执行顺序与上一轮不同，差异来源尚未定位，不能把原来的单次成败差异解释为稳定的适配器退化。新场景四组对照仍按冻结方案进行。'+'</div>' if (RUN/'historical-repeat-note.json').exists() else ''
historical_preview=''
if not summary:
 for arm in ARMS:
  if evidence and arm in evidence['historical']:continue
  name=arm+'-historical-clean';d=RUN/name
  if not (d/'summary.json').exists():continue
  paths=list((d/'runtime/eval_result').rglob('episode0.mp4'));assert len(paths)==1
  dest=A/name;dest.mkdir(exist_ok=True);shutil.copyfile(paths[0],dest/'episode0.mp4')
  shutil.copyfile(d/'summary.json',dest/'summary.json')
  historical_preview+=f'<details><summary>{LABELS[arm]} · 本轮历史诊断完整视频</summary><video controls playsinline preload="none" src="assets/norm-20260921/{name}/episode0.mp4" style="width:100%;max-width:640px"></video><p><a href="assets/norm-20260921/{name}/summary.json">逐轨迹记录</a></p></details>'
artifact_links=' · '.join(f'<a href="assets/norm-20260921/{f}">{html.escape(f)}</a>' for f in files if (A/f).exists())
body=f'''<div class="hero"><div class="eyebrow">Frozen four-arm follow-up / 2026.09.21</div><h1>恢复幅度，<br><em>能否恢复控制？</em></h1><p class="lead">{lead}</p></div>
<section id="status"><h2>{status}</h2>{interpretation}{resource_note}<p>更新于 {time.strftime('%Y-%m-%d %H:%M:%S %Z')}。原权重与适配器完全冻结，没有重新训练、插值强度搜索或按闭环结果选模型。</p><div class="callout caution">只有双臂抓瓶一个任务、每个条件 10 个新场景。旧失败场景是被挑选的诊断例子，不能作为泛化收益证据。<a href="robustness.html">上一轮完整结果</a>。</div></section>
<section id="design"><h2>四组对照，只改变归一化位置</h2>{table(['组别','操作'],[[LABELS['identity'],'原编码器 → 原策略'],[LABELS['renorm'],'原编码器 → 再执行原 LayerNorm → 原策略'],[LABELS['adapter'],'原编码器 → 冻结适配器 → 原策略'],[LABELS['adapter_renorm'],'原编码器 → 冻结适配器 → 原 LayerNorm → 原策略']])}<p>仅改变当前观测帧的 48 通道潜变量；未来槽位逐值核对不变。复用原编码器的非仿射 LayerNorm（ε=10⁻⁶）和 BF16 精度。原策略再归一化这一组用于控制重复归一化自身的数值影响。</p><p>从种子 400000 起，固定首批十个专家可行的场景和各自指令，独立于原策略成功与否；十条参考轨迹不计入主要成功率。随后四组全部重放相同起点，逐条检查初始图像与状态，不重复专家过滤。400 步上限、10 次去噪、扩散种子 42、关闭编译。</p><p>本轮闭环噪声固定为 σ=0.20，超过适配器训练的 σ=0.10；这是预先确定的更强压力测试，不是噪声强度搜索。仅主相机加噪，腕部、状态和指令不变。</p></section>
<section id="offline"><h2>离线：恢复幅度，不保证误差更低</h2><img src="assets/norm-20260921/offline.svg" style="width:100%;height:auto" alt="四组噪声恢复误差与潜变量幅度">{table(['组别','噪声恢复 MSE','干净改动 MSE','噪声 token RMS'],[[LABELS[k],f"{v['noise_restoration_mse']:.6f}",f"{v['clean_distortion_mse']:.8f}",f"{v['statistics']['noise']['mean_token_rms']:.6f}"] for k,v in stats.items()])}<p>此处复用五条验证演示的 σ=0.10 缓存，与本轮闭环 σ=0.20 不同。十条保留测试演示仍未打开；离线统计不参与本轮选型或是否继续实验的判定。再归一化使适配器输出 RMS 从约 0.962 回到 1，但恢复 MSE 从 0.10844 升至 0.11173。干净改动误差仅降低约 5%，大部分改动仍然保留；闭环结果才能判断这项改变是否有益。</p></section>
<section id="primary"><h2>主要评测：十个新场景</h2>{table(['组别','输入','已完成／计划','累计成功']+(['成功率','Wilson 95% 区间'] if summary else []),rows)}{comparison}<p>配对统计按同一个场景的成败计算；多个比较均报告，不挑选显著结果。十个场景仍只能提供有限证据，不能推断其他任务或真实机器人。</p></section>
<section id="next"><h2>下一步：先量化重复性与动作敏感性</h2><p>旧场景的原策略在上一轮成功、本轮失败，且重复原 LayerNorm 的微小 BF16 数值变化也对应了不同结局。下一轮先固定输入、提示词、随机种子与执行顺序，比较相同观测的重复动作输出；再用预先固定的少量场景重复完整闭环，区分策略数值敏感性与执行环境波动。先记录协议和资源上限，再运行；暂不继续搜索适配器结构、权重或强度。</p><p>阶段性交付优先是可重复的扰动评测、配对诊断及现有默认关闭的语言随机数修正。只有重复性核验通过后，才扩大新场景或尝试下一项表征改进。</p></section>
<section id="historical"><h2>旧失败场景：单独诊断</h2><p>Seed 300003 在上一轮原策略成功、适配器失败。本轮四组各重放一次，保持旧指令及其五项抽样的 RNG 消耗；新场景则保留十项抽样消耗。此处结果不并入新场景成功率。</p>{historical_note}{historical}{historical_preview}</section>
<section id="videos"><h2>完整录像与实际输入</h2><p>本页保存已核验的完整分组；完整实验共 94 段：80 段主要评测、4 段旧场景诊断、10 段可行性参考。录像显示仿真器的干净画面，各组另附实际送入策略的初始主相机图像。</p>{''.join(videos) if evidence else '<p>实验仍在进行，完成后统一核验并发布。</p>'}</section>
<section id="artifacts"><h2>协议、代码与来源</h2><p>{artifact_links}</p><p><a href="assets/norm-20260921/reproduction-scripts.tar.gz">复现脚本与固定场景</a>。来源冻结于执行前，完整模型及小适配器沿用<a href="robustness.html#artifacts">上一轮发布记录</a>。CPU 检查覆盖原归一化实现、BF16、未来槽位、输入不变和上游原始控制循环的十场景／单场景重放。</p></section>'''
old=(SITE/'robustness.html').read_text();style=''.join(re.findall(r'<link[^>]*rel="stylesheet"[^>]*>',old))+''.join(re.findall(r'<style.*?</style>',old,re.S))
page=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 归一化与控制验证</title>{style}</head><body><header><a href="index.html">Embodied Research Notebook</a><span>归一化验证 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / NORMALIZATION</p><nav aria-label="报告目录"><a href="#status">状态</a><a href="#design">实验设计</a><a href="#offline">离线统计</a><a href="#primary">新场景闭环</a><a href="#historical">旧失败复现</a><a href="#videos">全部录像</a><a href="#artifacts">复现记录</a><a href="robustness.html">上一轮</a></nav></aside><main>{body}</main></div></body></html>'
(SITE/'norm.html').write_text(page)
index=SITE/'index.html';text=index.read_text();callout=f'<div class="callout" id="norm-latest"><strong>归一化验证 · {complete}/80 次主要评测：</strong>四组冻结对照，10 个新场景，干净与更强噪声。<a href="norm.html">查看方案与结果</a>。{status}。{"最终结果：适配器再归一化干净 10/10、噪声 8/10，尚无明确净收益。" if summary else ""}</div>'
if 'id="norm-latest"' in text:text=re.sub(r'<div class="callout" id="norm-latest">.*?</div>',callout,text,flags=re.S)
else:text=text.replace('<main>','<main>'+callout,1)
index.write_text(text)
for svg in (A).glob('*.svg'):
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print(json.dumps({'primary_completed':complete,'final':bool(summary),'page':str(SITE/'norm.html'),'video_count':len(list(A.rglob('*.mp4')))},ensure_ascii=False))
