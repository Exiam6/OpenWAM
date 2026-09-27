#!/usr/bin/env python3
"""Publish the frozen fixed-input diagnostic, distinguishing raw chunks/first actions."""
import base64,html,json,re,shutil,tarfile,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;RUN=Path('/data02/zifanz4/openwam-experiments/results/repeatability-20260921');SITE=Path('/home/zifanz4/research-reports/dist');A=SITE/'assets/repeatability-20260921';A.mkdir(exist_ok=True,parents=True)
summary=json.loads((RUN/'summary.json').read_text())if(RUN/'summary.json').exists()else None
if summary:assert summary['all_integrity_checks_passed']
capture=json.loads((RUN/'capture-summary.json').read_text())if(RUN/'capture-summary.json').exists()else None
counts={}
for phase in ['A','B']:
 p=RUN/f'process-{phase}/records.json'
 try:counts[phase]=len(json.loads(p.read_text())) if p.exists() else 0
 except json.JSONDecodeError:counts[phase]=0
complete=sum(counts.values());status='已完成，来源与输入核验通过'if summary else f'推理进行中：{complete}/54 个动作块'
exit_path=RUN/'exit-code.txt'
if exit_path.exists() and exit_path.read_text().strip()!='0':status='诊断已停止：尚无完整结论'
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
files=['protocol.md','source-freeze.json','scenes.json','cpu-checks.json','capture-summary.json','summary.json','request-isolation-amendment.md','supplemental-audit.json']
for name in files:
 if (RUN/name).exists():shutil.copyfile(RUN/name,A/name)
links=' · '.join(f'<a href="assets/repeatability-20260921/{name}">{name}</a>'for name in files if(A/name).exists())
for phase in ['A','B']:
 d=RUN/f'process-{phase}'
 if d.exists():
  (A/f'process-{phase}').mkdir(exist_ok=True)
  for p in d.iterdir():
   if p.suffix in ['.npz','.json','.yaml']:shutil.copyfile(p,A/f'process-{phase}'/p.name)
if capture:
 (A/'inputs').mkdir(exist_ok=True)
 for row in capture['records']:
  src=RUN/'inputs'/f"{row['name']}.json";shutil.copyfile(src,A/'inputs'/src.name)
  payload=json.loads(src.read_text());(A/'inputs'/f"{row['name']}.png").write_bytes(base64.b64decode(payload['images']['head_camera']))
with tarfile.open(A/'reproduction-scripts.tar.gz','w:gz')as t:
 for name in ['REPEATABILITY_PROTOCOL.md','scripts/repeat_common.py','scripts/capture_repeat_inputs.py','scripts/repeat_action_inference.py','scripts/check_repeatability.py','scripts/summarize_repeatability.py','scripts/run-repeat-study.sh','scripts/resume-repeat-inference.sh','scripts/norm_client.py','scripts/norm_common.py','scripts/robustness_client.py','scripts/robustness_common.py','policy-requirements-lock.txt','benchmark-requirements-lock.txt']:
  t.add(ROOT/name,arcname=name)
 for name in ['scenes.json','source-freeze.json','cpu-checks.json','request-isolation-amendment.md']:t.add(RUN/name,arcname='results/repeatability-20260921/'+name)
with tarfile.open(A/'trace-data.tar.gz','w:gz')as t:
 for name in ['inputs','process-A','process-B','capture-summary.json','summary.json']:
  if (RUN/name).exists():t.add(RUN/name,arcname=name)
interpretation='<p>保存输入已通过核验；完整54个动作块及重复性比较完成前，不作数值稳定性或表征收益判断。</p>'
results='<p>尚在运行。本页是发布时快照。</p>'
if summary:
 agg=summary['aggregates'];repeat_names=['identity_within_A','identity_restart_vs_A0','identity_restart_vs_A1','renorm_within_B','adapter_within_B','adapter_renorm_within_B']
 all_exact=all(agg[k][f]['byte_equal_inputs']==6 for k in repeat_names for f in ['chunk','first_action'])
 interpretation=('<div class="callout"><strong>本次固定输入检查：重复输出逐字节一致。</strong>原策略同进程重复、重启后重复，以及三种干预各自的重复结果，在6份输入上均完全一致。这缩小了排查范围，但只涉及初始观测，不能推断整条仿真轨迹也可重复。</div>' if all_exact else '<div class="callout caution"><strong>存在相同输入的重复输出差异。</strong>请先查看以下逐项比较；当前不能把干预差异全部归因于表征。</div>')
 labels={'identity_within_A':'原策略：同进程重复','identity_restart_vs_A0':'原策略：重启 vs A首次','identity_restart_vs_A1':'原策略：重启 vs A第二次','renorm_within_B':'原策略再归一化：重复','adapter_within_B':'适配器：重复','adapter_renorm_within_B':'适配器再归一化：重复'}
 rows=[[labels[k],f"{agg[k]['chunk']['byte_equal_inputs']}/6",f"{agg[k]['chunk']['max_abs']:.8g}",f"{agg[k]['first_action']['byte_equal_inputs']}/6",f"{agg[k]['first_action']['max_abs']:.8g}"]for k in repeat_names]
 results=table(['比较','原始动作块完全相同','块最大绝对差','实际首动作完全相同','首动作最大绝对差'],rows)
 arm_names={'renorm':'原策略再归一化','adapter':'适配器','adapter_renorm':'适配器再归一化'}
 rows=[]
 for arm,label in arm_names.items():
  for rep in range(2):
   d=agg[f'{arm}_r{rep}_vs_identity'];rows.append([label,f'重复{rep}',f"{d['chunk']['byte_equal_inputs']}/6",f"{d['first_action']['byte_equal_inputs']}/6",f"{d['first_action']['max_abs']:.8g}"])
 coordinate_rows=[]
 for arm,label in arm_names.items():
  for condition in ['clean','noise']:
   selected=[r for r in summary['comparisons']if r['comparison']==f'{arm}_r0_vs_identity'and r['input'].startswith(condition)]
   coordinate_rows.append([label,'干净'if condition=='clean'else'噪声',f"{1000*np.mean([r['first_action']['groups']['xyz']['rmse']for r in selected]):.3f}",f"{1000*max(r['first_action']['groups']['xyz']['max_abs']for r in selected):.3f}"])
 renorm_xyz=max(r['first_action']['groups']['xyz']['max_abs']for r in summary['comparisons']if r['comparison']=='renorm_r0_vs_identity')
 interpretation+=f'<p>单纯重复原LayerNorm，也使一份输入的实际首动作位置分量最多变化 {1000*renorm_xyz:.3f} 毫米。它并非动作层面的恒等操作；这项变化在重复请求中稳定复现。仍不能据此断言它导致此前闭环成功率变化。</p>'
 results+='<h3>实际首动作的位置变化</h3>'+table(['干预','输入','3个场景的平均RMSE（毫米）','最大坐标变化（毫米）'],coordinate_rows)
 results+='<h3>冻结干预相对原策略：相同输入</h3>'+table(['干预','次数','动作块完全相同','首动作完全相同','首动作最大绝对差'],rows)
 fig,axs=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
 colors=['#779d81','#bd9953','#246449'];group_info=[('xyz','Translation RMSE (m)'),('rot6d','Rotation-6D parameter RMSE'),('gripper','Gripper RMSE (command units)')]
 for ax,(group,ylabel)in zip(axs,group_info):
  for i,arm in enumerate(arm_names):
   rows=[r for r in summary['comparisons']if r['comparison']==f'{arm}_r0_vs_identity']
   for noise,marker in [(False,'o'),(True,'x')]:
    vals=[r['first_action']['groups'][group]['rmse']for r in rows if r['input'].startswith('noise')==noise]
    ax.scatter(np.array([i-.1,i,i+.1])+(.06 if noise else -.06),vals,marker=marker,color=colors[i],label=('Noise sigma=.20'if noise else'Clean')if i==0 else None)
  ax.set_xticks(range(3),['Original\n+ norm','Adapter','Adapter\n+ norm'],fontsize=9);ax.set_ylabel(ylabel);ax.spines[['top','right']].set_visible(False)
 axs[0].legend(fontsize=8);fig.suptitle('First-action change versus original policy — 3 fixed scenes per condition')
 fig.savefig(A/'action-drift.svg');fig.savefig(A/'action-drift.png',dpi=180);plt.close(fig)
 results+='<img src="assets/repeatability-20260921/action-drift.svg" alt="固定输入下首动作变化，按位置、旋转参数和夹爪分别显示" style="width:100%;height:auto"><p>每个点是一份固定初始输入；圆点为干净，叉号为噪声。图中展示预定第一次干预输出；第二次输出也完整保留并比较。位置单位为米，rot6d误差不是角度。动作变化既不等于任务收益，也不等于控制退化。</p>'
inputs_html=''
if capture:
 for condition in ['clean','noise']:
  rows=[r for r in capture['records']if r['condition']==condition]
  inputs_html+='<h3>'+('干净初始输入'if condition=='clean'else'主相机噪声 σ=0.20')+'</h3><div style="display:flex;gap:16px;flex-wrap:wrap">'+''.join(f'<figure style="margin:0;max-width:30%"><img src="assets/repeatability-20260921/inputs/{r["name"]}.png" alt="Seed {r["seed"]} 主相机输入" style="width:100%"><figcaption>Seed {r["seed"]} · <a href="assets/repeatability-20260921/inputs/{r["name"]}.json">完整三相机请求</a></figcaption></figure>'for r in rows)+'</div>'
body=f'''<div class="hero"><div class="eyebrow">Fixed-input diagnostic / 2026.09.21</div><h1>同一份观测，<br><em>是否产生同一个动作？</em></h1><p class="lead">先量化动作重复性，再解释表征干预。两个独立进程、6份固定输入、54个完整生成块；本轮没有执行闭环轨迹。</p></div>
<section id="status"><h2>{status}</h2>{interpretation}<p>更新于 {time.strftime('%Y-%m-%d %H:%M:%S %Z')}。<a href="norm.html">上一轮80次闭环主要评测</a>已全部完成，适配器后再归一化没有显示明确净收益。</p></section>
<section id="design"><h2>冻结方案</h2><p>使用此前清单的前三个场景400000、400001、400002，不按成功或失败选择。保持原来的十选一指令随机数消耗，逐一核对原初始主相机图像、EEF状态与指令。采集阶段未执行任何机器人动作；原控制循环在该阶段打印的失败计数不属于策略评测。</p><p>每个场景保存干净及 σ=0.20 主相机噪声请求，包含原生三相机PNG、完整提示词和状态。调用原ModelClient.step的实际序列化路径，核对PNG解码逐像素一致。每次固定请求前重置动作缓冲区，保留原模型、BF16、10次去噪、随机种子42及原缓存设置，关闭编译。</p>{table(['进程','固定顺序','生成块数'],[['A','原策略：6份请求×2次',12],['B（重新加载）','原策略6份×1次，再归一化／适配器／适配器再归一化各6份×2次',42]])}<p>只读记录完整原始生成块，以及经过原策略边界处理的实际首动作，分别比较，不能把未执行的生成块当作完整控制轨迹。</p></section>
<section id="results"><h2>重复性与干预差异</h2>{results}</section>
<section id="inputs"><h2>实际保存的输入</h2>{inputs_html}</section>
<section id="next"><h2>这些结果能回答什么</h2><p>本实验分离同输入推理波动与表征干预引起的动作变化。它仅覆盖3个复用场景的初始观测，不能验证接触阶段、相机视角泛化、任务成功率或RAE优于VAE。</p><p>下一步在输入重放有效后，固定原策略在相同3个场景的干净／噪声条件下各重放两次，最多12条闭环，记录第一处观测或动作分歧。完整轨迹记录另行冻结后启动；若重复性不稳定，先定位原因，再考虑下一项表征改进。</p></section>
<section id="artifacts"><h2>复现记录</h2><p>执行修正：首次调用后发现原生预处理会修改请求字典，完整性检查及时停止。改为每次从同一份JSON重新解码，复现原WebSocket语义。一次后续加载因测试门槛检查被主动终止，未生成动作。两次记录都已保留；最终比较不混入这些工程尝试。详见请求隔离修正说明。</p><p>{links}</p><p><a href="assets/repeatability-20260921/reproduction-scripts.tar.gz">冻结脚本与场景清单</a> · <a href="assets/repeatability-20260921/trace-data.tar.gz">全部请求与逐块动作数据</a> · <a href="assets/repeatability-20260921/action-drift.png">科学图PNG</a></p><p>公开策略与小适配器沿用<a href="robustness.html#artifacts">已有权重与来源记录</a>。推理权重没有重新训练。</p></section>'''
if not summary:body=body.replace(' · <a href="assets/repeatability-20260921/action-drift.png">科学图PNG</a>','')
old=(SITE/'norm.html').read_text();style=''.join(re.findall(r'<link[^>]*rel="stylesheet"[^>]*>',old))+''.join(re.findall(r'<style.*?</style>',old,re.S))
page=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 动作重复性诊断</title>{style}</head><body><header><a href="index.html">Embodied Research Notebook</a><span>动作重复性 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / REPEATABILITY</p><nav aria-label="报告目录">'+''.join(f'<a href="#{k}">{v}</a>'for k,v in [('status','状态'),('design','固定方案'),('results','重复性结果'),('inputs','实际输入'),('next','解释与下一步'),('artifacts','复现记录')])+f'</nav></aside><main>{body}</main></div></body></html>'
(SITE/'repeat.html').write_text(page)
index=SITE/'index.html';text=index.read_text();callout=f'<div class="callout" id="repeat-latest"><strong>动作重复性诊断 · {complete}/54：</strong>{status}。6份固定请求、两个独立进程，区分推理重复性与表征干预。<a href="repeat.html">查看诊断报告</a>。</div>'
if'id="repeat-latest"'in text:text=re.sub(r'<div class="callout" id="repeat-latest">.*?</div>',callout,text,flags=re.S)
else:text=text.replace('<main>','<main>'+callout,1)
index.write_text(text)
for p in A.glob('*.svg'):p.write_text('\n'.join(x.rstrip()for x in p.read_text().splitlines())+'\n')
print(json.dumps({'page':str(SITE/'repeat.html'),'final':bool(summary),'chunks':complete}))
