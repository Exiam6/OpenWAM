"""Assemble audited local report; no publishing or model work."""
from pathlib import Path
import datetime,hashlib,json,runpy,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/home/zifanz4/openwam-experiments');R=Path('/data02/zifanz4/openwam-experiments/results/layers-20260922');O=B/'reports/layers-20260922';S=B/'studies/layers-20260922'
assert json.loads((R/'confirmation/audit.json').read_text())['passed'];assert json.loads((R/'initial-goal/audit.json').read_text())['passed'];assert (R/'describe_confirmation_v3-exit-code.txt').read_text().strip()=='0'
runpy.run_path(str(B/'scripts/layers/build_report.py'),run_name='__main__')
goal=json.loads((R/'initial-goal/fresh-summary.json').read_text());main=json.loads((R/'confirmation/summary.json').read_text());wrist=json.loads((R/'wrist-baseline/fresh-summary.json').read_text());fresh=json.loads((R/'fresh/manifest-v2.json').read_text());phase=json.loads((R/'confirmation/descriptive.json').read_text())
for src,dst in [('initial-goal/fresh-summary.json','initial-goal-summary.json'),('initial-goal/audit.json','initial-goal-audit.json'),('initial-goal/fresh-labels.json','initial-goal-labels.json'),('initial-goal/fresh-predictions.npz','initial-goal-predictions.npz'),('initial-goal/constants.json','initial-goal-training-scales.json'),('confirmation/descriptive.json','confirmation-descriptive.json'),('confirmation/decomposition-precision-diagnostic.json','decomposition-precision-diagnostic.json')]:shutil.copy2(R/src,O/dst)
shutil.copy2(B/'scripts/layers/audit_public_waypoint.py',O/'audit_public_waypoint.py')
points=[]
for method in ['ridge','mlp']:
 a=main['primary_paired'][method]['noise0.10'];points.append(('State readout / '+method,a['E_relative_change_percent'],a['relative_percent95'],'#b56538'))
for method in ['ridge','mlp']:
 a=goal['secondary_K4_SVAE_vs_L12_SVAE'][method]['noise0.10'];points.append(('Initial waypoint / '+method,a['E_relative_change_percent'],a['relative_percent95'],'#24697a'))
fig,ax=plt.subplots(figsize=(10,4.8));yy=[3,2,1,0]
for y,(name,value,ci,color) in zip(yy,points):
 ax.errorbar(value,y,xerr=[[value-ci[0]],[ci[1]-value]],fmt='o',capsize=5,color=color,markersize=8);ax.text(ci[1]+2,y,f'{value:+.1f}%',va='center',fontsize=11)
ax.axvline(0,color='#667680',linewidth=1);ax.axhline(1.5,color='#ccd5d8',linewidth=1,linestyle='--');ax.set_yticks(yy,[p[0] for p in points]);ax.set_xlim(-57,70);ax.set_ylim(-.6,3.6);ax.grid(axis='x',alpha=.15);ax.set_xlabel('K4-SVAE vs L12-SVAE: relative error change (%)\nNoise sigma=0.10; lower is better');ax.spines[['right','top','left']].set_visible(False);fig.suptitle('The same representation change has opposite effects on two readouts',fontsize=14)
fig.text(.07,.025,'Top: prespecified primary endpoint. Bottom: secondary endpoint fixed after development, before fresh scoring.\nSame 60 scenes and 3 fixed reducer seeds. Paired 95% intervals; no policy finetuning or closed-loop benefit claim.',fontsize=9,color='#435563');fig.subplots_adjust(left=.26,bottom=.24,top=.87);fig.savefig(O/'fresh-contrast.png',dpi=180,bbox_inches='tight');fig.savefig(O/'fresh-contrast.pdf',bbox_inches='tight');plt.close(fig)
g=goal['secondary_K4_SVAE_vs_L12_SVAE'];matrix=''
for method in ['ridge','mlp']:
 for name,v in goal['all_source_compressor_E'][method].items():matrix+=f'<tr><td>{method}</td><td>{name}</td><td>{v["clean"]:.4f}</td><td>{v["noise0.10"]:.4f}</td><td>{v["noise0.04"]:.4f}</td></tr>'
per_task=''
for i,t in enumerate(['adjust_bottle','handover_block','place_object_basket']):per_task+=f'<tr><td>{t}</td><td>{g["ridge"]["noise0.10"]["per_task_relative_percent"][i]:+.2f}%</td><td>{g["mlp"]["noise0.10"]["per_task_relative_percent"][i]:+.2f}%</td></tr>'
section=f'''<section><h2>同一改变，对两类信息产生相反效果</h2><img src="fresh-contrast.png" alt="短期状态主评测误差升高，但初始目标位置的次要读出误差下降，均展示配对置信区间。"><p><strong>主评测没有通过预设收益标准。</strong>同样的 K4-S-VAE，在初始目标位置这个补充问题上则显示改善：噪声条件 Ridge 误差变化 {g['ridge']['noise0.10']['E_relative_change_percent']:+.2f}%，95% 区间 [{g['ridge']['noise0.10']['relative_percent95'][0]:+.2f}%, {g['ridge']['noise0.10']['relative_percent95'][1]:+.2f}%]；固定 MLP 为 {g['mlp']['noise0.10']['E_relative_change_percent']:+.2f}%。干净输入分别为 {g['ridge']['clean']['E_relative_change_percent']:+.2f}% 和 {g['mlp']['clean']['E_relative_change_percent']:+.2f}%。</p><p>这个补充诊断在看过开发集、本轮新轨迹评分之前固定。105 条训练轨迹的初始本体状态在各任务内相同；输入仅为初始 head 图像，标签是首次夹爪从 &gt;0.5 降至 ≤0.5 时对应末端的 XY 位置。它是专家事件位置，不能当作物体真值或接触传感标签。</p><p>每任务仅 35 条训练图像；三个来源、PCA/S-VAE/原始特征、全部种子均保留。常数训练均值基线 E={goal['constant_baseline_E']:.4f}；K4-S-VAE 的 Ridge 噪声 E={goal['all_source_compressor_E']['ridge']['K4_svae']['noise0.10']:.4f}，支持此问题需要视觉信息。独立审计重新读取了 60 个原始事件标签、检查 105 个训练样本、重算 270 个预测条件和 6 个配对区间，均通过。</p><p><strong>收益不是所有任务一致。</strong>Ridge 的 adjust_bottle 噪声误差略升，MLP 的 handover_block 误差升高；低噪声下 MLP 总体变化区间跨零。这个结果是同一批 60 条轨迹上的次要证据，不能替代失败的主假设，也不是额外一次独立复现。</p><div class="scroll"><table><tr><th>任务</th><th>目标位置 / Ridge 噪声变化</th><th>目标位置 / MLP 噪声变化</th></tr>{per_task}</table></div><details><summary>展开全部来源与压缩矩阵（不挑选最优者作为主结论）</summary><div class="scroll"><table><tr><th>读出</th><th>表征</th><th>干净 E</th><th>噪声 0.10 E</th><th>噪声 0.04 E</th></tr>{matrix}</table></div></details><p><a href="initial-goal-summary.json">全部目标位置结果</a> · <a href="initial-goal-audit.json">原始标签及预测独立审计</a> · <a href="initial-goal-predictions.npz">原始预测</a> · <a href="initial-goal-labels.json">标签</a> · <a href="initial-goal-training-scales.json">训练均值与尺度</a> · <a href="audit_public_waypoint.py">NumPy 复算脚本</a> · <a href="fresh-contrast.pdf">PDF 图表</a></p></section>'''
camera=''
for method in ['ridge','mlp']:
 for src in ['L12_svae','K4_svae']:
  a=wrist['task_balanced_camera_comparison'][method][src];camera+=f'<tr><td>{method}</td><td>{src}</td><td>{a["head_train_head_input"]["E"]:.4f}</td><td>{a["head_train_wrist_input"]["E"]:.4f}</td><td>{a["wrist_train_wrist_input"]["E"]:.4f}</td></tr>'
views=f'''<section><h2>跨相机失败，包含读出器不匹配的影响</h2><p>已补齐同视角训练基线，全部权重在新轨迹运行前冻结。以 L12-S-VAE / Ridge 为例，head 训练→wrist 测试的 E=2.2597；用相同训练轨迹的 wrist 图像训练同预算读出后，E 降为 0.2446。压缩器始终使用 head 训练版本。</p><div class="scroll"><table><tr><th>读出</th><th>表征</th><th>head→head</th><th>head→wrist</th><th>wrist→wrist</th></tr>{camera}</table></div><p>这说明直接跨相机的失败有很大的读出训练域因素；仍有可见性、遮挡和视角内容差异，不能据此证明几何不变性或真实策略适配成功。</p><p><a href="wrist-camera-summary.json">全部 90 个相机基线条件</a> · <a href="confirmation-descriptive.json">夹爪变化阶段与误差分解</a></p></section>'''
page=(O/'index.html').read_text();page=page.replace('<h1>多层特征的优势，能穿过 48 维压缩吗？</h1>','<h1>目标位置读出改善，短期状态抗噪退化</h1>');marker='<section><h2>已完成的比较</h2>';assert marker in page;page=page.replace(marker,section+views+marker,1)
page=page.replace('独立重算了 282 个 Ridge 条件和 192 个 MLP 条件','开发集独立重算了 282 个 Ridge 条件和 192 个 MLP 条件')
page=page.replace('新确认集将保留这个对照。','新确认集 MLP 仅本体 E=0.1170，L12-S-VAE 为 0.1239，K4-S-VAE 为 0.1120；后者的平移读出更好而夹爪误差更高。因此不能把开发集上的本体优势直接推广，也不能由此断言视觉对控制无用。')
page=page.replace('它支持“当前压缩与读出配合更敏感”的描述','它支持“在短期状态读出上，当前压缩与读出配合更敏感”的描述')
page=page.replace('本文是本地报告；公网发布未完成。','误差分解先因 float32 消去误差触发数值检查，再因 JSON 类型转换失败；原失败均保留。使用同一批预测的 float64 分解与类型修正通过，主结果未改动。本文是本地报告；公网发布未完成。')
(O/'index.html').write_text(page)
status=json.loads((O/'summary.json').read_text());status.update(primary_audit='passed384conditions',secondary_waypoint_audit='passed270conditions',original_primary_passed=False,waypoint_secondary_K4_SVAE_vs_L12_SVAE=g,source_choice_not_universal=True,view_baseline_completed=True,phase_decomposition='v3complete; v1precision/v2serialization failures retained');(O/'summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
from html.parser import HTMLParser
class Links(HTMLParser):
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and not v.startswith(('http:','https:','#')):assert (O/v).exists(),v
Links().feed(page)
for p in O.iterdir():
 if p.suffix in ['.html','.json','.py','.patch']:
  text=p.read_text()
  for bad in ['/home/','/data02/','/vol13/','zifanz4','csl.illinois.edu','GPU-','172.22.']:assert bad not in text,(p,bad)
(O/'artifact-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.name!='artifact-sha256.json'},indent=2)+'\n');print('Local report assembled, links and infrastructure scan passed:',O/'index.html')
