"""Generate a portable scientific report; does not publish or edit Sites checkout."""
from pathlib import Path
import datetime,json,html,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/home/zifanz4/openwam-experiments');D=B/'studies/layers-20260922/development';S=B/'studies/layers-20260922';O=B/'reports/layers-20260922';O.mkdir(parents=True,exist_ok=True)
audit=json.loads((D/'audit.json').read_text());ridge=json.loads((D/'development-summary.json').read_text());mlp=json.loads((D/'mlp/summary.json').read_text());description=json.loads((D/'descriptive.json').read_text());runtime=Path('/data02/zifanz4/openwam-experiments/results/layers-20260922');fresh=json.loads((runtime/'fresh/progress-v2.json').read_text());now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4.7),sharey=True);x=np.arange(3)
for ax,method,title in zip(axs,['ridge','mlp'],['Grouped ridge (primary readout)','Fixed MLP (secondary readout)']):
    v=audit['source_compression_interaction_descriptive'][method]
    for shift,cond,label,color in [(-.18,'clean','Clean','#416c94'),(.18,'noise0.10','Noise sigma = 0.10','#d28438')]:
        yy=[v[c][cond]['relative_change_percent'] for c in ['raw','pca','svae']];bars=ax.bar(x+shift,yy,.34,label=label,color=color)
        for b,y in zip(bars,yy):ax.annotate(f'{y:+.1f}%',(b.get_x()+b.get_width()/2,y),xytext=(0,4 if y>=0 else -13),textcoords='offset points',ha='center',fontsize=10)
    ax.axhline(0,color='#263947',linewidth=1);ax.set_xticks(x,['Raw 768','PCA 48','S-VAE 48']);ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylim(-39,59)
axs[0].set_ylabel('K4 vs L12: relative change in error (%)\nLower is better');axs[1].legend(frameon=False,loc='upper left')
fig.suptitle('Development evidence: source ranking changes after compression',fontsize=15,y=.98)
fig.text(.07,.03,'15 previously used episodes, 3 tasks. S-VAE: mean over 3 reducer seeds.\nWithin each bar, source dimensions match. Raw vs 48D has different readout capacity. Descriptive comparisons.',fontsize=9,color='#435563')
fig.subplots_adjust(bottom=.23,top=.83,wspace=.11);fig.savefig(O/'development-crossover.png',dpi=180,bbox_inches='tight');fig.savefig(O/'development-crossover.pdf',bbox_inches='tight');plt.close(fig)
rows=''
for method,label in [('ridge','Ridge，主读出'),('mlp','MLP，次要检查')]:
    for comp,name in [('raw','原始 768 维'),('pca','PCA 48 维'),('svae','S-VAE 48 维')]:
        item=audit['source_compression_interaction_descriptive'][method][comp]
        rows+=f'<tr><td>{label}</td><td>{name}</td><td>{item["clean"]["relative_change_percent"]:+.2f}%</td><td>{item["noise0.10"]["relative_change_percent"]:+.2f}%</td></tr>'
status='新轨迹仍在采集；尚无新确认集结论。';confirmation_section=''
cp=runtime/'confirmation/summary.json';ca=runtime/'confirmation/audit.json'
wrist_complete=(runtime/'wrist-baseline/complete.json').exists()
wrist_scored=(runtime/'wrist-baseline/fresh-summary.json').exists()
wrist_status='腕部同视角基线已训练完成，正在等待配对评估。' if wrist_complete else '腕部同视角训练基线尚未完成。'
if wrist_scored:wrist_status='腕部同视角基线已完成配对评估，见独立相机比较附件。'
shutil.copy2(B/'contribution/svae-sample-provenance.patch',O/'svae-sample-provenance.patch')
shutil.copy2(B/'scripts/layers/audit_public_predictions.py',O/'audit_public_predictions.py')
if cp.exists():
    conf=json.loads(cp.read_text());passed=ca.exists() and json.loads(ca.read_text()).get('passed');status='新轨迹评分完成，独立审计'+('通过。' if passed else '尚未通过；以下仍为待审计结果。')
    p=conf['primary_paired']['ridge'];q=conf['primary_paired']['mlp'];confirmation_section=f'<section><h2>新轨迹确认</h2><p>60 条新任务内专家可行轨迹，720 个采样点；各压缩种子、噪声副本和帧先在轨迹内平均，不计为独立样本。没有重新拟合。</p><p>主读出噪声 E 变化 <strong>{p["noise0.10"]["E_relative_change_percent"]:+.2f}%</strong>，95% 配对区间 {p["noise0.10"]["relative_percent95"][0]:+.2f}% 至 {p["noise0.10"]["relative_percent95"][1]:+.2f}%；干净 E 变化 {p["clean"]["E_relative_change_percent"]:+.2f}%。固定 MLP 噪声 E 变化 {q["noise0.10"]["E_relative_change_percent"]:+.2f}%。</p><p>预设主收益标准：<strong>{"达到" if conf["predeclared_primary_success"] else "未达到"}</strong>。离线读出不等于策略闭环收益。</p><p><a href="confirmation-summary.json">全部条件及逐轨迹数据</a> · <a href="confirmation-audit.json">独立审计</a> · <a href="confirmation-predictions.npz">原始预测</a> · <a href="confirmation-labels-scales.npz">标签与训练尺度</a> · <a href="audit_public_predictions.py">仅依赖 NumPy 的统计复算脚本</a></p></section>'
    shutil.copy2(cp,O/'confirmation-summary.json')
    if ca.exists():shutil.copy2(ca,O/'confirmation-audit.json')
    import torch
    labels=np.load(runtime/'confirmation/vectors.npz');meta=json.loads((runtime/'confirmation/sample-manifest.json').read_text());coef=np.load(runtime/'readout/coefficients.npz');models=torch.load(runtime/'readout/mlp/models.pt',map_location='cpu',weights_only=False)
    arr={k:labels[k] for k in ['target','reference','condition']};arr.update(task=np.array([r['task'] for r in meta]),episode=np.array([r['seed'] for r in meta]))
    for task in conf['primary_paired']['ridge']['clean']['task_order']:
        arr['ridge/'+task+'/scale']=np.r_[coef[f'{task}/proprio/0/ys'],coef[f'{task}/proprio/1/ys']];arr['mlp/'+task+'/scale']=models['scales'][task+'/proprio'][3].numpy()
        for name in ['proprio']+[s+'_'+v for s in ['L12','L6','K4'] for v in ['raw','pca','svae42','svae43','svae44']]:
            assert np.array_equal(arr['ridge/'+task+'/scale'],np.r_[coef[f'{task}/{name}/0/ys'],coef[f'{task}/{name}/1/ys']])
            assert np.array_equal(arr['mlp/'+task+'/scale'],models['scales'][task+'/'+name][3].numpy())
    np.savez(O/'confirmation-labels-scales.npz',**arr);shutil.copy2(runtime/'confirmation/predictions.npz',O/'confirmation-predictions.npz')
    if wrist_scored:shutil.copy2(runtime/'wrist-baseline/fresh-summary.json',O/'wrist-camera-summary.json')
package={'updated_at':now,'status':status,'primary':'K4-SVAE48 versus L12-SVAE48','development':{'ridge':ridge['primary_paired'],'mlp':mlp['primary_paired'],'crossover':audit['source_compression_interaction_descriptive'],'audit_passed':audit['passed'],'independent_episodes':15,'previously_exposed':True},'fresh_collected':fresh['accepted_counts'],'fresh_attempts':len(fresh['attempts']),'fresh_refits':0,'policy_stage':'not launched: development gate failed; native bounded throughput profile incomplete','publication':'existing public project unavailable to current Sites connector; local report only'}
(O/'summary.json').write_text(json.dumps(package,ensure_ascii=False,indent=2)+'\n');shutil.copy2(D/'audit.json',O/'development-audit.json');shutil.copy2(D/'descriptive.json',O/'development-descriptive.json');shutil.copy2(D/'development-summary.json',O/'development-ridge.json');shutil.copy2(D/'mlp/summary.json',O/'development-mlp.json')
pro=description['controls']['mlp']['proprio']['clean']['E'];vbase=audit['source_compression_interaction_descriptive']['mlp']['svae']['clean']['L12_E'];vcand=audit['source_compression_interaction_descriptive']['mlp']['svae']['clean']['K4_E']
doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 层级表征与压缩实验</title><style>
:root{{color-scheme:light;--ink:#20333e;--muted:#586a75;--line:#dce3e5;--accent:#1c6470}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f5f2;color:var(--ink);font:17px/1.75 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:36px 24px 70px}}.eyebrow{{font-size:13px;letter-spacing:.1em;color:var(--accent);font-weight:700}}h1{{font-size:clamp(28px,5vw,43px);line-height:1.2;margin:12px 0 20px}}h2{{font-size:25px;margin-top:0}}p{{margin:12px 0}}section{{background:white;padding:26px;margin-top:22px;border:1px solid var(--line);border-radius:12px}}.note{{color:var(--muted);font-size:14px}}.banner{{border-left:4px solid #d28438;padding:12px 18px;background:#fff4e5}}a{{color:#196174}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted)}}code{{font-size:.9em}}li{{margin:7px 0}}.metric{{font-weight:700}}footer{{font-size:14px;color:var(--muted);margin-top:24px}}</style><main>
<div class="eyebrow">OPENWAM · REPRESENTATION STUDY · 2026-09-22</div><h1>多层特征的优势，能穿过 48 维压缩吗？</h1>
<p>同一 DINOv3 骨干、相同压缩预算，比较末层 L12、中间层 L6 与末四层等权平均 K4。主候选始终是 K4-S-VAE；结果不好也不换成其他候选。</p><p class="note">本地报告更新：{html.escape(now)}。原公网报告站点当前无法更新，不能把其旧页面视为当前状态。</p>
<div class="banner"><strong>开发集主假设未得到支持。</strong>噪声下 K4-S-VAE 的均衡误差比 L12-S-VAE 更高：Ridge +45.77%，MLP +14.79%。{status}</div>
{confirmation_section}
<section><h2>已完成的比较</h2><p>3 项任务，每项 35 条训练轨迹及 5 条曾使用过的开发轨迹。9 个 S-VAE（3 特征来源 × 3 种子）、3 个 PCA、原始特征对照、分组 Ridge 与固定 MLP 均已完成。独立重算了 282 个 Ridge 条件和 192 个 MLP 条件；15 条零噪声路径及固定训练样本的跨机器输出完全一致。</p><p>当前新轨迹：{html.escape(str(fresh['accepted_counts']))}；共保留 {len(fresh['attempts'])} 次采集尝试。目标为每项任务按连续种子取前 20 条专家可行且完整的轨迹；名单不依据模型表现选择。</p><img src="development-crossover.png" alt="开发集上，K4 相对 L12 的误差在原始768维下更低，压缩48维后噪声误差更高。"><p class="note">负值表示 K4 误差下降，正值表示升高。图为开发集描述性结果；原始特征和压缩特征的读出维度不同，只能在各自相同维度内比较来源。</p><div class="scroll"><table><thead><tr><th>读出</th><th>特征</th><th>干净 E 变化</th><th>噪声 E 变化</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>这个结果能解释什么</h2><p>原始 768 维中 K4 的总体读出误差较低；使用这套 PCA 或 S-VAE 压到 48 维后，噪声条件下的排序反转。两个读出族方向一致，值得在新轨迹上确认。</p><p>噪声误差分解显示，开发集上 K4-S-VAE 的固定读出预测偏移更大。分解使用恒等式 <code>Δ误差 = 偏移² + 2 × 原误差 × 偏移</code>，验证了所有条件。它支持“当前压缩与读出配合更敏感”的描述，尚不能区分压缩目标、优化预算、归一化或读出正则的作用。</p><p><strong>一个重要限制：</strong>固定 MLP 的仅本体基线，干净 E={pro:.4f}，优于视觉+本体的 L12-S-VAE {vbase:.4f} 和 K4-S-VAE {vcand:.4f}。因此这些未来状态标签可能主要由本体状态解释；视觉表征排序不能直接等同于控制价值。新确认集将保留这个对照。</p><p>全部来源都使用同一个预训练骨干。本实验不是 RAE 对像素 VAE 的完整比较，也没有证明信息不可逆丢失。</p></section>
<section><h2>下一步与决策门槛</h2><ol><li>冻结全部压缩器、读出器和训练尺度。新数据要求与已有轨迹在文件及首帧图像哈希上无重复；这不是对未知预训练语料的排除证明。</li><li>三任务各 20 条新轨迹全部通过检查后，一次性评估干净、两档噪声和跨相机输入，不进行重新拟合。主比较用任务内轨迹配对重采样；种子和噪声副本不增加独立样本数。</li><li>主收益要求噪声 E 至少降低 10%、95% 区间支持改善、干净退化上界不超过 3%，并满足逐任务及标签组的限制。</li><li>开发门槛已失败，因此本轮不启动条件式大模型适配。原生训练的有界测速也未完成计划计时；没有测得两卡完整吞吐，不能据此宣称两卡无法训练。</li></ol><p>跨相机比较同时改变可见性和遮挡，不能等同于纯几何不变性。{wrist_status}夹爪变化分析属于明确标注的补充诊断，不是接触传感标签。旧 80 次闭环未重复，本轮没有新的闭环收益结论。</p></section>
<section><h2>对原项目的可交付内容</h2><p>优先交付可复查的“特征来源 × 压缩 × 控制读出”比较入口、逐样本来源记录、分组指标、本体与错配对照，以及全部正负结果。新增可选样本来源记录补丁已完成，4 项针对性 CPU 测试及 Ruff 通过；尚未进行真实分布式 GPU 采集测试，也未提交上游。<a href="svae-sample-provenance.patch">下载独立补丁</a>。已有独立重建评测补丁保持原范围；不把本次研究脚本冒称为已合并功能，也不因开发结果改动默认表征。</p><p><a href="summary.json">状态与主要结果 JSON</a> · <a href="development-audit.json">独立审计</a> · <a href="development-ridge.json">Ridge 全部条件</a> · <a href="development-mlp.json">MLP 全部条件</a> · <a href="development-descriptive.json">误差分解及全部控制</a> · <a href="development-crossover.pdf">PDF 图表</a></p></section><footer>研究窗口：2026-09-22 13:15 至次日 01:15（CDT），不自动延长。最多 4 张 GPU 并发、48 GPU 小时。本文是本地报告；公网发布未完成。</footer></main></html>'''
(O/'index.html').write_text(doc)
# Validate portability, asset existence, and absence of infrastructure identifiers.
from html.parser import HTMLParser
class Links(HTMLParser):
 def handle_starttag(self,tag,attrs):
  for key,value in attrs:
   if key in ['href','src'] and not value.startswith(('http:','https:','#')):assert (O/value).is_file(),value
Links().feed(doc)
for p in O.iterdir():
 if p.suffix in ['.html','.json']:
  s=p.read_text()
  for forbidden in ['/home/','/data02/','/vol13/','zifanz4','csl.illinois.edu','GPU-','172.22.']:assert forbidden not in s,(p,forbidden)
(O/'artifact-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.name!='artifact-sha256.json'},indent=2)+'\n');print(str(O/'index.html'))
