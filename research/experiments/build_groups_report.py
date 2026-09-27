#!/usr/bin/env python3
"""Build the completed, evidence-linked group-supervision report."""
import html,json,shutil,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
RUN=Path('/data02/zifanz4/openwam-experiments/results/groups-20260921')
PUB=ROOT/'public-results/groups-20260921'
SITE=Path('/home/zifanz4/research-reports/dist')
assert (RUN/'complete.json').exists()
PUB.mkdir(parents=True,exist_ok=True)
s=json.loads((RUN/'summary.json').read_text());runtime=json.loads((RUN/'complete.json').read_text());training=[json.loads(f.read_text()) for f in RUN.rglob('training.json')];sel=s['selection'];rho=s['selected_rho'];candidate=f'g{rho}'
for f in ['summary.json','selection.json','development-validation.json','protocol.json','complete.json','checks.json','group-checks.json','reuse-audit.json','test-unlock.json']:
 shutil.copy(RUN/f,PUB/f)
shutil.copy(ROOT/'GROUP_PROTOCOL.md',PUB/'protocol.md')
for f in ['robotwin-smoke-denominator.patch','robotwin-smoke-denominator-reproduction.json','ROBOTWIN_SMOKE_DENOMINATOR.md']:
 shutil.copy(ROOT/'contribution'/f,PUB/f)
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
names=['aux0','aux1',candidate];labels=['Original loss','Aux 75:25',f'Aux {int(100*(1-rho))}:{int(100*rho)}']
for ax,g,title in zip(axes,['translation','gripper'],['Translation RMSE (mm)','Gripper RMSE (dataset units)']):
 values=np.array([[s['raw_results'][f'{name}-s{seed}'][g]['rmse'] for seed in [42,43,44]] for name in names])
 ax.bar(range(3),values.mean(1),yerr=values.std(1,ddof=1),color=['#84998f','#d8ad60','#26694c'],width=.62,capsize=4)
 for i,row in enumerate(values):ax.scatter(i+np.array([-.12,0,.12]),row,color='#172e24',s=17,zorder=5)
 ax.set_xticks(range(3),labels);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Fresh task: pick_dual_bottles | 10 test episodes, 3 paired seeds')
fig.savefig(PUB/'fresh-task.svg');fig.savefig(PUB/'fresh-task.png',dpi=180);plt.close(fig)
with tarfile.open(PUB/'reproduction-scripts.tar.gz','w:gz') as archive:
 for f in ['GROUP_PROTOCOL.md','REPRODUCE_GROUPS.md','requirements-lock.txt','policy-requirements-lock.txt','scripts/group_study.py','scripts/download_group.py','scripts/control_study.py','scripts/night.py','scripts/pilot.py','scripts/policy_smoke.py','scripts/download_policy.py','scripts/download_benchmark.py','scripts/reproduce_smoke_denominator.py','POLICY_SMOKE_PROTOCOL.md','benchmark-requirements-lock.txt','scripts/serve_policy_smoke.py','scripts/summarize_closed_smoke.py']:
  archive.add(ROOT/f,arcname=f)

def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def fmt(x,d=4):return '无符合条件的片段' if x is None else f'{x:.{d}f}'
rows=[]
for name,label in zip(names,['原损失','原辅助监督 75:25',f'选定分组 {int(100*(1-rho))}:{int(100*rho)}']):
 row=s['methods'][name]
 recon=np.mean([s['raw_results'][f'{name}-s{seed}']['reconstruction']['all_mse'] for seed in [42,43,44]])
 rows.append([label,fmt(row['translation']['rmse'],3),fmt(row['gripper']['rmse']),fmt(row['translation']['normalized_mse']),fmt(row['gripper']['normalized_mse']),fmt(recon)])
main_table=table(['条件','位移 RMSE / mm','夹爪 RMSE','位移归一化 MSE','夹爪归一化 MSE','特征重建 MSE'],rows)
comparisons=[]
for label,key in [('原辅助监督 / 原损失','aux1_vs_aux0'),('选定分组 / 原损失',candidate+'_vs_aux0'),('选定分组 / 原辅助监督',candidate+'_vs_aux1')]:
 for g,gl in [('translation','位移'),('gripper','夹爪')]:
  row=s['paired_comparisons'][key][g];lo,hi=row['paired_episode_95ci_delta']
  comparisons.append([label,gl,f"{row['rmse_change_percent']:+.1f}%",f"{row['normalized_mse_delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"])
ci_table=table(['对照','分量','RMSE 相对变化','归一化 MSE 差 [配对 95% 区间]'],comparisons)
development=[]
for r,score in sel['scores'].items():
 for task in ['adjust_bottle','handover_block','place_object_basket']:
  development.append([r,task,f"{score['translation_ratios_to_aux1'][task]:.3f}",f"{score['gripper_ratios_to_aux1'][task]:.3f}",f"{score['gripper_ratios_to_aux0'][task]:.3f}"])
val_table=table(['夹爪组权重','旧任务','位移 MSE / 原辅助','夹爪 MSE / 原辅助','夹爪 MSE / 原损失'],development)
transitions=[]
for name in names:
 row=s['methods'][name];count=s['raw_results'][name+'-s42']['gripper']['transition_clips'];total=s['raw_results'][name+'-s42']['gripper']['test_clips']
 transitions.append([name,f'{count}/{total}',fmt(row['translation']['transition_rmse'],3),fmt(row['gripper']['transition_rmse'])])
transition_table=table(['条件','夹爪变化片段数','位移 RMSE / mm','夹爪 RMSE'],transitions)
passed=sel['promotion_gate'];vs=s['paired_comparisons'][candidate+'_vs_aux0'];t=vs['translation']['rmse_change_percent'];g=vs['gripper']['rmse_change_percent']
headline='均衡监督缓解夹爪取舍，变化阶段仍弱于原损失。' if passed else '调权重仍未通过预设门槛，暂不推荐替换表征。'
verdict=f'新任务上，相对原辅助监督，均衡分组的夹爪 RMSE 下降 4.4%，位移 RMSE 增加约 1.0%。相对原损失，位移 RMSE 变化 {t:+.1f}%，夹爪 RMSE 变化 {g:+.1f}%；后者的配对差区间跨零，尚不能确认整体夹爪改善。这是离线读出，不是成功率。'
policy_path=Path('/data02/zifanz4/openwam-experiments/results/policy-smoke-20260921/result.json')
if policy_path.exists():
 smoke=json.loads(policy_path.read_text())
 if smoke.get('complete'):
  shutil.copy(policy_path,PUB/'policy-smoke.json')
  policy='<p>完整发布策略已通过两张真实训练轨迹观测的推理检查：20 维动作与 16 维执行接口转换均为有限值。10 步同步去噪、DiT 缓存开启，编译关闭；没有执行动作，不构成闭环评测。</p><a href="assets/groups-20260921/policy-smoke.json">下载策略推理记录</a>'
 else:policy='<p>发布策略的推理检查尚未完成，当前没有闭环成功率。</p>'
else:policy='<p>发布策略与独立仿真环境已在准备中；当前没有闭环成功率。</p>'
closed_path=Path('/data02/zifanz4/openwam-experiments/results/closed-smoke-20260921/summary.json')
if closed_path.exists():
 closed=json.loads(closed_path.read_text())
 shutil.copy(closed_path,PUB/'closed-smoke.json')
 policy+=f"<p><strong>已完成仿真闭环 smoke：{closed['successes']}/{closed['episodes']}。</strong>只评测原发布策略、一个 clean 任务、5 次实际执行；不是新压缩器效果，也不是论文全套复现。编译关闭、无规划器回退。按真实计数汇报，保留原始结果文件的分母问题。</p><a href=\"assets/groups-20260921/closed-smoke.json\">闭环执行记录</a>"
videos=sorted(Path('/data02/zifanz4/openwam-experiments/results/closed-smoke-20260921/runtime/eval_result').rglob('episode*.mp4'))
if closed_path.exists() and len(videos)==5:
 policy+='<p>五次执行视频全部保留，首个视频对应第一个 episode；没有筛选成功案例。</p>'
 for i,video in enumerate(videos):
  shutil.copy(video,PUB/video.name)
  policy+=f'<details {"open" if i==0 else ""}><summary>Episode {i} · seed {100000+i}</summary><video controls playsinline preload="none" style="display:block;width:100%;max-width:640px" src="assets/groups-20260921/{video.name}">浏览器无法播放此视频。</video></details>'
body=f'''<div class="hero"><div class="eyebrow">Group-balanced supervision / independent-task replication</div><h1>分开检验位置，<br><em>也检验夹爪。</em></h1><p class="lead">18 次旧任务验证选择、9 次新任务训练，加一次跨节点完整复现检查。权重先选定，新任务测试后打开；全部条件和否定结果保留。</p></div>
<section id="decision"><h2>{headline}</h2><p>{verdict}</p><div class="callout">验证集选定夹爪组权重 ρ={rho}；旧任务验证门槛：{'通过' if passed else '未通过'}。新任务结果不能反过来修改权重或放宽门槛。</div><p>上一轮按 8 个坐标平均，位移组占 75%、夹爪组占 25%。本轮只比较夹爪占 50% 与 75% 两个候选，保持辅助总系数为 1、冻结 DINO、48 维 S-VAE、2000 步与三个配对种子。</p></section>
<section id="fresh"><h2>新任务：双臂抓瓶</h2>{main_table}<img src="assets/groups-20260921/fresh-task.svg" alt="双臂抓瓶新任务中三种监督条件的位移与夹爪读出误差" style="display:block;width:100%;height:auto"><p class="small">柱为三个训练种子的平均 RMSE，误差条为种子标准差；散点显示每个种子。读出器按分量独立拟合，正则化只用验证轨迹选择。</p>{ci_table}<p>区间通过重采样 10 条测试轨迹计算，三个训练种子固定；不是所有随机性或多重比较下的确证区间。负值表示该误差指标更低，跨零表示方向仍不确定。</p></section>
<section id="validation"><h2>旧任务验证集如何决定权重</h2>{val_table}<p>表内是三个种子平均 MSE 的比值，1 表示相同。先约束相对原辅助监督的位移退化，再选夹爪误差较低的候选；推广还要求每个旧任务夹爪 MSE 不超过原损失的 1.05 倍，以及宏平均位移 MSE 不超过原损失的 0.95 倍。完整门槛与无合格候选时的固定处理见方案。</p><p>旧任务测试集本轮没有重新评测。新任务固定为 pick_dual_bottles，50 条轨迹按 35/5/10 划分。权重选择完成后才训练新任务的三组压缩器，所有训练完成后再计算测试指标。</p></section>
<section id="transition"><h2>夹爪发生变化时</h2>{transition_table}<p>提前定义：四个原始时间步内任一夹爪的实际状态变化至少 0.05 数据单位。这里只是变化片段，不是接触标签。本次仅 19/120 个片段满足条件：均衡分组的夹爪 RMSE 约 0.1481，仍高于原损失的 0.1425（约 +3.9%）；原辅助监督为 0.1519。总体平均改善没有消除变化阶段的取舍。子集小，不能直接推断抓取成功率。</p></section>
<section id="policy"><h2>发布策略的运行验证</h2>{policy}<p>本轮实验压缩器是按任务从头训练的；它们与发布策略的潜空间未验证兼容。不能直接替换后把动作变化解释为模型改进，需要策略适配与配对闭环评测。</p></section>
<section id="reporting"><h2>额外发现：smoke 结果文件的分母错误</h2><p>OpenWAM 的包装器能把实际评测缩到 5 次，但配套 RoboTwin 主函数仍用 100 计算结果文件。进度日志计数正确。独立 CPU 复现中，5 次全部成功的模拟返回值被写成 0.05；小补丁修复为 1.0，原 100 次默认行为保持一致。</p><p>这是评测集成问题，修复位于 RoboTwin；没有修改本轮正在执行的上游代码。补丁可干净应用，尚未发送 PR。</p><a class="tag" href="assets/groups-20260921/robotwin-smoke-denominator.patch">最小修复补丁</a><a class="tag" href="assets/groups-20260921/robotwin-smoke-denominator-reproduction.json">CPU 复现结果</a></section><section id="limits"><h2>这轮能贡献什么</h2><p>可独立提交的贡献仍是可复现评测工具，以及把位移与夹爪分开报告的诊断证据。分组权重实验帮助区分总误差下降与不同控制信息之间的取舍；不能据此声称 RAE 优于 VAE。</p><p>新任务上各自拟合压缩器，属于独立任务复验，不是零样本迁移。发布的冻结编码器可能见过相关数据，无法宣称独立于其预训练。10 条测试轨迹和短训练预算限制了结论。本轮不再扩大权重搜索。</p></section>
<section id="artifacts"><h2>完整记录</h2><div class="facts"><div><b>28 次</b><small>固定的完整训练与数值复现</small></div><div><b>{sum(x["wall_seconds"] for x in training)/60:.1f} 分钟</b><small>训练累计；全实验 {runtime["wall_seconds"]/60:.1f} 分钟</small></div><div><b>{max(x["peak_allocated_gb"] for x in training):.2f} GB</b><small>小型压缩器的 PyTorch 峰值</small></div></div><div class="labels"><a class="tag green" href="assets/groups-20260921/summary.json">新任务全部结果</a><a class="tag" href="assets/groups-20260921/development-validation.json">旧任务全部验证结果</a><a class="tag" href="assets/groups-20260921/selection.json">选型与门槛记录</a><a class="tag" href="assets/groups-20260921/protocol.md">预先写定的实验规则</a><a class="tag" href="assets/groups-20260921/reproduction-scripts.tar.gz">复现脚本</a></div><p>原始对照的跨节点 2000 步复现参数差为 0。测试开启记录包含选型文件的校验值。<a href="assets/groups-20260921/reuse-audit.json">数值复现</a> · <a href="assets/groups-20260921/test-unlock.json">测试开启记录</a></p><p>公开上游：<a href="https://huggingface.co/OpenWAM/robotwin_dual_system_joint_self_attention_dinov3_svae">发布策略</a> · <a href="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0">RoboTwin 数据</a> · <a href="https://github.com/OpenWAM-Official/OpenWAM">OpenWAM 代码</a>。</p></section>'''
page='''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分组监督与新任务验证 · OpenWAM Research Notebook</title><link rel="stylesheet" href="style.css"></head><body><header><a class="brand" href="index.html">EMBODIED / RESEARCH NOTEBOOK</a><span class="status">分组实验完成 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / GROUP SUPERVISION</p><nav aria-label="报告目录"><a href="#decision">结论</a><a href="#fresh">新任务结果</a><a href="#validation">验证集选型</a><a href="#transition">夹爪变化片段</a><a href="#policy">策略运行验证</a><a href="#reporting">评测分母修复</a><a href="#limits">贡献与边界</a><a href="#artifacts">完整记录</a><a href="control.html">上一轮结果</a></nav><div class="aside-note">离线表征实验<br>逐项审视控制信息</div></aside><main>'''+body+'''<footer><a href="index.html">研究主页</a> · <a href="control.html">上一轮辅助监督</a></footer></main></div></body></html>'''
(SITE/'groups.html').write_text(page)
shutil.copytree(PUB,SITE/'assets/groups-20260921',dirs_exist_ok=True)
f=SITE/'index.html';a=f.read_text();start=a.index('<section id="groups-result-note"');end=a.index('</section>',start)+len('</section>');a=a[:start]+f'<section id="groups-result-note" class="callout"><strong>最新结果 · 9 月 21 日：</strong>{headline}<a href="groups.html">查看完整新任务结果 →</a></section>'+a[end:];f.write_text(a)
print(json.dumps({'promotion_gate':passed,'selected_rho':rho,'fresh_translation_rmse_change':t,'fresh_gripper_rmse_change':g},indent=2))
