#!/usr/bin/env python3
"""Evidence-linked report for the completed, paired robustness study."""
import html,json,shutil,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
RUN=Path('/data02/zifanz4/openwam-experiments/results/robustness-20260921')
PUB=ROOT/'public-results/robustness-20260921';PUB.mkdir(parents=True,exist_ok=True)
SITE=Path('/home/zifanz4/research-reports/dist')
s=json.loads((RUN/'summary.json').read_text());conditions=s['conditions'];selection=s['adapter_selection']
assert s['paired_checks_all_passed'] and s['source_hashes_all_passed']
for name in ['summary.json','adapter-selection.json','adapter-training.json','adapter-extraction.json','adapter-latent-diagnostics.json','encoder-reference.json','standalone-encoder-check.json','perturbation-checks.json','instruction-seed-check.json','source-hashes.json','source-audit-final.json','postprocessing-fix.json','confirmation-unlock.json','adapter-hook.json','protocol.md','adapter-selected.pt','resource-amendment.md','fixed-scene-amendment.md','fixed-scenes.json','fixed-scene-checks.json','clean-regression-example.json','clean-regression-seed300003.jpg']:
 shutil.copyfile(RUN/name,PUB/name)
for name in ['robotwin-instruction-seed.patch','robotwin-instruction-seed-reproduction.json','ROBOTWIN_INSTRUCTION_SEED.md','openwam-scene-seeded-language.patch','openwam-language-integration.json','OPENWAM_LANGUAGE_PR_DRAFT.md']:
 shutil.copyfile(ROOT/'contribution'/name,PUB/name)
with tarfile.open(PUB/'reproduction-scripts.tar.gz','w:gz') as archive:
 for name in ['ROBUSTNESS_PROTOCOL.md','REPRODUCE_ROBUSTNESS.md','POLICY_SMOKE_PROTOCOL.md','policy-requirements-lock.txt','benchmark-requirements-lock.txt','scripts/robustness_common.py','scripts/robustness_client.py','scripts/serve_robustness.py','scripts/train_restoration.py','scripts/diagnose_adapter_latents.py','scripts/run-robustness.sh','scripts/run-restoration.sh','scripts/run-robust-confirmation.sh','scripts/run-confirmation-gpu0.sh','scripts/run-robustness-confirm-gpu0.sh','scripts/serve_robustness_confirmation.py','scripts/summarize_robustness.py','scripts/reproduce_instruction_seed.py','scripts/replay_confirmation_client.py','scripts/run-fixed-confirmation.sh','scripts/run-fixed-robustness.sh','scripts/check_fixed_scene_replay.py']:
  archive.add(ROOT/name,arcname=name)
 for name in ['fixed-scenes.json','fixed-scene-amendment.md','resource-amendment.md']:
  archive.add(RUN/name,arcname='results/robustness-20260921/'+name)

def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
labels={'baseline-clean':'发布策略 · 干净','baseline-noise':'发布策略 · 噪声','baseline-front':'发布策略 · 换为前方相机','identity-confirm-clean':'原策略 · 干净','identity-confirm-noise':'原策略 · 噪声','adapter-confirm-clean':'适配器 · 干净','adapter-confirm-noise':'适配器 · 噪声'}
base_names=['baseline-clean','baseline-noise','baseline-front'];confirm_names=['identity-confirm-clean','identity-confirm-noise','adapter-confirm-clean','adapter-confirm-noise']
def rows(names):
 out=[]
 for name in names:
  r=conditions[name];lo,hi=r['wilson_95ci']
  out.append([labels[name],f"{r['successes']}/{r['episodes']}",f'{100*lo:.1f}–{100*hi:.1f}%',f"{r['mean_actions_all_episodes']:.1f}",', '.join(str(x['seed'])for x in r['records'])])
 return out
noise_cmp=s['paired_comparisons']['adapter-confirm-noise_vs_identity-confirm-noise'];clean_cmp=s['paired_comparisons']['adapter-confirm-clean_vs_identity-confirm-clean']
noise_original=conditions['identity-confirm-noise']['successes'];noise_adapter=conditions['adapter-confirm-noise']['successes']
clean_original=conditions['identity-confirm-clean']['successes'];clean_adapter=conditions['adapter-confirm-clean']['successes']
gain=100*(1-selection['candidates']['mlp']['noise_ratio'])
headline=f'潜变量恢复误差降低 {gain:.1f}%；固定场景闭环：干净 {clean_original}/5→{clean_adapter}/5，噪声 {noise_original}/5→{noise_adapter}/5（原策略→适配器）。'
if clean_adapter<clean_original or noise_adapter<noise_original:
 verdict='出现新增失败，暂不建议启用这个恢复适配器。'
elif noise_adapter>noise_original:
 verdict='出现小样本正向信号，尚不足以确认普遍改进。'
else:
 verdict='尚无闭环收益证据；当前成功率处于小样本天花板。'
fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
for ax,names,title,ticks in [(axes[0],base_names,'Published policy | baseline starts',['Clean','Noise','Front camera']),(axes[1],confirm_names,'Frozen confirmation | fresh starts',['Original\nclean','Original\nnoise','Adapter\nclean','Adapter\nnoise'])]:
 values=[conditions[n]['success_rate'] for n in names];bounds=np.array([conditions[n]['wilson_95ci']for n in names]).T
 ax.bar(range(len(names)),values,color=['#84998f','#d8ad60','#26694c','#6f8757'][:len(names)],width=.6)
 ax.errorbar(range(len(names)),values,yerr=np.array([np.array(values)-bounds[0],bounds[1]-np.array(values)]),fmt='none',ecolor='#223b31',capsize=4)
 ax.set_xticks(range(len(names)),ticks,fontsize=9);ax.set_ylim(0,1.12);ax.set_title(title,fontsize=11);ax.set_ylabel('Success rate (5 episodes)');ax.spines[['top','right']].set_visible(False)
 for i,n in enumerate(names):ax.text(i,values[i]+.035,f"{conditions[n]['successes']}/5",ha='center',fontsize=10)
fig.savefig(PUB/'closed-loop.svg');fig.savefig(PUB/'closed-loop.png',dpi=180);plt.close(fig)
# Validation-only restoration curve, seed dots expose variability.
training=json.loads((RUN/'adapter-training.json').read_text())
deployed=next(r for r in training if r['kind']==selection['selected'] and r['seed']==42)
deployed_gain=100*(1-deployed['restoration_mse']/selection['identity_noise_mse'])
fig,ax=plt.subplots(figsize=(6,3.6),layout='constrained')
vals=[selection['identity_noise_mse']]+[selection['candidates'][k]['restoration_mse']for k in ['linear','mlp']]
ax.bar([0,1,2],vals,color=['#84998f','#d8ad60','#26694c'],width=.6)
for i,k in enumerate(['linear','mlp'],1):ax.scatter(i+np.array([-.09,0,.09]),[r['restoration_mse']for r in training if r['kind']==k],s=18,color='#223b31',zorder=4)
ax.set_xticks([0,1,2],['Identity','Linear (2,352)','MLP (9,360)']);ax.set_ylabel('MSE to published clean latent');ax.set_title('Validation only | 5 demonstration episodes');ax.spines[['top','right']].set_visible(False)
fig.savefig(PUB/'latent-restoration.svg');plt.close(fig)
video_sections=[]
for name in base_names+confirm_names:
 d=RUN/name;records=conditions[name]['records'];destination=PUB/name;destination.mkdir(exist_ok=True)
 video_blocks=[]
 for i,r in enumerate(records):
  matches=list((d/'runtime/eval_result').rglob(f'episode{i}.mp4'));assert len(matches)==1
  shutil.copyfile(matches[0],destination/f'episode{i}.mp4')
  video_blocks.append(f'<details><summary>Seed {r["seed"]} · {"成功" if r["success"] else "失败"} · {r["actions"]} 步</summary><p>{html.escape(r["instruction"])}</p><video controls playsinline preload="none" style="width:100%;max-width:640px" src="assets/robustness-20260921/{name}/episode{i}.mp4"></video></details>')
 # Display initial policy-input images, not only simulator clean video.
 first=records[0]['seed'];image=d/f'seed-{first}-policy.png';shutil.copyfile(image,destination/'policy-input.png')
 video_sections.append(f'<details><summary>{labels[name]} · {conditions[name]["successes"]}/5 · 全部五次</summary><p>首个 episode 的实际主相机输入：</p><img style="max-width:100%;width:320px" src="assets/robustness-20260921/{name}/policy-input.png" alt="{labels[name]}初始输入">'+''.join(video_blocks)+'</details>')
shutil.copyfile(RUN/'baseline-front/diagnostic-contact-sheet.jpg',PUB/'front-first-episode.jpg')
validation=table(['候选','参数量','噪声恢复 MSE','相对原误差','干净输入改动 MSE','是否过门槛'],[[k,2352 if k=='linear' else 9360,f'{r["restoration_mse"]:.5f}',f'{r["noise_ratio"]:.3f}',f'{r["clean_distortion_mse"]:.5f}','是'if r['eligible']else'否']for k,r in selection['candidates'].items()])
paired_table=table(['条件','新增成功','新增失败','相同结局','双侧配对精确 p 值'],[['干净',clean_cmp['candidate_wins'],clean_cmp['candidate_losses'],clean_cmp['same_outcome'],f'{clean_cmp["two_sided_exact_discordant_pair_p"]:.4f}'],['噪声',noise_cmp['candidate_wins'],noise_cmp['candidate_losses'],noise_cmp['same_outcome'],f'{noise_cmp["two_sided_exact_discordant_pair_p"]:.4f}']])
invalid=sum(len(v)for v in s['invalid_initial_attempt'].values())
refilter_invalid=sum(len(v)for v in s['invalid_refiltering_attempt'].values())
body=f'''<div class="hero"><div class="eyebrow">Frozen policy / paired observation robustness</div><h1>恢复表征，<br><em>是否改善动作？</em></h1><p class="lead">{headline}</p></div>
<section id="decision"><h2>{verdict}</h2><p>完成 35 次配对闭环：15 次发布策略扰动评测，加 20 次独立初始状态下的原策略／适配器对照。训练只用了 9,360 参数的残差模块；完整发布策略、DINO、S-VAE 和归一化保持冻结。</p><div class="callout caution">每个条件只有 5 次、只有双臂抓瓶一个任务。离线恢复误差、闭环成功率和跨任务泛化是不同证据；本轮没有接触标签或真实机器人实验，也没有证明 RAE 优于其他表征。</div></section>
<section id="baseline"><h2>发布策略：噪声与相机替换</h2>{table(['条件','成功次数','Wilson 95% 区间','平均动作数（包含失败）','实际场景种子'],rows(base_names))}<p>只在主相机加入 σ=0.10 的 RGB 高斯噪声（0–1 尺度），经过裁剪和 uint8 量化；腕部图像、状态和指令保持一致。前方相机替换是固定相机间迁移，不是小视角抖动。初始输入检查显示取景及物体尺度明显变化、部分瓶身被裁切，因此这个条件的失败不能单独归因于压缩表征；它同时改变了可见信息与相机几何。官方专家可行性过滤保留，实际通过的场景种子列在表中。</p></section>
<section id="confirmation"><h2>独立初始状态：适配器是否有效</h2>{table(['条件','成功次数','Wilson 95% 区间','平均动作数（包含失败）','实际场景种子'],rows(confirm_names))}<img src="assets/robustness-20260921/closed-loop.svg" style="width:100%;height:auto" alt="发布策略与独立适配器对照的成功率和小样本区间">{paired_table}<p>确认组额外遇到专家过滤接受名单不一致：场景 300002 在干净参考中被接受、在噪声组却被跳过，且过滤发生在噪声施加前。因此在任何适配器闭环结果出现前，固定首个参考运行的专家可行场景及原始指令，四组全部重跑。每条轨迹首个动作前核对参考图像和状态，汇总再核对所有配对，全部一致。这是明确记录的协议修正，不是原计划下独立重复过滤。<a href="assets/robustness-20260921/fixed-scene-amendment.md">修正原因</a> · <a href="assets/robustness-20260921/fixed-scenes.json">固定场景名单</a>。扩散噪声固定种子 42，每次生成使用相同设置。最后一个适配器噪声进程的计数器记录执行 {s['adapter_hook']['calls']} 次（该计数器每次服务器重启归零，并非全部十条轨迹的总和），该进程观测潜变量最大改动 {s['adapter_hook']['max_change']:.4f}；未来槽位逐次核对未变。</p><p>小样本不能确认总体性能下降，但这批固定场景中的新增失败，已经说明离线恢复门槛不能保证每条轨迹的控制行为保留。配对检验只使用结局不同的轨迹。5 次足以发现集成故障或较大现象，但不足以排除中小效应；所有结果保留，不基于测试成功率重选模型。</p></section>
<section id="case"><h2>一条新增失败：相同起点，抓取结果不同</h2><p>Seed 300003：最终固定名单对照的原策略 94 步成功，适配器运行到 400 步仍失败。下图上排是原策略，下排是适配器；前面几列为相同录像帧，最后一列为各自的末帧。可见适配器轨迹中红瓶发生倾倒。这里只是定性观察，不是接触标注，也尚未识别具体潜变量维度的因果作用。</p><img src="assets/robustness-20260921/clean-regression-seed300003.jpg" style="width:100%;height:auto" alt="同一初始场景下原策略成功和适配器失败的轨迹对比"><p>所有场景的视频仍完整保留。这是一个观察到的反例，不代表跨任务的总体性能估计。</p></section><section id="adapter"><h2>最小改动：恢复原潜空间坐标</h2><p>原发布编码器先把图像编码为归一化后的 48 通道潜变量，再添加零初始化残差：1×1 卷积 48→96、GELU、1×1 卷积 96→48。只改当前观测帧。这样目标仍在原策略使用的坐标系中，避免直接替换新压缩器造成接口失配。</p>{validation}<img src="assets/robustness-20260921/latent-restoration.svg" style="width:100%;max-width:680px;height:auto" alt="线性和两层适配器的验证集潜变量恢复误差"><p>35 条训练演示、5 条验证演示；每条 12 帧，训练两份独立噪声、验证一份，共 900 个噪声配对。10 条演示测试轨迹未打开。两个候选 × 三个种子，各 2,000 步；损失为恢复误差加等权干净恒等误差，未做超参数搜索。</p><p>先按验证集选择 MLP，再固定部署种子 42。门槛是恢复 MSE 至少降低 10%，干净改动 MSE 不超过原噪声误差的 10%。MLP 恢复误差降低 {gain:.1f}%（三种子均值，实际部署种子 42 为 {deployed_gain:.1f}%），干净改动占原噪声误差的 {100*selection['candidates']['mlp']['clean_ratio']:.1f}%。全编码器与独立提取器的数值一致性检查差为 0。</p><p>补充诊断（不参与选型）：原潜变量每个 token 的通道 RMS 约为 1，适配后干净输入约为 0.985，噪声输入约为 0.962。坐标相同不代表输出分布完全相同；这些变化是否影响动作必须由闭环结果判断。<a href="assets/robustness-20260921/adapter-latent-diagnostics.json">完整潜变量统计与布局区域误差</a>。</p></section>
<section id="repro"><h2>发现并修复语言种子缺口</h2><p>RoboTwin 在场景初始化时只固定 NumPy 与 Torch，指令生成却使用 Python 的 random.shuffle / choice。相同场景种子因此可能配到不同语言，混淆表征消融。独立 CPU 复现：五种 Python 随机状态产生五份原始指令列表；按场景种子局部固定后为同一份列表，外部随机状态保持不变。</p><p>初次语言未配对的 {invalid} 条已完成工程记录，以及专家重复过滤导致不匹配的 {refilter_invalid} 条已完成记录和中断日志，全部归档，未混入最终 35 次配对统计。发现问题后对所有条件统一修复并重跑；模型、噪声强度、种子范围和模型选择规则保持原样；确认场景改为参考名单固定重放。补丁仅供审阅，未发送 PR。</p><a class="tag" href="assets/robustness-20260921/robotwin-instruction-seed.patch">最小语言种子补丁</a> <a class="tag" href="assets/robustness-20260921/robotwin-instruction-seed-reproduction.json">上游真实代码复现</a><p>OpenWAM 侧也已准备默认关闭的可选开关 <code>ROBOTWIN_SCENE_SEEDED_LANGUAGE=1</code>，无需修改外部 RoboTwin。6 项单元测试、格式检查以及真实上游指令生成对照通过。<a href="assets/robustness-20260921/openwam-scene-seeded-language.patch">OpenWAM 补丁</a> · <a href="assets/robustness-20260921/openwam-language-integration.json">集成验证</a> · <a href="assets/robustness-20260921/OPENWAM_LANGUAGE_PR_DRAFT.md">PR 草稿</a>。</p><p>此前发现的结果分母错误仍保留原始输出：本页按实际成功数／5 汇报，汇总器交叉核验日志与逐轨迹记录，避免把 5 次误除以 100。</p></section>
<section id="videos"><h2>全部闭环视频与实际输入</h2><p>视频由仿真器记录干净主相机画面；每组另外展示实际送入策略的初始图像，因此噪声与前方替换在输入预览中可见。没有筛选成功案例，35 段全部保留。</p><p>第一条相机迁移失败的主相机录像采样（帧 0、200、399），只是定性展示，不是接触事件标注：</p><img src="assets/robustness-20260921/front-first-episode.jpg" style="width:100%;height:auto" alt="首条相机迁移失败轨迹的开始、中间和末尾画面">{''.join(video_sections)}</section>
<section id="artifacts"><h2>可复现记录与贡献边界</h2><p>当前可独立贡献：配对观测扰动评测、语言种子修复、以及保持原潜空间的可选残差适配器实验。是否建议模型默认启用，应由更大规模闭环证据决定。</p><div class="labels"><a class="tag green" href="assets/robustness-20260921/summary.json">全部指标与逐轨迹结果</a><a class="tag" href="assets/robustness-20260921/protocol.md">预先固定的规则与修正记录</a><a class="tag" href="assets/robustness-20260921/adapter-training.json">六次训练结果</a><a class="tag" href="assets/robustness-20260921/adapter-selection.json">验证集选型</a><a class="tag" href="assets/robustness-20260921/confirmation-unlock.json">闭环测试开启记录</a><a class="tag" href="assets/robustness-20260921/adapter-selected.pt">选定小适配器权重</a><a class="tag" href="assets/robustness-20260921/reproduction-scripts.tar.gz">复现脚本</a></div><p>发布策略版本 af1595c8，OpenWAM 7c5861e，RoboTwin 0aeea2d。同步 10 步去噪、DiT 缓存开启、编译关闭、400 动作上限、无规划器回退。曾尝试复用训练卡，但启动前发现已有其他作业，预检自动中止；确认评测恢复在原卡串行运行，全部配对使用同一张卡。<a href="assets/robustness-20260921/resource-amendment.md">运行安排调整记录</a>。与论文全量协议仍有范围与编译设置差异。</p></section>'''
if Path('/data02/zifanz4/openwam-experiments/results/norm-20260921/historical-repeat-note.json').exists():
 body='<div class="callout caution" id="historical-repeat-update">'+'后续重放更新：在新一轮单场景诊断中，原策略在 seed 300003 也于 400 步失败；初始主相机图像、状态与指令一致。此次执行顺序与上一轮不同，差异来源尚未定位，不能把原来的单次成败差异解释为稳定的适配器退化。新场景四组对照仍按冻结方案进行。'+' <a href="norm.html#historical">后续验证记录</a>。</div>'+body
page='''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>闭环扰动与表征恢复 · OpenWAM Research Notebook</title><link rel="stylesheet" href="style.css"></head><body><header><a class="brand" href="index.html">EMBODIED / RESEARCH NOTEBOOK</a><span class="status">闭环研究 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / ROBUSTNESS</p><nav aria-label="报告目录"><a href="#decision">结论</a><a href="#baseline">发布策略扰动</a><a href="#confirmation">独立闭环对照</a><a href="#adapter">表征恢复</a><a href="#repro">语言种子修复</a><a href="#videos">全部视频</a><a href="#artifacts">完整记录</a><a href="groups.html">上一轮分组实验</a></nav></aside><main>'''+body+'''<footer><a href="index.html">研究主页</a> · <a href="groups.html">上一轮结果</a></footer></main></div></body></html>'''
(SITE/'robustness.html').write_text(page)
shutil.copytree(PUB,SITE/'assets/robustness-20260921',dirs_exist_ok=True)
f=SITE/'index.html';text=f.read_text();marker='<section id="groups-result-note"';at=text.index(marker);new=f'<section id="robustness-result-note" class="callout"><strong>上一轮闭环结果 · 9 月 21 日：</strong>{headline} <a href="robustness.html">查看完整对照和 35 段视频 →</a></section>'
if '<section id="robustness-result-note"' in text:
 start=text.index('<section id="robustness-result-note"');end=text.index('</section>',start)+len('</section>');text=text[:start]+new+text[end:]
else:text=text[:at]+new+text[at:]
f.write_text(text)
for svg in (SITE/'assets/robustness-20260921').glob('*.svg'):
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print(json.dumps({'headline':headline,'verdict':verdict,'videos':35,'public_dir':str(PUB)},ensure_ascii=False,indent=2))
