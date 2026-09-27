#!/usr/bin/env python3
"""Publishable offline experiment report from completed artifacts."""
import json
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = Path('/home/zifanz4/research-reports/dist')
SRC = ROOT / 'public-results/control-20260921'
ASSETS = SITE / 'assets/control-20260921'
ASSETS.mkdir(parents=True, exist_ok=True)
s = json.loads((SRC / 'summary.json').read_text())
assert s['resources']['training_runs'] == 27
for name in ['summary.json', 'probe-results.svg', 'probe-results.png', 'group-probe-diagnostic.json']:
    shutil.copy2(SRC / name, ASSETS / name)
shutil.copy2(ROOT / 'CONTROL_PROTOCOL.md', ASSETS / 'protocol.md')
with tarfile.open(ASSETS / 'reproduction-scripts.tar.gz', 'w:gz') as tar:
    for name in ['CONTROL_PROTOCOL.md', 'requirements-lock.txt', 'REPRODUCE_CONTROL.md', 'scripts/control_study.py', 'scripts/control_robustness.py', 'scripts/control_group_probe.py', 'scripts/summarize_control.py', 'scripts/night.py', 'scripts/pilot.py']:
        tar.add(ROOT / name, arcname='control-study/' + name)
NAMES = {'adjust_bottle': '瓶子调整', 'handover_block': '双臂交接', 'place_object_basket': '物体入篮'}
selected = s['selection']['selected_weight']
group_diagnostic = json.loads((SRC / 'group-probe-diagnostic.json').read_text())
rows = []; paired = []; rawrows = []; component_rows = []; viewrows = []
for task, block in s['tasks'].items():
    raw = block['raw']['probes']
    baseline = block['methods']['aux0']['probes']
    rawrows.append(f"<tr><td>{NAMES[task]}</td><td>{raw['clean']['normalized_mse']:.4f}</td><td>{baseline['clean']['mean']:.4f}</td><td>{raw['noise10']['normalized_mse']:.4f}</td><td>{baseline['noise10']['mean']:.4f}</td></tr>")
    for weight in [0, .1, 1]:
        key = f'aux{weight:g}'; m = block['methods'][key]
        fmt = lambda v: f"{v['mean']:.4f} ± {v['std_across_training_seeds']:.4f}"
        rows.append(f"<tr><td>{NAMES[task]}</td><td>{weight:g}</td><td>{fmt(m['probes']['clean'])}</td><td>{fmt(m['probes']['noise10'])}</td><td>{fmt(m['probes']['brightness06'])}</td><td>{fmt(m['reconstruction']['all_mse'])}</td></tr>")
        component_rows.append(f"<tr><td>{NAMES[task]}</td><td>{weight:g}</td><td>{fmt(m['translation_rmse_mm']['clean'])}</td><td>{fmt(m['gripper_rmse']['clean'])}</td></tr>")
        viewrows.append(f"<tr><td>{NAMES[task]}</td><td>{weight:g}</td><td>{fmt(m['probes']['head_reencoded'])}</td><td>{fmt(m['probes']['front_replacement'])}</td><td>{m['front_vs_head_error_ratio']['mean']:.2f}×</td></tr>")
        if weight:
            for variant, label in [('clean', '干净'), ('noise10', '噪声'), ('brightness06', '亮度'), ('front_replacement', '视角')]:
                a = block['paired_vs_zero'][key][variant]; lo, hi = a['episode_paired_bootstrap_95ci']
                paired.append(f"<tr><td>{NAMES[task]}</td><td>{weight:g} / {label}</td><td>{a['relative_change_percent']:+.1f}%</td><td>{a['candidate_minus_baseline']:+.4f} [{lo:+.4f}, {hi:+.4f}]</td></tr>")
ratios = s['selection']['validation_ratio_scores']
scoretext = '；'.join(f"λ={float(k):g}：{v:.4f}" for k,v in ratios.items())
if selected == 0:
    verdict = '验证集仍选择原损失。'
    explanation = '预先规定的验证选择规则没有选中新辅助监督。保留所有方法与任务结果，暂不建议修改默认压缩器。'
else:
    verdict = f'λ={selected:g} 改善位移读出，夹爪出现退化。'
    explanation = '验证集选择的辅助权重为 1。相对同设置下的原损失，三个任务的干净状态变化预测误差低约 32%–37%，噪声下低约 7%–40%，换相机后低约 19%–48%。入篮任务噪声差异的配对区间跨过零，尚不能确认该项改善。以上均为离线读出误差。'
group_rows = []
for task, block in group_diagnostic['tasks'].items():
    a = block['paired_mean_rmse_changes']
    group_rows.append(f"<tr><td>{NAMES[task]}</td><td>{a['translation']['relative_change_percent']:+.1f}%</td><td>{a['gripper']['relative_change_percent']:+.1f}%</td></tr>")
notes = []
for weight in [.1, 1.]:
    macro = s['macro_relative_changes'][f'aux{weight:g}']
    notes.append(f"λ={weight:g}：三个任务平均相对变化，干净 {macro['clean']:+.1f}%，噪声 {macro['noise10']:+.1f}%，亮度 {macro['brightness06']:+.1f}%，换相机 {macro['front_replacement']:+.1f}%。")
maxdiff = max(s['checks']['old_vs_rerun_baseline_max_abs'].values())
max_control_diff = max(control['all_test_current_vector_max_abs_vs_clean'] for block in s['tasks'].values() for method in block['methods'].values() for control in method['numeric_controls'])
max_t1_diff = max(control['first_batch_T1_vs_T3_vector_max_abs'] for block in s['tasks'].values() for method in block['methods'].values() for control in method['numeric_controls'])
max_view_encoder_diff = max(block['view_extraction']['head_reencoded_vs_cached_max_abs'] for block in s['tasks'].values())
resources = s['resources']
page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>控制相关监督压缩实验 · OpenWAM Research Notebook</title><meta name="description" content="三个任务、三个种子、27次固定预算训练，检验辅助状态变化监督能否改善压缩表征。"><link rel="stylesheet" href="style.css"></head><body><header><a class="brand" href="index.html">EMBODIED / RESEARCH NOTEBOOK</a><span class="status">实验完成 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / CONTROL INFORMATION</p><nav aria-label="实验结果目录"><a href="#decision">实验结论</a><a href="#raw">原始特征对照</a><a href="#results">全部训练条件</a><a href="#gripper">夹爪退化诊断</a><a href="#views">真实视角迁移</a><a href="#controls">如何保证比较有效</a><a href="#artifacts">复现与下一步</a><a href="rae.html">研究假设</a><a href="night.html">昨晚的加权实验</a></nav><div class="aside-note">离线状态变化读出<br>未进行策略闭环评测</div></aside><main>
<div class="hero"><div class="eyebrow">Control-supervised compression / completed ablation</div><h1>让压缩器学习<br><em>保留任务信息。</em></h1><p class="lead">冻结 DINO，在相同的 48 维压缩器上增加当前观测到未来状态变化的辅助监督。27 次固定预算训练全部保留，使用独立读出器检验效果。</p></div>
<div class="facts"><div><b>27 次</b><small>3 任务 × 3 种子 × 3 个权重</small></div><div><b>2,000 步</b><small>固定最终检查点</small></div><div><b>48 维</b><small>所有压缩器结构相同</small></div></div>
<section id="decision"><h2>{verdict}</h2><p>{explanation}</p><p>选择依据是三个任务、三个种子的验证误差相对各自原损失对照的平均比值：{scoretext}。比值越低越好；选择记录在本轮测试指标计算前保存。</p><p>{' '.join(notes)} 正数表示误差更高。这里平均的是每个任务相对自身基线的变化，不是任务成功率。</p><div class="callout"><strong>不能忽略的取舍：</strong>干净末端位移 RMSE 降低约 16%–22%，但夹爪 RMSE 增加约 1%、26%、9%，特征重建误差也增加约 10%–12%。辅助监督使指定运动目标更容易读出，但不足以推荐直接替换原表征。</div><div class="callout caution">这轮仍是探索性离线诊断。旧 held-out 轨迹已参与前期选题，不能当作完全未见过的确认集；需要新轨迹和闭环执行才能确认可推广收益。</div></section>
<section id="raw"><h2>先比较压缩前后可读出的信息</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>原始 DINO / 干净</th><th>原 S-VAE / 干净</th><th>原始 DINO / 噪声</th><th>原 S-VAE / 噪声</th></tr></thead><tbody>{''.join(rawrows)}</tbody></table></div><p>同样仅取当前帧，以 2×2 网格汇总；特征与目标标准化只使用训练轨迹，正则化只由验证集选择。原始 DINO 与压缩器的读出输入维数不同，因此这是诊断对照，不能直接解释为“压缩造成了全部差异”。</p></section>
<section id="results"><h2>独立读出器的完整对照</h2><p>目标是四个原始时间步后的末端位移与夹爪状态，不是控制命令。训练辅助头只读取当前潜变量和当前本体状态；评价时丢开这个头，冻结编码器并重新拟合线性 ridge probe。</p><img src="assets/control-20260921/probe-results.svg" alt="三个任务在干净和噪声输入下的原损失与两种辅助监督对照，含三训练种子误差条" style="display:block;width:100%;height:auto;background:white"><p class="small">误差条为三个训练种子的标准差。虚线为原始 DINO；其 probe 输入更大，不是等容量策略比较。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>辅助权重 λ</th><th>干净 MSE</th><th>噪声 MSE</th><th>亮度 MSE</th><th>特征重建 MSE</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><details><summary>末端位移与夹爪分开检查</summary><p>下面是干净输入的两个预测分量；夹爪按数据集状态单位计算，不能解释为接触力。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>λ</th><th>末端位移 RMSE / mm</th><th>夹爪状态 RMSE</th></tr></thead><tbody>{''.join(component_rows)}</tbody></table></div></details><details><summary>相对原损失的配对差异与区间</summary><p>先对三个固定训练种子平均，再按同一组测试轨迹做 10,000 次配对 bootstrap。区间条件于这些种子，不包含全部训练随机性。正数表示更差。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>λ / 输入</th><th>相对变化</th><th>MSE 差值 [95% 区间]</th></tr></thead><tbody>{''.join(paired)}</tbody></table></div></details><p>另保留只用本体状态、打乱视觉的诊断，详见完整 JSON。打乱图像会制造不一致观测，只能用于检查视觉依赖，不能当作真实传感器扰动。</p></section>
<section id="gripper"><h2>夹爪退化是否只是读出器选择造成？</h2><p>发现夹爪退化后，补做了一个明确标记为事后分析的 CPU 诊断：分别为位移和夹爪拟合读出器，各自只在验证集选 ridge 正则化。没有重新训练表征，也没有改变主实验的 λ 选择或替换主结果。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>位移 RMSE 变化</th><th>夹爪 RMSE 变化</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div><p>分开读出后，夹爪误差仍高约 10%–14%。因此，共用一个正则化参数不足以解释这一退化。它仍不等于证明夹爪信息已不可逆丢失；需要进一步区分目标取舍、读出能力和可观测性。</p><p>下一项小改动应测试位移与夹爪监督的分组权重，并保留本轮方案作为基线；同时使用新轨迹验证，避免继续在已观察的测试集上调参。实际接触阶段收益仍需要有效标签与闭环执行。</p><a href="assets/control-20260921/group-probe-diagnostic.json">下载独立分组读出诊断</a></section><section id="views"><h2>真实相机对：同一状态，换一个视角</h2><p>把同步的 front_camera 图像放入原 head_camera 位置，保持两路腕部图像、本体状态和标签不变。读出器仍只在原 head 视角训练，不针对 front 重拟合。这个附加诊断在训练期间、查看本轮测试指标之前登记；不改变任何训练或权重选择。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>λ</th><th>匹配重编码的 head MSE</th><th>front 替换 MSE</th><th>front / head 误差比</th></tr></thead><tbody>{''.join(viewrows)}</tbody></table></div><p>这是 head 视角训练的固定读出器在另一相机上的迁移，不是连续的小幅视角扰动。误差不能单独归因为编码器丢失信息；原读出器对新投影的适应性也会影响结果。候选换相机后的误差仍约为同一模型 head 视角的 4.9–8.5 倍，尚未解决视角敏感性。头部相机重编码与原始缓存的最大特征差为 {max_view_encoder_diff:.3g}；配对 head 对照用于区分相机变化和数值变化。</p></section><section id="controls"><h2>数值与数据控制</h2><ul><li>每个任务单独训练压缩器，尚未验证共享多任务模型。按轨迹固定 35/5/10 训练、验证、测试划分；每任务 50 条示范、600 个短片。归一化与辅助标签统计只用训练轨迹。</li><li>全部条件启用相同的确定性设置，重新训练九个原损失对照。与旧设置下检查点的最大参数差为 {maxdiff:.6g}，本轮效果以新对照为准。</li><li>启动前检测到原损失重复反传也有微小差异；隔离后启用确定性 CUDA 设置。2×2 池化使用相同区域的 reshape + mean，以支持确定性反传，并检查与原池化前向在 1e-6 容差内一致。</li><li>零权重与原损失梯度在本轮设置下完全一致；辅助梯度只读取当前 latent，能到达编码器；改变未来输入不影响当前特征；测试标签变化不影响训练统计。</li><li>所有方法使用相同的测试扰动缓存。它们从未用于拟合压缩器、辅助头或读出器。最终扰动结果按干净路径匹配压缩器 B=2、T=3 的批量形状；全部测试当前向量控制的最大差为 {max_control_diff:.3g}，原 T=1 编码与 T=3 的首批最大差为 {max_t1_diff:.3g}。这项检查没有发现本轮扰动结果受压缩器批量形状影响。保留最初评测供审计，最终仅采用匹配形状的结果。</li><li>没有更改训练步数来追逐正结果，没有筛掉种子。额外加入固定相机对迁移，但没有连续视角扫描或经过核验的接触标签。</li></ul><div class="facts"><div><b>1 × L40S</b><small>顺序训练</small></div><div><b>{resources['training_seconds']/60:.1f} 分钟</b><small>27 次训练阶段累计</small></div><div><b>{resources['peak_allocated_gb']:.2f} GB</b><small>训练时 PyTorch 峰值分配</small></div></div><p class="small">主实验总运行时间 {resources['wall_seconds']/60:.1f} 分钟，包含检查、读取缓存、训练和初次评测；追加视角与数值诊断 {s['robustness_completion']['wall_seconds']/60:.1f} 分钟；不含此前数据下载与特征抽取。这不是完整 WAM 策略的资源测量。</p></section>
<section id="artifacts"><h2>复现与下一步</h2><div class="labels"><a class="tag green" href="assets/control-20260921/summary.json" download>全部结果与选择记录</a><a class="tag" href="assets/control-20260921/protocol.md">固定实验方案与修订</a><a class="tag" href="assets/control-20260921/reproduction-scripts.tar.gz" download>实验脚本</a><a class="tag" href="night.html">特征缓存复现入口</a></div><p>复用上一轮已抽取的特征与匹配批量的扰动缓存。脚本包包含本轮驱动、汇总脚本及其本地模块依赖；环境与数据准备见上一轮复现说明。模型检查点保留在实验存储中，不嵌入网页。</p><p>压缩器改变后，原策略的潜空间兼容性仍需验证。只有在新轨迹上保留收益，并在等预算策略适配后提高闭环成功率，才建议把辅助监督作为可选模型改动贡献。独立评测工具仍保持单独可审阅。</p></section>
<footer>探索性研究 · 2026.09.21 · <a href="index.html">研究主页</a> · <a href="rae.html">表征改进方案</a></footer></main></div></body></html>'''
(SITE / 'control.html').write_text(page)
for name in ['index.html', 'rae.html', 'night.html', 'gaps.html', 'experiments.html']:
    path = SITE / name; text = path.read_text()
    text = re.sub(r'<div class="callout" id="control-running-note">.*?</div>', '', text)
    if 'href="control.html"' not in text:
        text = text.replace('</nav>', '<a href="control.html">控制监督实验</a></nav>', 1)
    if name == 'index.html' and 'id="control-result-note"' not in text:
        text = re.sub(r'<main\b[^>]*>', lambda match: match.group(0) + f'<div class="callout" id="control-result-note"><strong>新实验已完成：</strong>27 次控制相关监督压缩训练。{verdict}<a href="control.html">查看对照、区间和复现材料 →</a></div>', text, count=1)
    path.write_text(text)
print('Rendered completed control study and artifacts')
