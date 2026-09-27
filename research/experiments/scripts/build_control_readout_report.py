"""Publish all CPU readout outcomes and completed CLI/sample-alignment gates."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/zifanz4/openwam-experiments')
RUNTIME = Path('/data02/zifanz4/openwam-experiments/results')
R = RUNTIME / 'control-readout-20260922'
P = RUNTIME / 'representation-preflight-20260922'
SITE = Path('/home/zifanz4/research-reports/dist')
A = SITE / 'assets/control-readout-20260922'

def main():
    s = json.loads((R / 'summary.json').read_text())
    assert json.loads((R / 'audit.json').read_text())['passed']
    A.mkdir(exist_ok=False)
    for name in ['summary.json','audit.json','audit.log','source-manifest.json','selection.json','readout-coefficients.npz','predictions.npz','PROTOCOL.md','launch.json','run.log','exit-code.txt']:
        shutil.copy2(R / name, A / name)
    (A / 'preflight').mkdir()
    for name in ['PLAN.json','validation.json','fixture-manifest.json','sample-manifest.json','aligned-state-targets.npz','cli-runs.json','cli-batch1.json','cli-batch2.json','cli-batch1.log','cli-batch2.log','preflight.log','readout-gates.log','readout-gates-final.log']:
        shutil.copy2(P / name, A / 'preflight' / name)
    for name in ['control_restoration_readout.py','audit_control_readout.py','check_representation_preflight.py','robustness_common.py','build_control_readout_report.py']:
        shutil.copy2(ROOT / 'scripts' / name, A / name)
    shutil.copy2(ROOT / 'CONTROL_READOUT_FINDINGS.md', A / 'FINDINGS.md')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, group in zip(axes, ['translation','gripper']):
        for condition, color, offset in [('clean','#2b755b',-.10),('noisy','#bd5a36',.10)]:
            rows=[s['paired_vs_identity'][f'adapter{seed}_{condition}'][group] for seed in [42,43,44]]
            means=np.array([v['normalized_mse_delta'] for v in rows]);bounds=np.array([v['episode_bootstrap_percentile95'] for v in rows])
            ax.errorbar(np.arange(3)+offset,means,yerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',capsize=4,color=color,label=condition)
        ax.axhline(0,color='#777',linewidth=1);ax.set_xticks(range(3),['42 (fixed primary)','43','44']);ax.set_title(group.capitalize());ax.set_xlabel('Existing adapter training seed');ax.set_ylabel('Change in normalized readout MSE');ax.grid(axis='y',alpha=.15)
    axes[0].legend();fig.suptitle('Restoration improves noisy translation readout, not every control group')
    fig.text(.5,.015,'Development reuse: 5 episodes, 60 frames. Paired episode bootstrap; lower is better. Not policy success.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(A/'group-deltas.png',dpi=180);fig.savefig(A/'group-deltas.svg');plt.close(fig)
    table=[]
    with (A/'all-conditions.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['condition','all_nmse','translation_nmse','translation_rmse_mm','gripper_nmse','gripper_rmse'])
        for name,r in s['results'].items():
            g=r['groups'];values=[name,g['all']['normalized_mse'],g['translation']['normalized_mse'],g['translation']['rmse'],g['gripper']['normalized_mse'],g['gripper']['rmse']];writer.writerow(values)
            table.append('<tr>'+''.join('<td>'+html.escape(x if isinstance(x,str) else f'{x:.5f}')+'</td>' for x in values)+'</tr>')
    pairrows=[]
    for name,r in s['paired_vs_identity'].items():
        pairrows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [name,*[f"{r[k]['relative_change_percent']:+.2f}%" for k in ['all','translation','gripper']]])+'</tr>')
    prefix='assets/control-readout-20260922/'
    links=''.join(f'<li><a href="{prefix}{p.relative_to(A)}">{p.relative_to(A)}</a></li>' for p in sorted(A.rglob('*')) if p.is_file() and p.suffix not in ['.png','.svg'])
    now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>潜变量恢复能改善控制读出吗？</title><link rel="stylesheet" href="style.css"></head><body><header><a class="brand" href="index.html">EMBODIED / RESEARCH NOTEBOOK</a><span class="status">{now}</span></header><div class="layout"><aside><p>FROZEN CONTROL READOUT</p><nav><a href="#finding">本轮发现</a><a href="#design">固定方案</a><a href="#results">全部结果</a><a href="#limits">解释与取舍</a><a href="#software">入口与数据检查</a><a href="#artifacts">全部材料</a><a href="review.html">贡献审阅包</a></nav><div class="aside-note">CPU诊断 · 已完成<br>35训练 / 5开发验证轨迹<br>无新增策略评测</div></aside><main>
<div class="hero"><div class="eyebrow">Latent restoration ≠ uniform control-readout improvement</div><h1>位移读出变好，<br><em>夹爪与干净输入未同步改善。</em></h1><p class="lead">固定部署种子42：噪声下位移标准化误差下降11.84%，夹爪上升5.82%；干净输入总误差上升2.64%。这是已有验证集上的开发诊断，不是独立确认或闭环提升。</p></div>
<section id="finding"><h2>为什么不能只看恢复MSE？</h2><p>同一个已训练的适配器让潜变量恢复MSE下降约34.72%，固定读出器的噪声总体误差也下降11.04%，但总体均值掩盖了平移和夹爪的不同变化。三个已有适配器种子的变化方向一致，全部保留；未挑选新部署种子。</p><figure><img src="{prefix}group-deltas.png" alt="三个适配器种子的干净和带噪分组读出误差变化" style="max-width:100%"><figcaption>相对同条件identity的标准化MSE差值；负数较好。区间来自5条轨迹的配对重采样，非60个独立样本。<a href="{prefix}group-deltas.svg">SVG</a></figcaption></figure><p>主要种子42的噪声夹爪区间跨过零；不能声称已经证明普遍退化。<strong>开发结果没有满足“平移、夹爪同时改善”的条件，暂不追加适配器损失搜索。</strong></p></section>
<section id="design"><h2>在看结果前固定读出方案</h2><p>使用此前缓存的当前帧48通道表征，2×2池化为192维，加16维当前本体状态。预测4个原始帧后的实际末端位移（6维）与夹爪状态（2维）。35条训练轨迹共420个唯一干净观测；两份训练噪声副本不重复计为训练样本。</p><p>视觉+本体与仅本体各拟合一个线性ridge读出器。正则系数只由训练轨迹的5折选择，候选固定为0.0001、0.01、1、100；每折单独拟合训练归一化。最终选择分别为1与0.01，并在验证评分前落盘。随后冻结读出器，评价干净、带噪、三个固定适配器种子的干净／带噪，以及循环错配视觉对照，共10条件。</p><p>冻结提交<code>6687e266a985125b4d7e2ecf8f36222c67e60933</code>，01:07:24CDT冻结，01:07:50启动，01:07:52完成，退出0。2CPU线程、CUDA不可见、300秒上限；没有训练适配器、编码器或策略。5项CPU门槛通过，11份冻结来源／输入未改变；保存预测的所有指标与区间独立重算通过。</p></section>
<section id="results"><h2>全部10个条件</h2><div class="table-wrap"><table><thead><tr><th>条件</th><th>总体NMSE</th><th>位移NMSE</th><th>位移RMSE mm</th><th>夹爪NMSE</th><th>夹爪RMSE</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>总体是8个按训练尺度标准化坐标的均值，因此位移占6/8、夹爪占2/8；并非任务成功率。夹爪RMSE使用数据中的原生单位。视觉错配将同一采样序号的图像表征换成下一条验证轨迹，保留本体和标签，是依赖诊断而非真实扰动。</p><h3>每个适配器相对同条件identity</h3><div class="table-wrap"><table><thead><tr><th>条件</th><th>总体MSE变化</th><th>位移MSE变化</th><th>夹爪MSE变化</th></tr></thead><tbody>{''.join(pairrows)}</tbody></table></div><p>所有种子共用同5条验证轨迹和读出器，不能当作15条独立轨迹；种子42仍是原先固定的主要条件。</p></section>
<section id="limits"><h2>这能说明什么，不能说明什么？</h2><p>本体基线的夹爪NMSE为0.03729，低于干净视觉+本体的0.07744；说明读出器的正则、容量和多余视觉变化也可能造成取舍。我们观察到的是固定读出器对表征变化的响应，不能据此断言S-VAE不可逆地丢失了夹爪信息。</p><p>干净视觉使位移NMSE从仅本体0.61977降至0.46757，错配视觉为0.58381；这支持该设置下视觉对读出有用，仍不能隔离预训练的因果作用。</p><p>验证的5条轨迹已经用于选适配器架构；发布表征也可能见过这些示范。没有新的独立测试轨迹，没有旋转、接触或控制器命令标签。CPU适配器使用FP32运算与BF16输入输出契约，未验证GPU数值一致性。区间仅描述这些开发轨迹，不能外推多任务性能。</p><p><strong>贡献取舍：</strong>保持发布推理默认设置，把这个反例及按控制分组、带本体基线的读出方法写入评测材料。237行S-VAE补丁仍只测重建；本页读出脚本是独立研究材料，不冒称已经合入上游。若要声称改进，另需新轨迹和闭环方案；本轮不自动继续调权重。</p></section>
<section id="software"><h2>本轮补齐的软件与样本检查</h2><p>文档中的S-VAE CLI已在现有策略环境正常运行：既有真实训练检查点，固定3个特征clip打包为原生分片格式，CPU batch2和batch1都退出0，指标在1e-6容差内一致，检查点归一化与输入哈希保留。没有叶模块绕过，也未重装环境；torch2.7.0、transformers5.17.0、diffusers0.40.0。完整上游测试套件仍未运行；此小样本仅验证软件。</p><p>480个干净、900个带噪缓存的引用／副本、35训练和5验证轨迹划分、原始帧索引及frame+4状态标签全部核对。未打开示范测试轨迹。原采集没有逐图像哈希，因此不能追溯证明每个潜变量的字节级图像来源；当前图像和标签哈希随清单保留。原始帧号不等于物理时间戳。</p></section>
<section id="artifacts"><h2>方案、日志、模型系数与全部预测</h2><ul>{links}</ul></section></main></div></body></html>'''
    (SITE/'readout.html').write_text(page)
    p=SITE/'index.html';text=p.read_text();assert 'id="readout-latest"'not in text
    text=text.replace('<main id="top">','<main id="top"><div class="callout" id="readout-latest"><strong>新诊断：噪声位移读出改善，夹爪与干净输入未同步改善。</strong>固定适配器的分组误差与全部种子结果已公布；文档CLI验证也完成。<a href="readout.html">新实验、完整指标和局限</a>。</div>',1);p.write_text(text)
    p=SITE/'review.html';text=p.read_text().replace('<main>','<main><section class="callout"><strong>后续进展：</strong>正常S-VAE CLI门槛已补齐；新增冻结控制读出实验显示分组取舍。<a href="readout.html">全部结果与解释边界</a>。本页其余内容保留审阅时状态。</section>',1);p.write_text(text)
    m={str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};(A/'artifact-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'page':'readout.html','artifacts':len(m),'conditions':len(s['results'])}))

if __name__ == '__main__':
    main()
