"""Consolidate existing scientific evidence and a new CPU patch review."""
from pathlib import Path
import hashlib
import html
import json
import shutil
from datetime import datetime

ROOT = Path('/home/zifanz4/openwam-experiments')
OUT = Path('/data02/zifanz4/openwam-experiments/results/contribution-review-20260922')
SITE = Path('/home/zifanz4/research-reports/dist')
DEST = SITE / 'assets/contribution-review-20260922'

def main():
    validation = json.loads((OUT / 'validation.json').read_text())
    assert validation['passed']
    night = json.loads((SITE / 'assets/night-20260921/summary.json').read_text())
    norm = json.loads((SITE / 'assets/norm-20260921/summary.json').read_text())
    evidence = {'scope': 'Arithmetic summary of existing results; no new samples or training', 'tasks': {}, 'closedloop': {}}
    rows = []
    for name, task in night['tasks'].items():
        pca, svae = [task['methods'][key] for key in ['pca48', 'svae_w1']]
        rec = 100 * (svae['reconstruction']['all_mse']['mean'] / pca['reconstruction']['all_mse']['mean'] - 1)
        probe = 100 * (svae['probes']['clean']['mean'] / pca['probes']['clean']['mean'] - 1)
        evidence['tasks'][name] = {'reconstruction_mse_change_pct': rec, 'clean_achieved_state_probe_mse_change_pct': probe}
        rows.append(f'<tr><td>{name}</td><td>{rec:+.2f}%</td><td>{probe:+.2f}%</td></tr>')
    closed = []
    for arm in ['identity', 'renorm', 'adapter', 'adapter_renorm']:
        evidence['closedloop'][arm] = {c: {k: norm['primary'][arm + '-' + c][k] for k in ['successes', 'episodes']} for c in ['clean', 'noise']}
        vals = [str(evidence['closedloop'][arm][c]['successes']) + '/10' for c in ['clean', 'noise']]
        closed.append(f'<tr><td>{arm}</td><td>{vals[0]}</td><td>{vals[1]}</td></tr>')
    assert sum(v['episodes'] for arm in evidence['closedloop'].values() for v in arm.values()) == 80
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    DEST.mkdir(exist_ok=False)
    names = ['PLAN.json', 'validation.json', 'tests.xml', 'cpu-tests.log', 'ruff.log', 'environment.log', 'audit.log', 'tool-install.log', 'evidence.json', 'svae.patch', 'robustness.patch', 'trace.patch']
    for name in names:
        shutil.copy2(OUT / name, DEST / name)
    for name in ['REVIEW_PACKET.md', 'SVAE_PR_DRAFT.md']:
        shutil.copy2(ROOT / 'contribution' / name, DEST / name)
    for name in ['audit_contribution_packet.py', 'build_contribution_review.py']:
        shutil.copy2(ROOT / 'scripts' / name, DEST / name)
    prefix = 'assets/contribution-review-20260922/'
    links = ''.join(f'<li><a href="{prefix}{name}">{html.escape(name)}</a></li>' for name in names + ['REVIEW_PACKET.md', 'SVAE_PR_DRAFT.md', 'audit_contribution_packet.py', 'build_contribution_review.py'])
    patches = ''.join(f'<tr><td>{i}</td><td>{p["name"]}</td><td><code>{p["commit"][:7]}</code></td><td>{html.escape(p["diffstat"])}</td><td><a href="{prefix}{p["name"]}.patch">独立补丁</a></td></tr>' for i, p in enumerate(validation['patches'], 1))
    now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>表征贡献：证据与最小补丁</title><link rel="stylesheet" href="style.css"></head><body>
<header><a class="brand" href="index.html">EMBODIED / RESEARCH NOTEBOOK</a><span class="status">{now}</span></header><div class="layout"><aside><p>CONTRIBUTION REVIEW</p><nav><a href="#decision">贡献取舍</a><a href="#evidence">证据链</a><a href="#patches">三个独立补丁</a><a href="#next">下一步与边界</a><a href="#artifacts">复查材料</a></nav><div class="aside-note">本轮：CPU代码审阅<br>新增科学实验：0<br>未提交上游PR</div></aside><main>
<div class="hero"><div class="eyebrow">Representation evidence → reviewable contribution</div><h1>先把“重建更好”和<br><em>“控制更好”分开测清。</em></h1><p class="lead">首选贡献是237行的独立S-VAE诊断补丁。已有实验支持补测量工具，尚不支持默认启用新的表征模块。</p></div>
<section id="decision"><h2>明确选择：评测工具优先</h2><p>离线评测器复用训练时的均值与方差，以确定性后验均值测量当前条件帧和池化未来目标的重建误差；按元素加权，并报告相对目标能量的误差。模型、训练、检查点格式和默认推理保持原样。</p><p>语言配对和请求／动作记录器各自有用，但可以独立审阅，不要求一次接受整个工具包。当前没有匹配的RAE-vs-VAE策略消融；DINO/S-VAE本来已使用预训练特征，不能把这些实验写成RAE胜出。</p></section>
<section id="evidence"><h2>现有结果能支持到哪一步？</h2><h3>重建误差大幅下降，读出收益小且不一致</h3><div class="table-wrap"><table><thead><tr><th>任务</th><th>S-VAE相对PCA：重建MSE变化</th><th>干净状态变化probe MSE变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>负数表示误差降低。三个训练种子的均值，固定2000步；每任务35/5/10条轨迹划分，10条测试轨迹。probe预测四个原始步后的实际位移／夹爪状态，非控制器命令。百分比为旧结果的算术汇总，不新增样本，不表示统计显著性。<a href="night.html">全部方法、种子差异与区间</a>。</p>
<p>后续控制监督复用了这些轨迹，应归为开发结果；平移与夹爪要分别解释，夹爪变化不能代替接触标注。<a href="control.html">控制监督全部结果</a> · <a href="groups.html">分组误差与取舍</a>。</p>
<h3>恢复潜变量更准确，也未建立闭环净收益</h3><p>去噪适配器验证集的潜变量恢复MSE下降约35%。后续冻结80次评测的全部条件如下，噪声强度与之前35次试跑不同，不能合并成功率。</p><div class="table-wrap"><table><thead><tr><th>条件</th><th>干净成功数</th><th>噪声成功数</th></tr></thead><tbody>{''.join(closed)}</tbody></table></div><p>一个任务、每条件10场景；例如renorm噪声相对identity的配对检验p=0.5。样本量和轨迹重复性限制解释。保留全部条件，不凭9/10挑选新的默认模块。<a href="norm.html">80次完整结果、配对检验与视频</a> · <a href="robustness.html">最初的恢复误差与35次试跑</a>。</p>
<p>重复性诊断不增加表征实验样本量。最新上下文探针仅1/4段完成，端口预检失败后已停止，没有A/B结论；不会因自检而补跑。<a href="context-pilot.html">完整失败记录</a>。</p></section>
<section id="patches"><h2>三个独立、固定来源的补丁</h2><div class="table-wrap"><table><thead><tr><th>优先级</th><th>补丁</th><th>提交</th><th>改动量</th><th>下载</th></tr></thead><tbody>{patches}</tbody></table></div><p>固定上游基线：<code>{validation['base']}</code>。三者在新隔离源码副本共同应用成功；17项针对性CPU测试通过（5项表征指标、6项语言采样、6项记录器），Ruff通过。CUDA不可见、两CPU线程，未改写原补丁或科学结果。</p><p><strong>验证边界：</strong>尚未运行完整上游测试套件；此前真实检查点／原生分片smoke采用叶模块加载，正常CLI在完整依赖栈中的启动仍待验证。源码可共同应用不等于真实模型／模拟器组合已经验证。最初工具准备因venv无pip失败，随后使用已有隔离工具；失败日志一并保留。</p>
<p>记录器只测RPC请求／返回动作边界，不能解释为机器人实际运动；额外I/O也可能改变时序。场景语言采样选项仍需逐条件检查实际接纳种子和最终指令。<a href="contribution.html">记录器原始验证</a>。</p></section>
<section id="next"><h2>下一步只解决具体缺口</h2><p><strong>软件：</strong>在完整上游依赖环境中，用固定的小检查点／分片核对文档CLI能否直接运行。此项需要CPU，不需要再跑策略或训练。训练／测试轨迹是否分离由采集清单证明，不能由分片文件名推断。</p><p><strong>研究：</strong>先核对适配器缓存的样本ID、划分与原始HDF5时间戳，确认干净／扰动／恢复表征能在无泄漏条件下匹配当前状态与未来实际状态。再预定训练集拟合的冻结读出器，测试恢复是否改善控制相关方向。现有验证集属于开发数据；读出改善仍不是策略收益。当前未启动新读出训练、适配器权重搜索或闭环试验。</p><p>这份材料是可审阅的本地贡献包。未向维护者发消息、未提交PR，也不声称已被社区接纳。</p></section>
<section id="artifacts"><h2>复查材料</h2><ul>{links}</ul></section></main></div></body></html>'''
    (SITE / 'review.html').write_text(page)
    p = SITE / 'index.html'
    s = p.read_text()
    assert 'id="contribution-review-latest"' not in s
    s = s.replace('<main id="top">', '<main id="top"><div class="callout" id="contribution-review-latest"><strong>当前决定：优先贡献表征诊断工具。</strong>三个独立补丁的17项CPU兼容性检查通过，重建、读出与闭环证据已统一核对。<a href="review.html">贡献取舍、完整边界与审阅包</a>。</div>', 1)
    s = s.replace('最新：诊断贡献补丁', '轨迹诊断补丁')
    s = s.replace('<strong>下一步：固定动作，检验在线推理活动。</strong>', '<strong>历史准备记录：固定动作上下文对照。</strong>')
    s = s.replace('GPU尚未启动。', '后续运行已停止，见上方结果。')
    s = s.replace('研究记录 · 更新于 2026.09.21', '研究记录 · 更新于 2026.09.22')
    p.write_text(s)
    p = SITE / 'rae.html'
    s = p.read_text().replace('<main>', '<main><section class="callout"><strong>当前结论与交付：</strong>后续80次对照尚未建立闭环净收益；首选贡献是表征诊断工具。<a href="review.html">最新证据与可审阅补丁</a>。下文保留早期假设及当时状态。</section>', 1)
    p.write_text(s)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file()}
    (DEST / 'artifact-manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'page': 'review.html', 'artifact_count': len(hashes), 'new_scientific_trials': 0, 'targeted_cpu_tests': 17}))

if __name__ == '__main__':
    main()
