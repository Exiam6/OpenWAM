"""Package the completed CPU replay and local contribution for the existing report."""
import hashlib
import html
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

PRIVATE = Path('/home/zifanz4/openwam-experiments')
CODE = Path('/home/zifanz4/openwam-trace-contribution')
RESULT = Path('/data02/zifanz4/openwam-experiments/results/policy-trace-replay-20260921')
SITE = Path('/home/zifanz4/research-reports/dist')
ASSET = SITE / 'assets/policy-trace-20260921'
ASSET.mkdir(exist_ok=True)
audit = json.loads((RESULT/'audit.json').read_text())
validation = json.loads((RESULT/'validation.json').read_text())
assert audit['records'] == 1881 and audit['traces'] == 18 and len(audit['comparisons']) == 9
for name in ['audit.json','source-manifest.json','validation.json','policy-trace.patch']:
    shutil.copy2(RESULT/name, ASSET/name)
shutil.copy2(CODE/'benchmarks/POLICY_TRACE.md', ASSET/'POLICY_TRACE.md')
shutil.copy2(PRIVATE/'scripts/audit_policy_trace_replay.py', ASSET/'audit_policy_trace_replay.py')
brief = f'''# Draft contribution: optional policy request/action tracing

Status: local reviewable patch; no upstream PR or maintainer message sent.
Base: {validation['upstream_base']}
Contribution: {validation['contribution_commit']}
Patch SHA256: {validation['patch_sha256']}

Repeated runs can agree on task success while observations diverge before the
next buffered action chunk. An optional wrapper records submitted request,
image-payload, prompt and state hashes, returned server actions and step counters.
The comparator reports each field's first differing event, lengths and metadata.
Reset and error events count as events. No default evaluation or training changes.

Four files, 363 added lines including documentation and tests. Runtime utility:
benchmarks/utils/policy_trace.py. Usage: benchmarks/POLICY_TRACE.md.

Validation: 6 unit tests including actual RoboTwin eval 20D-to-16D action parity;
Ruff and diff checks passed. Patch applies to the pinned upstream base. CPU-only
saved-response replay processed all 18 existing traces / 1881 requests; all 1899
source-file hashes remained unchanged. It recovered request/image/state index1
and action index32 differences in all 6 original closed-loop pairs, and equality
in all 3 short fixed-command pairs. This is software validation using old data,
not new independent evaluation or live WebSocket validation.

Scope/limits: closes its underlying client; use one wrapper per synchronous loop.
Log/serialization failures propagate; tracing adds I/O and timing overhead.
Recorded actions are server responses, not measured joint motion. Native RoboTwin
joint_action.vector is drive targets. Encoded image hashes are not pixel metrics.
Finite trace equality proves neither global determinism nor a fixed root cause.
No claim of RAE superiority or closed-loop representation improvement.

Apply in a clean checkout of the pinned base:
  git apply --check policy-trace.patch
  git apply policy-trace.patch
  python -m unittest tests.test_policy_trace -v
  python -m benchmarks.utils.policy_trace repeat0.jsonl repeat1.jsonl

Use the existing thin-client dependencies. The full replay script reads the
original saved request gzip files; adjust SOURCE/CODE/OUT for another workspace.
The public trace archive includes generated lightweight logs, not all raw images.
'''
(ASSET/'REVIEW.md').write_text(brief)
(PRIVATE/'TRACE_CONTRIBUTION.md').write_text(brief)
with tarfile.open(ASSET/'recorded-replay-traces.tar.gz','w:gz') as tar:
    traces=sorted(RESULT.glob('*.jsonl'));assert len(traces)==18
    for p in traces:tar.add(p,arcname=p.name)
rows=[]
for c in audit['comparisons']:
    d=c['first_different_event']
    rows.append('<tr><td>'+('</td><td>'.join(html.escape(str(x)) for x in [
        '闭环记录' if c['study']=='closedloop' else '两步固定动作记录', c['condition'], c['seed'],
        '/'.join(map(str,c['event_counts'])), d['request_sha256'] if d['request_sha256'] is not None else '无',
        d['action_sha256'] if d['action_sha256'] is not None else '无', '相同' if c['same_recorded_trace'] else '不同'
    ]))+'</td></tr>')
now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 可审阅的策略轨迹诊断补丁</title><link rel="stylesheet" href="style.css"></head><body>
<header><a href="index.html">Embodied Research Notebook</a><span>CPU 验证 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / CONTRIBUTION</p><nav aria-label="报告目录"><a href="#status">交付状态</a><a href="#scope">改动范围</a><a href="#validation">验证结果</a><a href="#next">下一步</a><a href="#artifacts">审阅材料</a></nav></aside><main>
<div class="hero"><div class="eyebrow">Local contribution draft</div><h1>把重复性证据，<br><em>变成可复用的诊断工具。</em></h1><p class="lead">新增一个默认不启用的请求／动作记录器与首次差异比较器。补丁可下载审阅，尚未提交上游 PR。</p></div>
<section id="status"><h2>已完成：补丁、测试、全部旧记录回放</h2><div class="callout">6项测试通过；18条已有记录、1,881次请求的纯CPU回放通过；1,899个来源文件的哈希前后不变。这是软件验证，没有增加评测样本、模型推理或仿真。原分歧根因仍未查明，也没有新的RAE或适配器成功率收益结论。</div><p>更新于 NOW。<a href="trajectory.html">原12条闭环诊断</a> · <a href="attribution.html">六段两步固定动作探针</a> · <a href="norm.html">80次归一化评测</a>。</p></section>
<section id="scope"><h2>仅四个文件，默认行为不变</h2><p>新增运行时工具benchmarks/utils/policy_trace.py，配套6项测试、使用说明和README链接，共363行。需要使用者主动包装已有WSPolicyClient；不接入默认评测，不修改模型、训练、执行器或仿真设置。</p><ul><li>保留predict与predict_once原有重试选择，只委派一次，并返回同一个响应对象。</li><li>记录完整请求、编码图像字节、提示词和状态的哈希，以及状态值、返回动作与服务端步数；记录reset及错误事件。</li><li>逐字段报告第一次差异，额外检查事件数量和元数据；长度不同即使前缀一致，也不判为相同。</li><li>拒绝覆盖已有记录；写入和序列化失败会中止调用。使用单个同步循环，关闭包装器会关闭底层客户端。</li></ul><p>测试覆盖参数与返回对象一致性、随机数状态不变、错误传播、已有文件保护、不等长记录、损坏JSON／哈希、正负零差异，以及实际RoboTwin eval转换后16维目标的一致性。Ruff和diff检查通过；补丁可应用到固定上游提交。</p></section>
<section id="validation"><h2>旧数据的九对结果全部保留</h2><div class="table-wrap"><table><thead><tr><th>记录来源</th><th>条件</th><th>场景</th><th>两轮事件数</th><th>首次请求差异</th><th>首次动作差异</th><th>整段记录</th></tr></thead><tbody>ROWS</tbody></table></div><p>索引从0开始。这批回放只记录预测事件，故索引与原控制步一致；通用工具中的reset和错误也占据事件索引。全部六对闭环记录的图像与状态哈希也在索引1首次不同，动作在索引32首次不同。三对短固定动作记录一致。回放只读保存的请求并返回保存的动作，未创建模型、仿真或网络客户端。</p><p>验证能准确区分现有正反例，不等于在新的独立运行中复现了相同现象。底层WebSocket在线整合尚未验证，日志IO也可能改变运行时序。</p></section>
<section id="next"><h2>这项贡献能解决什么，仍不能解决什么</h2><p>它让后续表征实验能够区分“输入先变化”与“策略动作先变化”，减少只看成功率或latent误差导致的错误归因。返回动作不是实测运动；RoboTwin原生joint_action.vector是驱动目标。图像编码字节哈希也不是像素误差或接触信息。</p><p>今晚GPU分配的截止时间为23:45:59 CDT。新的上下文对照未在预留窗口内完成检查，已延期且未启动，当前只完成CPU工作。下一项GPU实验仍需单独冻结并通过检查：匹配首场景、模型驻留和执行顺序，比较实时请求路径与固定响应路径，判断此前未复现是否与运行上下文有关。仅改变transport仍可能改变推理负载，方案须明确控制或记录这一差别。它用于缩小原因范围，不承诺定位根因。此前80次评测不重跑，不按本轮旧样本挑选新权重。</p></section>
<section id="artifacts"><h2>可下载审阅</h2><p>上游基线：<code>BASE</code><br>本地贡献提交：<code>COMMIT</code>。未推送OpenWAM上游，也未向维护者发消息。</p><ul>LINKS</ul><pre>git apply --check policy-trace.patch
git apply policy-trace.patch
python -m unittest tests.test_policy_trace -v
python -m benchmarks.utils.policy_trace repeat0.jsonl repeat1.jsonl</pre><p>在固定基线的干净checkout中应用。记录文件来自各自匹配的场景和协议；比较器不替代实验设计。补丁SHA256：<code>PATCH</code>。</p></section></main></div></body></html>'''
files=[('REVIEW.md','审阅说明'),('policy-trace.patch','完整补丁'),('POLICY_TRACE.md','使用文档'),('validation.json','代码检查与补丁校验'),('audit.json','全部九对CPU回放结果'),('source-manifest.json','1,899个来源文件哈希'),('audit_policy_trace_replay.py','CPU回放脚本'),('recorded-replay-traces.tar.gz','全部18条轻量记录')]
page=page.replace('NOW',now).replace('ROWS',''.join(rows)).replace('BASE',validation['upstream_base']).replace('COMMIT',validation['contribution_commit']).replace('PATCH',validation['patch_sha256']).replace('LINKS',''.join(f'<li><a href="assets/policy-trace-20260921/{f}">{label}</a></li>' for f,label in files))
(SITE/'contribution.html').write_text(page)
p=SITE/'index.html';s=p.read_text();assert 'id="trace-contribution-latest"' not in s
s=s.replace('<main id="top">','<main id="top"><div class="callout" id="trace-contribution-latest"><strong>新增：可审阅的策略轨迹诊断补丁。</strong>6项测试与全部18条旧记录的CPU回放通过；未启动新GPU实验。<a href="contribution.html">补丁、验证证据和下一步</a>。</div>',1)
s=s.replace('<nav aria-label="报告目录"><a href="trajectory.html">最新：闭环重复性</a>','<nav aria-label="报告目录"><a href="contribution.html">最新：诊断贡献补丁</a><a href="attribution.html">两步归因诊断</a><a href="trajectory.html">闭环重复性</a>',1)
s=s.replace('状态：闭环重复性诊断已完成<br>12条轨迹与逐步记录','状态：本地贡献补丁可审阅<br>6项测试与1,881条记录验证')
p.write_text(s)
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ASSET.iterdir() if f.is_file()}
(ASSET/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'page':'contribution.html','assets':len(manifest),'patch_sha256':validation['patch_sha256']},indent=2))
