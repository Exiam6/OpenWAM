"""Artifact-only status refresh; never launches jobs or changes experimental sources."""
import datetime
import html
import json
from pathlib import Path

P = Path('/home/zifanz4/openwam-experiments')
S = P / 'studies/endtoend-20260923'
E = Path('/data02/zifanz4/openwam-experiments/endtoend-20260923')
O = E / 'recovery/evaluation-20260925'
STATE = Path('/home/zifanz4/.local/state/openwam-selfcheck')
SITE = Path('/home/zifanz4/research-reports')

def read(p):
    return json.loads(p.read_text()) if p.exists() else None

def write(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)

t = datetime.datetime.now().astimezone().isoformat()
q = read(S / 'progress.json')
plan = read(S / 'evaluation-recovery-20260925.json')
controller = read(O / 'controller/progress.json')
cexit = read(O / 'controller/exit.json')
gate = read(S / 'evaluation-ready.json')
cohort = read(S / 'fresh-cohort-integrity.json')
queues = {}
for i in range(1,7):
    root=O/'queues'/str(i)
    queues[str(i)]={name:read(root/(name+'.json')) for name in ['started','progress','admitted','exit']}
records=[read(p) for p in (E/'evaluation').glob('*/*/*/seed-*/result.json')]
complete=sum(x.get('status')=='complete' for x in records)
failed=sum(x.get('status')=='infrastructure_failure' for x in records)
started=sum(v['started'] is not None for v in queues.values())
admitted=sum(v['admitted'] is not None for v in queues.values())
finished=sum(v['exit'] is not None for v in queues.values())
active=sum(v['admitted'] is not None and v['exit'] is None for v in queues.values())
diag=sum(sum(1 for _ in p.open()) for p in (E/'action-diagnostics').glob('*/rows.jsonl'))
closed=read(E/'analysis/closed-loop.json');audit=read(E/'analysis/audit.json')
status='evaluation_recovery_failed' if cexit and cexit['returncode'] else 'evaluation_recovery_complete' if cexit else 'paired_evaluation_in_progress' if active else 'independent_evaluation_queues_waiting' if started else 'cpu_integrity_gate_running'
if failed: status='evaluation_partial_infrastructure_failure'
nextstep='Run each original queue only on its own verified idle slot; preserve all fixed checkpoints/scenes/conditions. Complete all 5400 rollouts and 48600 diagnostics before independent raw-result audit. Resolve whether PCA improves native robustness versus S-VAE without clean-success regression. Do not repeat expired stages or failed queues; absolute evaluation cutoff Sep28 10:41:59 CDT, stage cutoff Sep28 13:41:59 CDT.'
if failed: nextstep='Preserve every partial result and failed queue. CPU reproduction confirms prompt-cache eviction raises on distinct prompt33. Isolated repair passed20000 cache operations and five first-action predictions on one training frame match exactly, including post-eviction recomputation; not applied to frozen code or validated in full simulator/WebSocket evaluation. Prepare training-only integration validation before any separately registered recovery; no automatic replay, deadline extension or benefit claim.'
if cexit: nextstep='Inspect the complete terminal evidence and retain every failure. No automatic replay or deadline extension. A primary benefit claim requires all 5400 rollouts plus the independent audit.'
q.update(updated_at=t,status=status,next_step=nextstep,selfcheck_reasoned_at=t,owned_gpu_workers=active,closed_loop=closed,closed_loop_audit=audit,controlled_native295M_primary_passed=bool(closed and audit and audit.get('passed') and closed.get('joint_primary_pass')))
q['evaluation_recovery'].update(controller=controller,exit=cexit,queues=queues,gate_passed=bool(gate and gate.get('passed')),cohort_integrity_passed=bool(cohort and cohort.get('passed')),complete_rollouts=complete,technical_failures=failed,diagnostic_predictions=diag,stage_deadline=plan['stage_deadline'],evaluation_deadline=plan['evaluation_deadline'])
q['resource_note']='Six original cm009GPU1-6 queues independently check resources; no simultaneous-idle barrier. No GPU is claimed reserved merely because a queue exists. GPU0 ECC failure remains excluded.'
q['public_report_status']='Existing public site access restored Sep25; see publication.json for actual deployment status.'
q['monitoring_schedule'].pop('next_scheduled',None)
q['monitoring_schedule']['delivery_note']='3-hour cron retained. Sep25 00:00 trigger was processed at19:35 after a direct continue; queued triggers do not prove timely reasoning.'
write(S/'progress.json',q)
write(STATE/'last-reasoned.json',{'time':t,'status':status,'next_step':nextstep,'study':str(S),'evidence':str(S/'progress.json')})
head=f'''# Current end-to-end validation — {t}

All nine fixed trainings completed exit0 (12000 microsteps / 6000 optimizer updates). All three seed-paired batch streams match. No training is queued or still running. The 150 fresh expert trajectories passed the frozen file/image/pose exclusion check; all nine full final-checkpoint payloads were hashed and the evaluation-ready gate passed: {bool(gate and gate.get('passed'))}.

The old six-slot resource barrier expired Sep25 19:11:52 CDT. Its waiter and controller exited1; original records/deadlines remain unchanged. Under the existing complete-validation authorization and direct continue, the separate first-start recovery began19:46:45. It changes operational admission only: each original cm009GPU1-6 queue independently waits for three idle observations. Same scientific code, scenes, routes/seeds, final checkpoints, conditions, inference and statistics; no completed experiment repeated.

Current snapshot: {started}/6 finite queues started; {admitted} admitted, {active} still active, {finished} terminal. Completed learned-policy rollouts {complete}/5400; technical failures {failed}; action diagnostics {diag}/48600. Status: {status}. Queue start is not GPU reservation or policy inference. Primary PCA-vs-SVAE benefit remains unestablished unless the full matrix and independent audit pass. Model scale remains295M, not5B; PCA is not fullRAE.

Absolute evaluation cutoff Sep28 10:41:59 CDT; overall stage cutoff Sep28 13:41:59 CDT, unchanged from the previous outer-controller deadline. Each queue also retains the original232800-second execution cap. Failure/partial results retained without automatic retry. No layers-window restart, old80 or expired56 replay, new agents, new recurring timers or unrelated process termination.

Next: {nextstep}

The existing public site is accessible again; actual publication outcome is in studies/endtoend-20260923/publication.json. Local report: reports/endtoend-20260923/index.html. Keep the existing3-hour monitor; the delayed00:00 message is not backfilled as a completed selfcheck. Read the recovery plan and live controller/queue records at every check; refresh this snapshot with scripts/endtoend/report_evaluation_recovery_20260925.py, not the old frozen update_status.py.
'''
for name in ['STATUS.md','NEXT_STEPS.md']:
    p=P/name;old=p.read_text();begin='<!-- END_TO_END_CURRENT_BEGIN -->';end='<!-- END_TO_END_CURRENT_END -->'
    if begin in old:
        prefix,remainder=old.split(begin,1);_,suffix=remainder.split(end,1)
        p.write_text(prefix+begin+'\n'+head+'\n'+end+suffix)
    else:
        p.write_text(begin+'\n'+head+'\n'+end+'\n\n'+old)
failure_note='<p><strong>评测基础设施故障：</strong>已独立复现文本缓存第33条不同指令写入时的异常。缓存修复候选通过20000次CPU操作检查，并完成单训练帧的5次原生推理首动作一致性检查（差值0），未修改冻结代码，尚未验证完整WebSocket／模拟器流程。失败回合不记作任务失败；当前不完整矩阵不能支持收益结论。后续时间压缩评测也包含同一缓存实现。</p>' if failed else ''
queue_rows=''
for i,v in queues.items():
    label=('完成' if v['exit']['returncode']==0 else '失败，保留记录') if v['exit'] else '已进入评测队列执行' if v['admitted'] else '等待本卡空闲' if v['started'] else '尚未启动'
    queue_rows+=f'<tr><td>{i}</td><td>{label}</td></tr>'
training_rows=''.join(f'<tr><td>{html.escape(k)}</td><td>完成，退出码 0</td><td>6000 / 6000</td></tr>' for k in sorted(q['training']))
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 完整表征验证</title><style>body{{font:17px/1.65 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;background:#f5f8fc;color:#182b3b}}a{{color:#07588c}}.card{{background:white;border:1px solid #dae4ed;padding:22px;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}.stats{{display:flex;flex-wrap:wrap;gap:22px}}.stats b{{font-size:1.8rem;display:block}}.sub{{color:#445c70}}</style></head><body><nav><a href="index.html">研究主页</a></nav><h1>新表征能否改善 OpenWAM 的控制？</h1><p class="sub">状态快照：{t}（美国中部时间）。此页按核查更新，并非实时遥测。</p><div class="card"><div class="stats"><div><b>9 / 9</b>固定预算训练完成</div><div><b>{complete} / 5400</b>闭环评测回合完成</div><div><b>{diag} / 48600</b>动作诊断完成</div></div><p><strong>尚未证实新表征带来闭环收益。</strong>下一步直接比较 PCA48、原生 DINO-SVAE48 和 Wan48 在同一批场景上的控制效果。</p></div>
<div class="card"><h2>本次推进</h2>{failure_note}<p>9 月 25 日 19:49，150 条专家轨迹的数据身份审计、9 份最终权重的完整哈希检查和训练批次配对检查全部通过。新评测阶段已启动；当前 {started} 个有限队列已建立，{admitted} 个已获准进入执行，{active} 个仍在执行，{finished} 个已退出。排队不代表正在占用 GPU。</p><p>先前调度等待 6 张指定卡同时空闲，于 19:12 到期退出，未产生策略评测结果。现已改为各队列独立等卡，保留原始失败记录。训练、数据、评测条件及统计方法未改变。绝对评测截止时间为 9 月 28 日 10:41（中部时间），总阶段截止时间仍为当日 13:41。</p><table><thead><tr><th>原评测队列</th><th>状态</th></tr></thead><tbody>{queue_rows}</tbody></table><p>技术失败：{failed} 个回合。失败与未完成项全部保留，不自动重跑或选择有利结果。</p></div>
<div class="card"><h2>公平比较与验收标准</h2><p>相同 105 条训练轨迹、原生三相机输入、配对初值和批次；3 种表征 × 3 个训练种子，每组固定 6000 次优化更新，只使用最终检查点。</p><table><thead><tr><th>训练</th><th>状态</th><th>优化更新</th></tr></thead><tbody>{training_rows}</tbody></table><p>每任务预先固定 50 个专家可完成的新场景，共 150 个。全部交叉测试 3 个种子、3 种表征、干净输入、噪声 0.04、噪声 0.10、头部相机真实偏转 ±5°，合计 5400 回合。</p><p><strong>主判据：</strong>PCA 相对原生 S-VAE 的干净成功率差，95% 区间下界高于 −5 个百分点；同时三种扰动的平均成功率差，95% 区间下界高于 0。完整结果须经独立原始记录审计。只用 3 个训练种子，对种子总体变异的估计仍有限。</p></div>
<div class="card"><h2>已有结论及其边界</h2><p>独立 60 场景的离线诊断已经数值审计：在噪声 0.10 下，PCA48 相对 Wan48 的目标误差降低 73.05%（95% 区间 −79.60% 至 −65.72%），短时状态误差降低 39.11%（−48.70% 至 −27.30%）。夹爪收益仍不确定。这些是离线信息读出结果，不能代替成功率，也不能证明 PCA 胜过原生 S-VAE。</p><p>当前控制模型约 2.95 亿可训练参数，不能代表 5B 模型收益。PCA 没有像素解码器，不能称完整 RAE。只有闭环结果支持改进后，才另行固定大模型验证方案。</p></div>
<div class="card"><h2>可贡献到原项目的内容</h2><p>候选交付包括可选 PCA 编码器、训练与部署一致性检查、可复现的配对扰动评测，以及平移、旋转和夹爪分项诊断。动作误差与真实接触记录分别报告，避免把夹爪闭合代理指标当成接触测量。无提升或任务间取舍也会保留。</p><p>下一步要回答：更容易从表征读出信息，是否真的转化为更稳健的控制，而且不牺牲干净输入下的成功率。</p></div></body></html>'''
assert gate and gate['passed'], 'Do not publish the completed-gate wording before verification'
for p in [P/'reports/endtoend-20260923/index.html',SITE/'dist/endtoend.html']:p.write_text(page)
home=SITE/'dist/index.html';h=home.read_text();h=h.replace('更新于 2026.09.22','更新于 2026.09.25')
h=h.replace('当前计划 · 09.22：自检已暂停，新实验尚未启动。','历史计划 · 09.22：当时暂停后制定的 12 小时方案（窗口已结束）。')
callout=f'<div class="callout" id="endtoend-latest"><strong>当前核查 · 09.26：9 组训练完成；闭环出现缓存技术故障。</strong>完整闭环评测 {complete}/5400 回合；技术失败{failed}回合；保留不完整结果，尚无成功率收益结论。<a href="endtoend.html">最新状态、固定主判据与结论边界 →</a></div>'
import re
h=re.sub(r'<div class="callout" id="endtoend-latest">.*?</div>','',h,count=1)
h=h.replace('<main id="top">','<main id="top">'+callout,1)
if 'href="endtoend.html">完整表征验证' not in h:h=h.replace('<nav aria-label="报告目录">','<nav aria-label="报告目录"><a href="endtoend.html">完整表征验证</a>',1)
home.write_text(h)
print(json.dumps({'time':t,'status':status,'queues_started':started,'admitted':admitted,'complete_rollouts':complete,'technical_failures':failed,'gate_passed':bool(gate and gate['passed'])}))
