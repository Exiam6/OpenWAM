"""Publish all planned context trials, including incomplete/failed outcomes."""
import html
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

ROOT=Path('/data02/zifanz4/openwam-experiments');R=ROOT/'results/context-pilot-20260922';SITE=Path('/home/zifanz4/research-reports/dist');A=SITE/'assets/context-pilot-20260922'


def main():
    A.mkdir(exist_ok=False)
    summary=json.loads((R/'summary.json').read_text());audit=json.loads((R/'audit.json').read_text());plan=json.loads((R/'run-plan.json').read_text());launch=json.loads((R/'launch.json').read_text())
    assert audit['all_completed_record_checks_passed']
    for name in ['summary.json','audit.json','result-manifest.json','source-manifest.json','PROTOCOL.md','run-plan.json','commands.json','scenes.json','launch.json','launch-invocation.json','launcher.log','exit-code.txt','outer-exit-code.txt','lifecycle-cpu-checks.json','integration-cpu-checks.json','port-preflight-diagnostic.json','video-audit.json']:
        if(R/name).exists():shutil.copy2(R/name,A/name)
    with tarfile.open(A/'source-snapshot.tar.gz','w:gz')as tar:
        frozen=json.loads((R/'source-manifest.json').read_text())
        for relative in frozen['runtime_files']:
            if relative.startswith('scripts/') or relative.endswith(('.yaml','.yml')):tar.add(ROOT/relative,arcname=relative)
        for name in ['PROTOCOL.md','run-plan.json','source-manifest.json']:tar.add(R/name,arcname=name)
    trial_rows=[];gallery=[];links=[]
    for spec in plan['trials']:
        name=spec['id'];t=summary['trials'][name];d=R/name;dst=A/name;dst.mkdir()
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.suffix in ['.json','.jsonl','.log','.png']:shutil.copy2(p,dst/p.name)
            if(d/'requests').exists():shutil.copytree(d/'requests',dst/'requests')
        if(dst/'transport-events.jsonl').exists():
            events=[json.loads(x)for x in(dst/'transport-events.jsonl').read_text().splitlines()[1:]]
            attempts={e['fixed_command_index']:e['online_calls']for e in events if 'fixed_command_index'in e}
            online=sum(attempts.values())
        else:online=0
        status=t['status']['status'];error=t['status'].get('error','')
        trial_rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in [name,spec['mode'],status,t['commands_recorded'],online,round(t['status'].get('elapsed_seconds',0),2),error or '—'])+'</tr>')
        available=[p for p in dst.iterdir()if p.is_file()]
        links.append('<li>'+name+': '+(' · '.join(f'<a href="assets/context-pilot-20260922/{name}/{p.name}">{p.name}</a>'for p in sorted(available))or'未启动或无输出')+'</li>')
        if(dst/'step1-head.png').exists():gallery.append(f'<figure style="max-width:46%;margin:0"><img loading="lazy" src="assets/context-pilot-20260922/{name}/step1-head.png" alt="{name}首动作后下一帧" style="width:100%"><figcaption>{name} · 首动作后的控制观测 · <a href="assets/context-pilot-20260922/{name}/requests/step-0001.json.gz">完整请求</a></figcaption></figure>')
    for video in audit['videos']:
        shutil.copy2(R/video['source'],A/video['trial']/'two-command.mp4')
        links.append(f'<li><a href="assets/context-pilot-20260922/{video["trial"]}/two-command.mp4">{video["trial"]}完整两步视频</a>（非完整任务）</li>')
    fields=['initial_request_equal','first_post_action_request_equal','first_post_action_cameras_equal','first_post_action_eef_equal','before_first_action_state_equal','after_first_action_state_equal','first_planners_equal']
    comparison_rows=[]
    for c in summary['comparisons']:
        values=[c['left']+'/'+c['right']]+[('相同'if c[k]else'不同')if c['valid']else'未完成，不作比较'for k in fields]
        comparison_rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>'for v in values)+'</tr>')
    complete=summary['completed_trials']==4
    all_same=complete and all(c['valid']and all(c[k]for k in fields)for c in summary['comparisons'])
    title='四段固定动作上下文对照均未出现记录分歧'if all_same else('上下文对照完成，逐项差异如下'if complete else'上下文对照已停止，保留全部完成与未完成记录')
    finding=('两种条件各两次，所有配对在初始请求、首动作后的下一次请求／图像／EEF、动作前后实测状态及首动作规划记录上都相同。本轮仍未复现此前完整闭环的分歧；不能据此断言在线活动没有影响或系统全局确定。'if all_same else('所有预定配对均列出；差异只作描述，不能单独定位GPU负载、物理、渲染或规划根因。'if complete else'未完成的试次和配对按原方案保留，不补跑、不更换种子。当前数据不足以完成预定对照，不能据此得出在线活动是否影响重复性的结论。'))
    now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    engineering = ''
    if not complete:
        engineering = '<section id="engineering"><h2>本次停止的具体原因</h2><p>运行器报告：<code>'+html.escape(launch.get('error','unknown'))+'</code>。A0已完成；B0三个GPU检查样本均空闲、无ECC错误、无计算进程，但随后端口18848的普通bind检查失败。B0没有启动模型或仿真进程，B1和A1也未运行。</p><p>后续纯CPU测试在临时本地端口复现了一个预检缺陷：没有监听程序、仅存在TIME_WAIT时，普通bind仍报同样的错误；使用SO_REUSEADDR的bind可以成功。本次失败当时没有保存socket状态，随后检查18848已无连接，所以这只能说明一个已验证的可能机制，不能确定原失败仅由它造成。</p><p>没有修改被冻结的运行器或方案后补跑。<a href="assets/context-pilot-20260922/port-preflight-diagnostic.json">CPU诊断原始记录</a>包含TIME_WAIT状态及两种bind的结果。需要在未来独立验证端口预检，但它不是OpenWAM模型性能或仿真根因结论。</p></section>'
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 固定动作上下文对照结果</title><link rel="stylesheet" href="style.css"></head><body><header><a href="index.html">Embodied Research Notebook</a><span>{now}</span></header><div class="layout"><aside><p>OPENWAM / CONTEXT PILOT</p><nav aria-label="报告目录"><a href="#result">结论</a><a href="#design">固定方案</a><a href="#trials">全部试次</a><a href="#pairs">全部配对</a><a href="#observations">观测</a><a href="#next">下一步</a><a href="#artifacts">复现材料</a></nav></aside><main><div class="hero"><div class="eyebrow">Bounded four-trial diagnostic</div><h1>固定执行动作，<br><em>检验在线活动。</em></h1><p class="lead">新模型／仿真进程，ABBA顺序，四段两步上限。不重复此前80次评测，不估计任务成功率。</p></div><section id="result"><h2>{title}</h2><div class="callout">{finding}</div><p>完成{summary['completed_trials']}/4段，保留{summary['commands_recorded']}条完整动作记录。运行器退出码{launch.get('exit_code','unknown')}；该运行器记录耗时{launch.get('elapsed_seconds',0):.2f}秒（源码校验发生在其计时前，也计入外层600秒时限）。外层退出码{(R/'outer-exit-code.txt').read_text().strip()}。<a href="context.html">设计与CPU检查</a> · <a href="attribution.html">前次短探针阴性结果</a> · <a href="trajectory.html">原闭环分歧</a>。</p></section>
{engineering}<section id="design"><h2>两个条件，只改变在线活动</h2><p>A组保留原模型驻留，正常ping／reset，不发送预测；B组照常发送每条请求并等待完整返回，但仍执行与A组完全相同的保存命令。两组同样记录规划与实际状态。B组的返回动作即使不同，也不用于改写执行动作或筛除试次。</p><p>仅首场景400000、clean，顺序A0、B0、B1、A1，每次全新服务器和仿真进程。每段最多两条命令；原同步32动作缓存。差别包含在线推理、通信和等待，不能解释成纯GPU负载对照。服务器启动限180秒且受总时限约束，整个新运行有600秒外层限时及最多5秒退出清理。失败即停，后续试次不替换。</p><p>运行前7项进程／客户端检查及此前14项上下文检查通过；{audit['runtime_sources_verified']}个运行来源、{audit['study_inputs_verified']}个方案／输入文件在运行后再次校验。所有完整记录的{audit['requests_audited']}个请求、{audit['decoded_images_audited']}张相机图像和{audit['planner_calls_recorded']}次规划记录通过审计。</p></section>
<section id="trials"><h2>全部四个计划试次</h2><div class="table-wrap"><table><thead><tr><th>试次</th><th>条件</th><th>状态</th><th>完整动作记录</th><th>在线请求尝试</th><th>进程耗时s</th><th>错误</th></tr></thead><tbody>{''.join(trial_rows)}</tbody></table></div><p>完整动作记录数不代表失败中途绝无部分运动；若出现不完整记录，请查看incomplete-trace.json。两步结束的原生Fail标签不是完整任务失败率。</p></section>
<section id="pairs"><h2>预先指定的全部配对</h2><div class="table-wrap"><table><thead><tr><th>配对</th><th>初始请求</th><th>动作后请求1</th><th>动作后图像</th><th>动作后EEF</th><th>动作前实测状态</th><th>动作后实测状态</th><th>首动作规划</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div><p>A0/A1和B0/B1为组内重复，A0/B0和A1/B1为指定跨组比较。索引从0开始。实际qpos／qvel来自getter；原生joint_action.vector是驱动目标。左右臂在此机器人中读取同一整机articulation，不算独立样本。</p></section>
<section id="observations"><h2>首动作后的下一帧</h2><div style="display:flex;gap:16px;flex-wrap:wrap">{''.join(gallery)or'<p>没有完整的第二步观测可展示。</p>'}</div></section>
<section id="next"><h2>贡献与下一步</h2><p>本轮只诊断评测路径，不比较RAE与VAE，也不证明适配器或成功率提升。已交付的请求／动作记录器、完整负结果与测量语义仍可作为社区贡献审阅。按照停止规则，这项限定探针结束后不无限追加仿真排查；回到表征／扰动实验，明确区分特征恢复误差、动作差异与闭环收益，并使用未用于挑选权重的验证数据。</p><p>初始状态和运行历史未记录的部分、共享节点活动以及有限重复次数仍限制解释。即使所记录字段一致，也不表示覆盖了求解器全部隐藏状态。</p></section>
<section id="artifacts"><h2>来源、退出码与全部输出</h2><p>方案源提交：<code>{plan['private_source_commit']}</code>。模型检查点SHA验证通过；权重不包含在网页下载包中。</p><ul>'''
    for name,label in [('PROTOCOL.md','冻结方案'),('summary.json','完整结果'),('audit.json','逐项数据审计'),('source-manifest.json','全部来源哈希'),('run-plan.json','固定顺序与预算'),('launch.json','所有进程退出与清理记录'),('launch-invocation.json','外层运行命令'),('source-snapshot.tar.gz','脚本与配置快照')]:page+=f'<li><a href="assets/context-pilot-20260922/{name}">{label}</a></li>'
    page+=''.join(links)+'</ul></section></main></div></body></html>'
    (SITE/'context-pilot.html').write_text(page)
    p=SITE/'index.html';s=p.read_text();assert 'id="context-pilot-latest"'not in s;s=s.replace('<main id="top">',f'<main id="top"><div class="callout" id="context-pilot-latest"><strong>上下文短对照已结束：{summary["completed_trials"]}/4段完成。</strong>{html.escape(title)}。<a href="context-pilot.html">全部试次、配对、视频及退出记录</a>。</div>',1);p.write_text(s)
    p=SITE/'context.html';s=p.read_text();s=s.replace('<main>','<main><div class="callout"><strong>运行结果已公布：</strong><a href="context-pilot.html">查看全部上下文对照结果</a>。本页保留运行前的方案与检查记录。</div>',1);p.write_text(s)
    hashes={str(p.relative_to(A)):__import__('hashlib').sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};(A/'artifact-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'page':'context-pilot.html','title':title,'completed_trials':summary['completed_trials'],'assets':len(hashes)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
