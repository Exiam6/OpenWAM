"""Publish every short open-loop attribution pair with explicit measured semantics."""
import base64,gzip,hashlib,html,json,re,shutil,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RUN=Path('/data02/zifanz4/openwam-experiments/results/attribution-20260921');SITE=Path('/home/zifanz4/research-reports/dist');A=SITE/'assets/attribution-20260921'
s=json.loads((RUN/'summary.json').read_text());assert s['all_integrity_checks_passed']and(RUN/'exit-code.txt').read_text().strip()=='0'
A.mkdir(parents=True,exist_ok=True)
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
def link(n,label=None):return f'<a href="assets/attribution-20260921/{n}">{html.escape(label or n)}</a>'
yes=lambda x:'相同'if x else'不同'
files=['protocol.md','source-freeze.json','fixed-commands.json','scenes.json','launch.json','cpu-checks.json','summary.json','supplemental-audit.json','exit-code.txt']
for name in files:
 if(RUN/name).exists():shutil.copyfile(RUN/name,A/name)
for p in RUN.glob('round*-preflight.json'):shutil.copyfile(p,A/p.name)
with tarfile.open(A/'reproduction-scripts.tar.gz','w:gz')as t:
 t.add(ROOT/'ATTRIBUTION_PROTOCOL.md',arcname='ATTRIBUTION_PROTOCOL.md')
 for name in ['attribution_trace.py','attribution_client.py','check_attribution_trace.py','summarize_attribution.py','run-attribution-probe.sh','trajectory_trace.py','repeat_common.py','capture_repeat_inputs.py','norm_client.py','norm_common.py','robustness_client.py','robustness_common.py']:t.add(ROOT/'scripts'/name,arcname='scripts/'+name)
 for name in ['fixed-commands.json','scenes.json','source-freeze.json','protocol.md']:t.add(RUN/name,arcname='results/attribution-20260921/'+name)
 t.add(ROOT/'audit_attribution_results.py',arcname='audit_attribution_results.py')
video_manifest=[]
for rid in[0,1]:
 name=f'round{rid}';d=RUN/name;(A/name).mkdir(exist_ok=True)
 videos=sorted((d/'runtime/eval_result').rglob('episode*.mp4'),key=lambda p:int(re.search(r'episode(\d+)',p.name).group(1)));assert len(videos)==3
 shutil.copyfile(d/'summary.json',A/name/'summary.json')
 for seed,video in zip([400000,400001,400002],videos):
  trace=d/f'seed-{seed}-trace.jsonl';shutil.copyfile(trace,A/name/trace.name)
  dest=f'{name}/seed-{seed}.mp4';shutil.copyfile(video,A/dest);video_manifest.append({'round':rid,'seed':seed,'path':dest,'sha256':hashlib.sha256(video.read_bytes()).hexdigest()})
  for step in [0,1]:
   payload_path=d/'requests'/f'seed-{seed}'/f'step-{step:04d}.json.gz';payload=json.loads(gzip.decompress(payload_path.read_bytes()));shutil.copyfile(payload_path,A/name/f'{seed}-step{step}.json.gz')
   (A/name/f'{seed}-step{step}.png').write_bytes(base64.b64decode(payload['images']['head_camera']))
(A/'video-manifest.json').write_text(json.dumps(video_manifest,indent=2)+'\n')
with tarfile.open(A/'all-traces-and-requests.tar.gz','w:gz')as t:
 for rid in[0,1]:
  d=RUN/f'round{rid}'
  for p in d.glob('seed-*-trace.jsonl'):t.add(p,arcname=f'round{rid}/'+p.name)
  t.add(d/'requests',arcname=f'round{rid}/requests');t.add(d/'summary.json',arcname=f'round{rid}/summary.json')
rows=[];planner_rows=[];initial_equal=paths_equal=post_different=0
for pair in s['comparisons']:
 seed=pair['seed'];first=pair['steps'][0]
 initial_equal+=first['states']['before_execution']['all_recorded_exact']
 paths_equal+=all(x['inputs_exact']and x['results_exact']for x in first['planners'])
 post_different+=not first['states']['after_execution']['actual_articulations_exact']
 for step in pair['steps']:
  before=step['states']['before_execution'];after=step['states']['after_execution']
  maximum=max(after['actual_values'][side]['qpos']['max_abs']for side in ['left','right'])
  rows.append([seed,step['step'],yes(before['all_recorded_exact']),yes(all(x['inputs_exact']for x in step['planners'])),yes(all(x['results_exact']for x in step['planners'])),yes(after['actual_articulations_exact']),f'{maximum:.9g}',yes(after['drive_targets']['exact']),yes(step['raw_cameras_exact'])])
  for plan in step['planners']:
   planner_rows.append([seed,step['step'],plan['side'],yes(plan['inputs_exact']),'/'.join(plan['statuses']),yes(plan['results_exact']),plan.get('position',{}).get('max_abs',str(plan.get('position',{}).get('shapes','—'))),plan.get('velocity',{}).get('max_abs','—')])
headline='固定动作后，分歧发生在哪里？'
if initial_equal==paths_equal==post_different==3:
 headline='记录初态与规划路径相同，执行后真实状态不同'
conclusion=f'三个场景的首次动作：{initial_equal}/3对的全部已记录初态一致，{paths_equal}/3对的规划输入与返回路径一致，{post_different}/3对的执行后实测关节状态出现差异。'
all_equal=all(step['request_exact']and step['eef_exact']and step['fixed_commands_exact']and all(st['all_recorded_exact']for st in step['states'].values())and all(p['inputs_exact']and p['results_exact']for p in step['planners'])for pair in s['comparisons']for step in pair['steps'])
if all_equal:
 headline='两步固定动作探针未复现上一轮分歧'
 conclusion='六段短重放中，两轮的规划输入、返回路径、动作前后实测状态、末端位姿、图像和请求均完全一致；所有十二条动作都确实改变了实测qpos。这是一项未复现分歧的阴性结果，不能据此认定问题已经修复。'
conclusion+='这里的实测状态来自qpos/qvel getter；上一轮joint_action.vector是驱动目标，二者不能混用。未记录所有隐藏物理状态，不能据此认定某个仿真库有缺陷。'
images=''
for seed in[400000,400001,400002]:
 images+=f'<details><summary>场景{seed}：首动作后的下一帧</summary><div style="display:flex;gap:16px;flex-wrap:wrap">'+''.join(f'<figure style="margin:0;max-width:46%"><img loading="lazy" src="assets/attribution-20260921/round{rid}/{seed}-step1.png" alt="场景{seed} round{rid} 首动作后主相机" style="width:100%"><figcaption>Round{rid} · '+link(f'round{rid}/{seed}-step1.json.gz','完整请求')+'</figcaption></figure>'for rid in[0,1])+'</div></details>'
video_links=' · '.join(link(v['path'],f"r{v['round']} / {v['seed']}")for v in video_manifest)
body=f'''<div class="hero"><div class="eyebrow">Fixed-command attribution / 2026.09.21</div><h1>先固定动作，<br><em>再分开看规划与执行。</em></h1><p class="lead">六段两步重放，十二条保存的动作命令，零次策略推理。本页是执行路径诊断，不是任务成功率评测。</p></div>
<section id="status"><h2>{headline}</h2><div class="callout">{conclusion}</div><p>6/6段完成，来源、固定命令、计数和数据完整性审计通过。更新于{time.strftime('%Y-%m-%d %H:%M:%S %Z')}。<a href="trajectory.html">上一轮12条闭环</a>已发现观测在第1步、策略动作在第32步分歧。</p></section>
<section id="design"><h2>固定方案</h2><p>同样三个清洁场景400000–400002，两个独立仿真进程，每轮相同顺序。每个场景直接返回上一轮round0保存的前两条20维动作，由原生客户端转换为16维目标并调用take_action。初始完整请求与原清单一致。新观测不会改变所重放的动作。</p><p>通过原生任务步数覆盖机制把本次探针限制为2步，仅改变探针进程内设置；原400步文件、已完成80次评测及54块推理诊断、12条闭环全部保留。两步结束打印的Fail不是策略失败统计。后续场景经历的前序动作数也不同于完整轨迹，因此不能声称复制了全部历史运行上下文。</p><p>记录器不增加规划、get_obs、渲染或物理步调用，保持规划器返回对象与随机数状态。保存动作前后实测qpos/qvel、根位姿、末端链接位姿、两个瓶子的位姿，以及左右臂规划器的原始参数、状态、位置和速度轨迹。9项CPU检查通过；单张空闲GPU，硬限10分钟且未超过原23:45截止时间。</p></section>
<section id="results"><h2>全部三对、两个动作</h2>{table(['场景','动作索引','执行前记录全同','规划输入','规划输出','执行后实测状态','执行后qpos最大差','驱动目标','动作前图像'],rows)}<p>索引从0开始；动作前图像1即首条动作后的下一次控制观测。qpos差为原生关节坐标数值，不统一换算成米或角度；完整dtype与数组在原始记录中。即使qpos差为0，速度或根位姿仍可能不同。本配置左右臂共享同一个双臂articulation，left/right的qpos字段是整机状态的重复读取，不作为独立样本。</p><h3>分别检查左右臂规划器</h3>{table(['场景','动作','臂','输入','两轮规划状态','完整返回值','位置轨迹最大差','速度轨迹最大差'],planner_rows)}<p>第二条动作的规划输入可能已经承接第一条执行后的状态差异，不能把不同输入下的输出差异称为同输入规划随机性。</p></section>
<section id="images"><h2>首动作后的观测</h2>{images}<p>六个原生两帧视频仅用于记录完整性，长度约0.2秒：{video_links}。它们不是完整任务展示。</p></section>
<section id="next"><h2>对表征研究意味着什么</h2><p>这个检查不检验RAE与VAE，也不评估新适配器；它检验评测路径本身的重复性。应把特征恢复误差、动作差异与任务收益分别报告，并在表征对照中量化重复运行的基线波动。</p><p>本探针移除了策略服务器和在线推理，缩短了前序场景，且增加了被动读取；这些运行上下文与完整闭环不同。第一场景的两步请求与旧round0一致，后两个场景只在初始请求一致。下一步应先设计匹配运行上下文的对照，再检查最早差异所在边界，不能把没有复现解释为排除了物理或规划因素。若相同规划轨迹后状态变化，优先检查未记录的初始速度、接触／求解器状态与仿真设置；若规划先分歧，先检查规划器上下文。当前没有修改求解器设置或追加表征训练。瓶子速度和隐藏求解器状态未记录，不能下具体根因结论。</p></section>
<section id="artifacts"><h2>全部复现材料</h2><p>{' · '.join(link(n)for n in files if(A/n).exists())}</p><p>{link('all-traces-and-requests.tar.gz','全部数值、规划路径和请求')} · {link('reproduction-scripts.tar.gz','冻结脚本')} · {link('video-manifest.json','六段短视频SHA256')}</p></section>'''
old=(SITE/'norm.html').read_text();style=''.join(re.findall(r'<link[^>]*rel="stylesheet"[^>]*>',old))+''.join(re.findall(r'<style.*?</style>',old,re.S))
nav=''.join(f'<a href="#{k}">{v}</a>'for k,v in[('status','结论'),('design','固定方案'),('results','全部结果'),('images','观测'),('next','下一步'),('artifacts','复现材料')])
(SITE/'attribution.html').write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 规划与物理执行诊断</title>{style}</head><body><header><a href="index.html">Embodied Research Notebook</a><span>固定动作诊断 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / ATTRIBUTION</p><nav aria-label="报告目录">{nav}</nav></aside><main>{body}</main></div></body></html>')
for name in ['index.html','trajectory.html']:
 p=SITE/name;t=p.read_text();callout=f'<div class="callout" id="attribution-latest"><strong>新增：六段两步固定动作诊断已完成。</strong>{headline}。<a href="attribution.html">规划路径、实测状态和全部配对记录</a>。</div>'
 if'id="attribution-latest"'in t:t=re.sub(r'<div class="callout" id="attribution-latest">.*?</div>',callout,t,flags=re.S)
 else:t=re.sub(r'(<main\b[^>]*>)',lambda m:m.group(1)+callout,t,count=1)
 p.write_text(t)
print(json.dumps({'traces':6,'commands':12,'policy_calls':0,'initial_equal_pairs':initial_equal,'equal_plan_pairs':paths_equal,'post_state_different_pairs':post_different,'headline':headline},ensure_ascii=False))
