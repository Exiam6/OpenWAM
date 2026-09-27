#!/usr/bin/env python3
"""Render all twelve frozen trajectory repeats without outcome filtering."""
import base64,gzip,hashlib,html,json,re,shutil,tarfile,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
RUN=Path('/data02/zifanz4/openwam-experiments/results/repeatability-20260921/closedloop')
SITE=Path('/home/zifanz4/research-reports/dist');A=SITE/'assets/trajectory-20260921'
s=json.loads((RUN/'summary.json').read_text());assert s['all_integrity_checks_passed'] and s['episodes']==12
assert (RUN/'exit-code.txt').read_text().strip()=='0'
A.mkdir(parents=True,exist_ok=True)
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
def link(name,label=None):return f'<a href="assets/trajectory-20260921/{name}">{html.escape(label or name)}</a>'
files=['protocol.md','source-freeze.json','launch.json','cpu-checks.json','summary.json','supplemental-audit.json','first-divergence-inspection.json','measurement-semantics.md','next-diagnostic-draft.md','exit-code.txt']
for name in files:
 if(RUN/name).exists():shutil.copyfile(RUN/name,A/name)
for p in RUN.glob('round*-*.json'):shutil.copyfile(p,A/p.name)
for p in RUN.glob('round*-effective-config.yaml'):shutil.copyfile(p,A/p.name)
shutil.copyfile(RUN.parent/'scenes.json',A/'scenes.json')
video_rows=[];trace_paths=[];video_manifest=[]
for group,data in s['groups'].items():
 d=RUN/group;(A/group).mkdir(exist_ok=True)
 shutil.copyfile(d/'summary.json',A/group/'summary.json')
 videos=sorted((d/'runtime/eval_result').rglob('episode*.mp4'),key=lambda p:int(re.search(r'episode(\d+)',p.name).group(1)))
 assert len(videos)==3
 for row,video in zip(data['records'],videos):
  seed=row['seed'];name=f'{group}/seed-{seed}.mp4';shutil.copyfile(video,A/name)
  trace=d/f'seed-{seed}-trace.jsonl';shutil.copyfile(trace,A/group/trace.name);trace_paths.append((trace,group+'/'+trace.name))
  video_manifest.append({'group':group,'seed':seed,'source':str(video),'public_file':name,'sha256':hashlib.sha256(video.read_bytes()).hexdigest()})
  video_rows.append((group,seed,row['success'],row['actions'],name))
(A/'video-manifest.json').write_text(json.dumps(video_manifest,indent=2)+'\n')
with tarfile.open(A/'lightweight-traces.tar.gz','w:gz')as t:
 for path,name in trace_paths:t.add(path,arcname=name)
 t.add(RUN/'summary.json',arcname='summary.json');t.add(RUN/'source-freeze.json',arcname='source-freeze.json')
with tarfile.open(A/'reproduction-scripts.tar.gz','w:gz')as t:
 for name in ['audit_trajectory_results.py','inspect_trajectory_divergence.py']:t.add(ROOT/name,arcname=name)
 t.add(ROOT/'TRAJECTORY_PROTOCOL.md',arcname='TRAJECTORY_PROTOCOL.md')
 for name in ['trajectory_trace.py','trajectory_client.py','serve_trajectory.py','check_trajectory_trace.py','summarize_trajectories.py','run-trajectory-study.sh','norm_client.py','norm_common.py','robustness_client.py','robustness_common.py','repeat_common.py','capture_repeat_inputs.py']:
  t.add(ROOT/'scripts'/name,arcname='scripts/'+name)
 for name in ['protocol.md','source-freeze.json','launch.json','cpu-checks.json']:t.add(RUN/name,arcname='results/repeatability-20260921/closedloop/'+name)
 t.add(RUN.parent/'scenes.json',arcname='results/repeatability-20260921/scenes.json')
comparisons=s['paired_trajectories'];exact=sum(p['all_recorded_values_identical']for p in comparisons)
display=lambda v:'未出现'if v is None else str(v)
rows=[];examples=[]
for pair in comparisons:
 first=pair['first_different_step'];condition=pair['condition'];seed=pair['seed'];label=('干净'if condition=='clean'else'噪声')+f' / {seed}'
 rows.append([label,' / '.join(map(str,pair['lengths'])),' / '.join('成功'if x else'失败'for x in pair['successes'])]+[display(first[k])for k in ['raw_images','eef_state','joint_state','request','server_action','env_target']])
 steps=sorted({v for k,v in first.items()if v is not None and k!='success_flag'})
 if not steps:steps=[0]
 for step in steps:
  for rid in [0,1]:
   group=f'r{rid}-{condition}';src=RUN/group/'requests'/f'seed-{seed}'/f'step-{step:04d}.json.gz';payload=json.loads(gzip.decompress(src.read_bytes()))
   name=f'examples/{condition}-{seed}-step{step}-r{rid}';(A/'examples').mkdir(exist_ok=True)
   shutil.copyfile(src,A/(name+'.json.gz'))
   (A/(name+'.png')).write_bytes(base64.b64decode(payload['images']['head_camera']))
  examples.append(f'<details><summary>{label} · step {step} · 两轮请求</summary><div style="display:flex;gap:16px;flex-wrap:wrap">'+''.join(f'<figure style="margin:0;max-width:46%"><img loading="lazy" src="assets/trajectory-20260921/examples/{condition}-{seed}-step{step}-r{rid}.png" alt="{label} step{step} round{rid}" style="width:100%"><figcaption>Round{rid} · '+link(f'examples/{condition}-{seed}-step{step}-r{rid}.json.gz','完整请求JSON.gz')+'</figcaption></figure>'for rid in [0,1])+'</div></details>')
fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
keys=['raw_images','eef_state','joint_state','request','server_action','env_target'];markers=['o','s','^','D','x','+'];colors=['#789f86','#b69761','#a3819f','#448384','#c07053','#284a3c']
for j,k in enumerate(keys):
 vals=[(i,p['first_different_step'][k])for i,p in enumerate(comparisons)if p['first_different_step'][k]is not None]
 if vals:ax.scatter([v for i,v in vals],[i+(j-2.5)*.09 for i,v in vals],marker=markers[j],color=colors[j],label=k,s=48)
ax.set_yticks(range(6),[f"{p['condition']} / {p['seed']}"for p in comparisons]);ax.set_xlabel('First differing action index (zero-based)');ax.set_title('Two original-policy repeats: first observed divergence');ax.spines[['top','right']].set_visible(False)
if any(any(v is not None for v in p['first_different_step'].values())for p in comparisons):ax.legend(fontsize=8,ncol=3)
else:ax.text(.5,.5,'No recorded divergence in these six pairs',transform=ax.transAxes,ha='center');ax.set_xlim(0,1)
fig.savefig(A/'first-divergence.svg');fig.savefig(A/'first-divergence.png',dpi=180);plt.close(fig)
for p in A.glob('*.svg'):p.write_text('\n'.join(x.rstrip()for x in p.read_text().splitlines())+'\n')
initial_exact=sum(x['exact']for p in comparisons for x in p['first_action_vs_fixed_request_diagnostic'])
inspection=json.loads((RUN/'first-divergence-inspection.json').read_text())
precision_rows=[]
for pair in inspection['pairs']:
 d=pair['first_step_numeric_differences']
 if not d:continue
 cam=pair['policy_camera_pixel_differences_at_first_difference']['head_camera']
 precision_rows.append([pair['condition']+' / '+str(pair['seed']),pair['first_recorded_difference'],f"{d['eef']['groups']['xyz']['max_abs']*1e6:.3f}",f"{cam['changed_pixel_fraction']*100:.1f}%",cam['max_byte_difference']])
precision_html=table(['条件 / 场景','首差索引','EEF位置最大差（微米）','策略主图变化像素比例','主图最大通道差（0–255）'],precision_rows)
headline=f'{exact}/6 对轨迹的全部已记录值一致'
interpretation=('本次两次独立服务器运行中，六对轨迹的相机、状态、请求、输出动作、执行目标及长度逐项一致。这个有限检查支持当前配置在这些场景上的重复性；不能据此证明整个模拟器确定性，也未解释此前场景300003的历史变化。'if exact==6 else '完整轨迹存在分歧。下表保留全部六对比较，按首次变化顺序区分观测、请求和动作；这还不能单独定位物理仿真、渲染、逆运动学或策略内部的原因。')
obs_first=sum(p['first_different_step']['request'] is not None and p['first_different_step']['server_action'] is not None and p['first_different_step']['request']<p['first_different_step']['server_action'] for p in comparisons)
same_outcomes=sum(p['successes'][0]==p['successes'][1]for p in comparisons)
if obs_first:
 interpretation=f'六对中{obs_first}对先出现观测请求差异，随后才出现动作差异；{same_outcomes}/6对的成功／失败结果相同。'+interpretation
 if obs_first==6:headline='6/6对先出现观测差异，再出现动作差异'
video_html='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px">'+''.join(f'<figure style="margin:0"><video controls preload="none" playsinline style="width:100%" src="assets/trajectory-20260921/{name}"></video><figcaption>{group} · seed {seed} · '+('成功'if success else'失败')+f' · {actions} actions</figcaption></figure>'for group,seed,success,actions,name in video_rows)+'</div>'
body=f'''<div class="hero"><div class="eyebrow">Closed-loop repeatability / 2026.09.21</div><h1>分歧最早出现<br><em>在哪一步？</em></h1><p class="lead">原策略，三场景，两种输入条件，两轮独立服务器。全部12条闭环与每步记录，不进行表征干预或挑选成功案例。</p></div>
<section id="status"><h2>{headline}</h2><div class="callout">{interpretation}</div><p>12/12条轨迹完成；来源、初始请求、逐步计数及数值哈希检查通过。共{s['requests']}次请求。更新于{time.strftime('%Y-%m-%d %H:%M:%S %Z')}。</p><p>此前<a href="repeat.html">54块固定输入诊断</a>已经完成；这里单独检查闭环。<a href="norm.html">80次表征对照评测</a>没有重跑。</p></section>
<section id="design"><h2>固定方案与范围</h2><p>场景400000、400001、400002，均来自已有清单前三项，独立于成功结果选择。每轮依次干净、主相机噪声σ=.20，各3条。轮间重新加载原模型，场景间使用原客户端reset，保持十选一指令随机数消耗、seed42、BF16、10次去噪、关闭编译、原缓存与完整32步动作缓冲。</p><p>记录器不改传入值或返回对象，9项CPU检查验证实际接口与上游循环，包括400步限制。每步保存三相机哈希、EEF向量和关节驱动目标、完整请求、20维服务器动作及传给take_action的16维目标；后者是命令，不是测量的物理轨迹。原joint_action.vector记录关节驱动目标，而非实测关节位置；EEF中的夹爪分量也是命令值，不能用作接触标签。此语义澄清来自固定版本源码，实验与记录均未改动。只用空闲GPU2，遵守原90分钟总预算，未终止其他任务。</p><p>两轮的12个初始完整请求均匹配此前保存输入；其中{initial_exact}/12个首动作与固定请求诊断完全一致。首动作匹配不是整条轨迹匹配的替代证据。</p></section>
<section id="results"><h2>全部六对轨迹</h2>{table(['条件 / 场景','两轮动作数','两轮结果','原图首差','EEF首差','驱动目标首差','请求首差','20D动作首差','16D目标首差'],rows)}<p>索引从0开始。未出现表示共享前缀内没有差异；长度不同仍判为整轨迹不一致。数值采用保存dtype逐字节比较，不能把极小数值差异直接当作控制失败。</p><img src="assets/trajectory-20260921/first-divergence.svg" alt="全部六对轨迹的第一处观测和动作分歧" style="width:100%"><h3>第一处差异的大小</h3>{precision_html}<p>EEF位置差只比较xyz；夹爪命令与旋转参数不混入长度单位。变化像素比例表示任一RGB通道不同，不代表语义信息受损。</p><p>若观测先于动作改变，它支持先排查执行与观测路径；若相同请求下动作先变，仍需核对32步缓冲和生成上下文。仅凭顺序不能认定具体模块有缺陷。</p></section>
<section id="examples"><h2>第一处分歧的请求</h2><p>逐对保留每个首次变化索引的两轮完整请求；完全相同的对展示初始输入。完整逐步请求压缩包留在本地（{s['compressed_request_bytes']/1024**2:.1f}MiB）；轻量逐步数值与哈希全部公开。</p>{''.join(examples)}</section>
<section id="videos"><h2>全部12条轨迹视频</h2>{video_html}</section>
<section id="next"><h2>解释与下一步</h2><p>这是一项重复性诊断，使用三个已见场景和两次重复，不能估计泛化成功率，不能证明RAE优于VAE，也不能验证接触阶段鲁棒性。下一步准备仅重放前两条固定命令的小诊断，分别记录规划器返回路径和实测关节状态，区分规划与物理执行。方案尚未启动，需另行固定预算并通过记录器检查；当前不追加适配器训练或参数搜索。</p><p>表征改动能够改变动作已有证据，但“改变动作”与“提高控制成功率”是两个问题。先建立可信的对照评测，再用独立新场景验证小改进。</p></section>
<section id="artifacts"><h2>可复现材料</h2><p>{' · '.join(link(n)for n in files if(A/n).exists())}</p><p>{link('scenes.json','固定场景')} · {link('lightweight-traces.tar.gz','全部逐步数值与哈希')} · {link('reproduction-scripts.tar.gz','冻结脚本')} · {link('video-manifest.json','视频SHA256')} · {link('first-divergence.png','科学图PNG')}</p></section>'''
old=(SITE/'norm.html').read_text();style=''.join(re.findall(r'<link[^>]*rel="stylesheet"[^>]*>',old))+''.join(re.findall(r'<style.*?</style>',old,re.S))
nav=''.join(f'<a href="#{k}">{v}</a>'for k,v in [('status','结论'),('design','固定方案'),('results','全部比较'),('examples','请求示例'),('videos','全部视频'),('next','下一步'),('artifacts','复现材料')])
(SITE/'trajectory.html').write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · 闭环轨迹重复性</title>{style}</head><body><header><a href="index.html">Embodied Research Notebook</a><span>轨迹重复性 · 2026.09.21</span></header><div class="layout"><aside><p>OPENWAM / TRAJECTORIES</p><nav aria-label="报告目录">{nav}</nav></aside><main>{body}</main></div></body></html>')
for name in ['index.html','repeat.html']:
 p=SITE/name;text=p.read_text();callout=f'<div class="callout" id="trajectory-latest"><strong>12条闭环重复性诊断已完成：</strong>{headline}。<a href="trajectory.html">逐步对照、全部视频与记录</a>。</div>'
 if'id="trajectory-latest"'in text:text=re.sub(r'<div class="callout" id="trajectory-latest">.*?</div>',callout,text,flags=re.S)
 else:text=re.sub(r'(<main\b[^>]*>)',lambda m:m.group(1)+callout,text,count=1)
 if name=='index.html':
  text=text.replace('研究提案 · 更新于 2026.09.20','研究记录 · 更新于 2026.09.21')
  text=text.replace('状态：首轮离线实验已完成<br>闭环策略：尚未评测','状态：闭环重复性诊断已完成<br>12条轨迹与逐步记录')
  if 'href="trajectory.html"'not in text.split('</nav>')[0]:text=text.replace('<nav aria-label="报告目录">','<nav aria-label="报告目录"><a href="trajectory.html">最新：闭环重复性</a><a href="repeat.html">固定输入动作诊断</a><a href="norm.html">80次归一化对照</a>',1)
 p.write_text(text)
print(json.dumps({'episodes':12,'requests':s['requests'],'exact_pairs':exact,'video_count':len(video_manifest),'page':'trajectory.html'}))
