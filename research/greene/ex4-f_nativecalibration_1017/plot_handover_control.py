"""Post-hoc six-checkpoint view of one matched, completed handover control scene."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
SCENE=6201000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trace(p):return [json.loads(x) for x in p.read_text().splitlines()]

def main():
    out=ROOT/'handover-first-scene';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13,13));grid=fig.add_gridspec(6,3,width_ratios=[1.15,1.4,1.],hspace=.55,wspace=.3)
    data=[]
    for i,(variant,seed) in enumerate((v,s) for v in ['raw768_native','svae48_native'] for s in [42,43,44]):
        cell=f'{variant}-seed{seed}';conditions={}
        for condition in ['old','fixed']:
            d=ROOT/'evaluation'/cell/condition/'handover_block'/f'seed-{SCENE}'
            result=json.loads((d/'result.json').read_text());assert result['status']=='complete'
            records=trace(d/'trace.jsonl');action=np.array([x['action'] for x in records]);state=np.array([x['proprio'] for x in records])
            stats={}
            for name,start,grip in [('left',0,9),('right',10,19)]:
                errors=np.linalg.norm(state[1:,start:start+3]-action[:-1,start:start+3],axis=1)*1000
                stats[name]=dict(median_position_error_mm=float(np.median(errors)),min_gripper_command=float(action[:,grip].min()),closed_command_steps=int((action[:,grip]<.5).sum()))
            conditions[condition]=dict(directory=str(d),result=result,stats=stats,result_sha256=sha(d/'result.json'),trace_sha256=sha(d/'trace.jsonl'))
            if condition=='fixed':fixed_action=action;fixed_records=records
        assert conditions['old']['result']['initial_state']==conditions['fixed']['result']['initial_state']
        d=Path(conditions['fixed']['directory']);frames=sorted((d/'frames').glob('*-head_camera.jpg'));image_path=frames[-1]
        ax=fig.add_subplot(grid[i,0]);ax.imshow(Image.open(image_path));ax.axis('off');ax.set_title(f"{'Raw768' if variant=='raw768_native' else 'Compact48'} / seed {seed}",loc='left',fontsize=11)
        ax=fig.add_subplot(grid[i,1]);ax.plot(fixed_action[:,9],label='Left',lw=1,color='#0072B2');ax.plot(fixed_action[:,19],label='Right',lw=1,color='#D55E00')
        ax.axhline(.5,ls='--',color='grey',lw=.8);ax.set_ylim(-.05,1.08);ax.set_ylabel('Gripper command');ax.set_xlim(0,len(fixed_action)-1)
        if i==0:ax.legend(loc='lower right',ncol=2,frameon=False)
        if i==5:ax.set_xlabel('Action index (fixed executor)')
        ax=fig.add_subplot(grid[i,2]);x=np.arange(2)
        for offset,condition,color,label in [(-.18,'old','#999999','Old'),(.18,'fixed','#009E73','Fixed')]:
            values=[conditions[condition]['stats'][side]['median_position_error_mm'] for side in ['left','right']]
            ax.bar(x+offset,values,.36,color=color,label=label)
        ax.set_xticks(x,['Left','Right']);ax.set_ylabel('Position error (mm)');ax.set_ylim(0,15)
        if i==0:ax.legend(frameon=False,fontsize=8)
        data.append(dict(cell=cell,scene_seed=SCENE,conditions=conditions,final_head_frame=str(image_path),final_head_frame_sha256=sha(image_path)))
    fig.suptitle('On one matched handover scene, all six policies keep both grippers open',fontsize=15,y=.99)
    fig.text(.5,.018,'Final native head view | corrected gripper commands | median target-to-next-observation position error\nPost-hoc description of scene6201000; old/fixed trajectories diverge. Native wrist views and complete clips are in the evidence bundle.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.95,bottom=.07)
    for ext in ['png','pdf']:fig.savefig(out/f'control-observation.{ext}',dpi=160)
    plt.close(fig)
    assert all(x['conditions']['fixed']['stats'][side]['closed_command_steps']==0 for x in data for side in ['left','right'])
    (out/'data.json').write_text(json.dumps(dict(scope='Post-hoc one-scene description across six checkpoints. Not a new hypothesis test, full pilot verdict, or population estimate.',rows=data),indent=2)+'\n')
    print(json.dumps(dict(path=str(out),cells=len(data),closed_command_steps=0)))

if __name__=='__main__':main()
