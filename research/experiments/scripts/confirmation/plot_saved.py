"""Render stored contrasts only; does not compute or change statistical results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data02/zifanz4/openwam-experiments/results/confirmation-20260923');O=Path('/home/zifanz4/openwam-experiments/reports/layers-20260922');q=json.loads((R/'result.json').read_text())
items=[('PCA48: goal XY',q['co_primary']['goal']),('PCA48: state E',q['co_primary']['state']),('PCA48: gripper',q['co_primary']['state']['gripper']),('Raw DINO: gripper',q['paired']['state']['noise0.10']['Wan48_native_scale']['DINO_raw768']['gripper'])]
fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
for y,(label,d) in enumerate(items):
 x=d['relative_change_percent'];lo,hi=d['paired95'];color='#245a86' if hi<0 else ('#af462d' if lo>0 else '#74634b')
 ax.errorbar(x,3-y,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color=color,linewidth=2)
 ax.text(143,3-y,f'{x:+.1f}%  [{lo:+.1f}, {hi:+.1f}]',va='center',fontsize=9)
ax.axvline(0,color='#777777',linestyle='--',linewidth=1);ax.set_yticks(range(4),[x[0] for x in items[::-1]]);ax.set_xlim(-90,242);ax.set_xticks([-80,-40,0,40,80,120]);ax.set_xlabel('Relative normalized error change vs native Wan48 (%) — lower is better');ax.grid(axis='x',alpha=.2)
ax.set_title('Fresh 60-scene confirmation • noise σ=0.10\nStored scorer outputs; independent numerical audit pending',loc='left',fontsize=12)
for spine in ['top','right']:ax.spines[spine].set_visible(False)
fig.savefig(O/'confirmation-20260923-intervals.png',dpi=180);fig.savefig(O/'confirmation-20260923-intervals.pdf');plt.close(fig)
print('Rendered stored intervals; no rescoring or new confidence-interval computation')
