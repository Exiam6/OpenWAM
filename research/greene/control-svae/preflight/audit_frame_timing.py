"""Read-only audit: physical time spanned by the planned t+4 recorded-frame label horizon.

RoboTwin (envs/_base_task.py, timestep 1/250 s, save_freq 15) saves a frame before a motion
segment's first step, after step 0, every 15 steps, and again at segment end, so recorded
frames are not uniformly spaced: segment end and next segment start share a sim instant.
Without per-frame sim-step timestamps in the hdf5, the exact span is unrecoverable; this
bounds it: identical consecutive joint vectors => zero-motion pair (boundary duplicate or hold).
"""
import glob,hashlib,json,datetime,sys
import h5py,numpy as np
T='/scratch/zz4330/openwam-runtime/endtoend-root/results/layers-20260922/native-preflight/train-data'
fs=sorted(glob.glob(T+'/*/aloha-agilex_clean_50/data/episode*.hdf5'));assert len(fs)==105
rows=[];pairs=ident=win=win_dup=win_all_static=0
for f in fs:
 with h5py.File(f,'r') as h:q=h['joint_action/vector'][:]
 same=np.abs(np.diff(q,axis=0)).max(1)<1e-9;w=np.lib.stride_tricks.sliding_window_view(same,4)
 pairs+=len(same);ident+=int(same.sum());win+=len(w);win_dup+=int(w.any(1).sum());win_all_static+=int(w.all(1).sum())
 rows.append({'path':f,'frames':len(q),'identical_pairs':int(same.sum()),'t4_windows':len(w),'t4_windows_with_identical_pair':int(w.any(1).sum())})
out={'time':datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'scope':'read-only training-data timing audit (105 episodes); no fitting, no heldout data',
 'robotwin':{'timestep_s':1/250,'save_freq_steps':15,'nominal_frame_s':0.06,'nominal_t4_s':0.24,'source':'benchmarks/RoboTwin/envs/_base_task.py:223,817-882; task_config/demo_clean.yml'},
 'adjacent_pairs':pairs,'identical_joint_vector_pairs':ident,'identical_fraction':round(ident/pairs,4),
 't4_windows':win,'t4_windows_with_identical_pair':win_dup,'t4_windows_fraction_irregular_lower_bound':round(win_dup/win,4),'t4_windows_fully_static':win_all_static,
 'physical_horizon_verified':False,
 'conclusion':'t+4 recorded frames is NOT a fixed 0.24 s: it spans <= 0.24 s and less whenever a window crosses a segment boundary (identical frames) or a post-step-0 frame (0.004 s). Exact per-window span needs sim-step timestamps (instrumented replay).',
 'rows':rows}
p=sys.argv[1];open(p,'w').write(json.dumps(out,indent=1)+'\n')
print({k:v for k,v in out.items() if k not in('rows','robotwin')})
