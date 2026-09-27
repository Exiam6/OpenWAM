# Planned within-camera baseline completion —2026-09-22 14:33CDT

This fills the previously disclosed missing wrist-trained baseline from the
original12hour plan. It is secondary to the frozen head-input primary study.
No head readout, compressor, primary criterion, source winner or fresh manifest
changes. All primary files remain hashed and unchanged. No fresh outcomes have
been scored. Full RAE/VAE and physical-contact comparisons remain out of scope.

Use only existing35training episodes/task and their exact12current-frame indices,
left_camera images resized and encoded exactly as the fixed fresh left_transfer
path. Same9image repeat preserves backbone batch geometry; use firstlatentonly.
Same3fixed sources and15source/compressor/seed paths, no compressor refit. Image
hash,task/episode/frame,currentstate/futurelabel metadata are retained.

Fit separate task-specific wrist-trained readouts: grouped ridge, original fixed
4alphas and training-episode5foldCV, eachfold independent normalization; fixed
MLP128x128ReLU,500steps,batch64,seed42,AdamW1e-3/wd1e-4, groupbalancedloss. Identical
budget to original head-trained readouts. Proprio-only original readouts reused,
because that input is camera independent; do not retrain unchanged baselines.
All45visualMLPs and45paired-group ridge models kept; no development/test model
selection. Save final weights/scalers/CV metadata/checksums before fresh scoring.

Compare on the SAME newly collected60episodes: head-trained/headinput,
head-trained/wristinput, wrist-trained/wristinput. The last two isolate readout
training-camera mismatch conditional on the SAME head-trained compressors. They
do not isolate camera geometry/occlusion, and do not test wrist-trained reducers.
Report all sources/seeds/tasks and proprio baseline. No camera comparison can
replace or rescue the primary noise hypothesis. This completion was fixed after
aggregate head-development outcomes, not described as initial preregistration.

One idle healthy GPU,6CPUthreads,max30minutes including extraction/fitting,
within original4GPU/48GPUh/12h/80GB caps. No new simulator data. If unavailable or
failed, keep original missing-baseline limitation, preserve failure, no retries
selected by performance. Main fresh confirmation is independent of this stage.
