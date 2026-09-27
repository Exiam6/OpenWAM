# Secondary initial-scene waypoint probe — frozen before new fitting/scoring

Motivation from development only: proprio-only MLP outperforms visual+proprio on
short-horizon achieved-state labels. Training-only audit of105episodes finds
exactly identical initial16Dproprio within eachtask, while the first downward
native-gripper0.5-crossing endpose varies across episodes. A second proxy can
ask whether visual features predict scene-dependent future expert waypoints.
It does not replace the original K4-SVAE noise hypothesis or rescue a failure.

Definition fixed now: input ONLY head image at rawframe0; target XY of the arm's
native endpose at the EARLIEST downward crossing fromgripper>0.5 to<=0.5. If both
cross at the same frame, choose left (fixed tie rule and report tiecount). This
is an expert event position, NOT a contact/object-pose/controller-command label.
XY chosen because the training audit shows tabletop planar variation and tinyZ
variation; this choice is explicit, before fitting this probe. If any fresh
accepted episode lacks an event, report missingness and do not replace it or
claim a complete60episode confirmatory result. Training audit found105/105 valid.

Reuse fixed15source/compressor/seed paths; NO encoder or compressor refit.
Readouts trained only on the35existing training episodes/task, one initialframe
per episode; no intermediate frames, development labels or new episode labels
enter fitting. Constant task-specific training target mean is the baseline;
visual readouts receive no proprio, preventing tiny numerical state drift from
acting as a proxy. Per-task target normalization is train-only.

Ridge uses the existing fouralphas1e-4/.01/1/100, same training-episode5foldCV,
fold-specific standardization, identical budget across sources. MLP has two128
ReLUhiddenlayers,2outputs,500steps,batch64,seed42,AdamW1e-3/wd1e-4; no tuning,
early stopping or checkpoint selection. All45ridge+45MLPreadouts retained.
Constant predictor is not fitted against fresh outcomes. A small105-example
training set limits the conclusions and readout-seed variance is not measured.

Evaluate once using the existing yet-unscored fixed60fresh episodes and their
initial-frame vectors: clean and sigma.10/.04(3paired copies each). This is a
prospectively fixed SECONDARY diagnostic added AFTER head-development outcomes;
it is not the initial preregistered primary, not a new independent task set,
and not an independent replication after choosing a favorable source. Preserve
all3sources/raw/PCA/SVAE and allseed outcomes. Keep K4SVAEvsL12SVAE contrast, report
train-mean baseline and all per-task XYerrors; do not silently promote a winner.
If confidence intervals are shown, they are descriptive paired2000task-stratified
episode intervals for thissecondary question; no new policy-gain declaration.
Primary metric mean standardized XYsquared error, equal task weighting; also
report XYRMSE in native units. Raw768 readout dimension differs from48 as before.

Single healthy idle GPU,6CPUthreads,10min hardcap; within existing4card/48GPUh,
80GBcache and original01:15:12CDT studydeadline. Fresh data collection unchanged.
Weights/scalers/labels/sourcehashes must precede fresh scoring; if not, mark
secondary diagnostic exploratory or incomplete rather than alter timestamps.
