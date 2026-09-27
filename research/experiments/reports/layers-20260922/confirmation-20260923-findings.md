# New independent representation confirmation — completed, numerical audit pending

The frozen scorer completed 2026-09-23T12:20:42.185146-05:00, exit0, inside the unchanged
15:38:51CDT deadline. Three collectors ended12:11:44/12:14:25/12:20:06, all exit0.
60expert-feasible scenes:20per task,78consecutive attempts (20/22/36),18expert-infeasible
attempts retained.12fixed state frames/scene,5040clean/noisy feature rows. No readout
refit, heldout recalibration, winner change, old80rerun or expired56/60 reuse.

## Stored primary outcomes — not yet independently numerically audited

PCA48 versus native Wan48, same192pooled visual dimensions and frozen readouts
trained using matched noise augmentation. Primary condition sigma0.10. Paired
scene bootstrap2000draws,20scenes/task, seed20260922; repeated frames/noise/SVAEseeds
are not independent observations.

| Endpoint | Relative change | Paired95% interval |
|---|---:|---:|
| Initial first-close XY | -73.05% | [-79.60, -65.72]% |
| Short-state E | -39.11% | [-48.70, -27.30]% |
| Translation component | -40.13% | [-49.87, -27.76]% |
| Gripper component | -12.33% | [-26.12, 5.91]% |

The frozen scorer flags both co-primary absolute-difference intervals below0.
This is a stored-run result, not a completed independent audit. Gripper's interval
crosses0: no gripper improvement or noninferiority claim. Alltask goal/state point
changes favorPCA, butstate change varies substantially (-9.96/-58.74/-42.70%).
RawDINO grippererror increases79.65%
[42.65,126.73]%, despite better totalstateE.
Both compact routes retain that tradeoff information; no uniform bestrepresentation
or causal pretraining explanation is established.

## Full matrix from saved outputs


### goal

| Route | Clean E | Noise.04 E | Noise.10 E |
|---|---:|---:|---:|
| Wan48_native_scale | 0.365753 | 0.275882 | 0.349102 |
| Wan48_per_token_layernorm | 0.320602 | 0.290487 | 0.382935 |
| DINO_raw768 | 0.103898 | 0.093238 | 0.105863 |
| DINO_PCA48 | 0.090232 | 0.079935 | 0.094070 |
| DINO_SVAE48_seed_mean | 0.097108 | 0.086187 | 0.097396 |

### state

| Route | Clean E | Noise.04 E | Noise.10 E |
|---|---:|---:|---:|
| Wan48_native_scale | 0.322095 | 0.307213 | 0.308760 |
| Wan48_per_token_layernorm | 0.332222 | 0.309250 | 0.307575 |
| DINO_raw768 | 0.141406 | 0.129641 | 0.142973 |
| DINO_PCA48 | 0.178877 | 0.165821 | 0.187994 |
| DINO_SVAE48_seed_mean | 0.180951 | 0.166820 | 0.185817 |
| proprio | 0.331021 | 0.331021 | 0.331021 |

## Provenance and remaining checks

Pipeline identity gate passed:60episodes/720frames,0recordedpriorencodedhashoverlap.
All370frozen input/code hashes were verified unchanged after the window. This is
byte integrity, not numerical validation. Independent audit.py exists but was not
executed beforedeadline; no audit.json exists. Do not remove its deadline guard or
silently relabel this as an audited confirmation. Toolautomaticapproval failed due
to usage exhaustion during the12:08report update; the action didnotexecute. Tools
becameavailable at15:49. Backgroundboundedexperiments completed normally at12:20.

L40S trainingfixture: WanRMS.10603failedfixed.05tolerance; DINOexact. A40dispatch
onrainier initially failedbeforePython becausehostglibc2.31wasincompatible. Original
Python/package/weights were retained; aprivatecm001glibc2.35loader/librarybundle
allowedstartup, without changing systemlibraries. A40trainingparity thenpassed:
Wanexact, DINOmaxRMS.045968andmaxcoordinate.224641withinoriginal.05/.5bounds.
Both failures retained, zero newscene inference beforeparitypass. Only3collectors+
1featureGPU concurrent. No studyGPUworkers remain at15:49.

The implication, if the independent audit confirms these saved outputs, is that
compact pretrained visual features are a useful low-cost offline control-readout
baseline against pixelVAE latents under the matched training/noise protocol. It is
not fullRAE (no trainedpixeldecoder), not closedloopcontrol, not truecontact, not
unseen-task or unseen-noise generalization. Architecture/objective/pretraining
remainconfounded. RawDINO has3072pooledinputs versus192forcompact/Wan controls.

Next: authorize a bounded CPUartifact audit, then contribute the smallest useful
reproducible baseline comparison with train-only normalization, scene provenance,
and separate translation/gripper reporting. Do not change the defaultpolicy or
claim RAE/policygain on the basis of these offline metrics.
