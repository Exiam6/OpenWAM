# RAE comparison scope — 2026-09-22T23:28:11.609894-05:00

User steering: “不用rae吗”. Restore direct RAE/representation-versus-pixel-VAE
comparison as the next research priority. The earlier goal-auxiliary proposal
compared losses within DINOv3+SVAE48; it did not answer whether RAE is better.
This is a scope note, NOT a frozen launch protocol or a claim of a started run.
Existing protocols/results and original deadline2026-09-23T01:15:12-05:00remain
unchanged. The expired56/60cohort remains incomplete and unscored.

## Verified distinctions

Official RAE uses a frozen pretrained representation encoder with a trained
pixel decoder. Decoder-only training does not change the frozen encoder features.
Source: https://github.com/bytetriper/RAE and https://arxiv.org/abs/2510.11690.
OpenWAM's project page identifies frozen Wan2.2 VAE as the released alpha encoder:
https://openwam-official.github.io/ . Local DinoV3VideoEncoder docstring explicitly
sets pixel_decode=False: it is not a complete pixel-reconstructing RAE by itself.
The existing DINOv3 B/16 feature study compared raw768/PCA48/SVAE48; raw-feature
results already exist and must not be repeated or relabeled as a new full RAE run.
SVAE reconstructs semantic features, not pixels. Setting its KL coefficient to
zero would be an AE ablation, not by itself an implementation of original RAE.
Local dinov3.yaml requires from_scratch=true for the current DiT integration;
changed input/latent shape is not compatible with directly swapping the released
policy encoder. Matched downstream adaptation is required for a policy claim.

## Proposed comparison sequence

A. Native Wan2.2 pixel VAE baseline.
B. Frozen DINOv3 high-dimensional latent path, using the same checkpoint as C.
   This tests the representation side of an RAE-style route. Add/validate a pixel
   decoder before claiming complete RAE reconstruction/generation reproduction.
C. The identical DINOv3 features with the native48-channel SVAE reducer.
D. Only after A–C are interpretable, treat control-aware compression as a separate
   matched-loss intervention. Preserve all existing six models and results.

First question: which route retains visual goal and short-horizon state information
under noise and camera shift? Second question: how much of the difference is
compression/readout capacity rather than the pretrained encoder? A–C does not
isolate pretraining alone; architectures/objectives differ. Use both native-width
and budget-controlled readout analyses; freeze normalization, training-only scales,
spatial/temporal alignment, output dimensionality controls and parameter/compute
accounting. Same frames/tasks/episode splits/noise draws. Frame replicas and codec
seeds do not increase the independent scene count. Camera-shift diagnostics need
same-camera trained controls. Waypoint and gripper transitions are not contact
or object-pose ground truth. Full policy claims need matched world/action training
and closed-loop evaluation, not just reconstruction quality or offline probes.

## Next concrete work and bounds

Check the cached raw-feature controls and available Wan encoder/checkpoints;
validate preprocessing and causal temporal alignment on training-only fixtures;
measure interface cost before fixing a runnable comparison protocol. These steps
resolve whether a fair baseline can run within the existing resource/time limits.
Do not start a new cohort, overwrite a freeze, extend a deadline, rerun old80policy
evaluations, or call an old exposed cohort an independent confirmation. Any new
experiment requires a separate prospective protocol and bounded idle resources.
No new GPU job, dependency installation, upstream message or PR has started from
this scope note. Existing public publishing failure remains unresolved.
