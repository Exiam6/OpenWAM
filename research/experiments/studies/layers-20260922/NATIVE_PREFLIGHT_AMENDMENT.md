# Native profile retry, before any optimizer step or policy result

Trial1 loaded the full native architecture and discovered105training episodes,
then failed before optimizer construction: published cfg.encoder.model_path
still points to /path/to/DINOv3, and startup deployment-asset export cannot find
config.json there. Exit1/log/config preserved. No training step or score exists.

Trial2 resolves only that sidecar path to the same checkpoint's dinov3 directory.
Original two-card pair is no longer available while fresh collection continues;
the new bounded feasibility test uses one healthy card, same globalbatch8 via
accumulation8,560microsteps=70optimizersteps,CPUoptimizer offload. This measures
ONE-card throughput, not the planned two-card value.20min timeout, no saved
weights or external experiment logging. If successful, any downstream allocation
must be explicitly reconciled with measured time before policy fitting. Do not
pretend the original two-card preflight passed. Source/checkpoint unchanged.
