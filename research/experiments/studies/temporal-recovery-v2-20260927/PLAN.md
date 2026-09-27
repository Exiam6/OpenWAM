# Bounded recovery v2

{
  "created_at": "2026-09-27T12:24:22.608537-05:00",
  "authorization": "User explicitly requests prompt closed-loop recovery on2026-09-27; preserve old failed attempts and all prior deadlines.",
  "original_study": "/home/zifanz4/openwam-experiments/studies/temporal-20260926",
  "output": "/data01/zifanz4/openwam-experiments/temporal-recovery-v2-20260927",
  "deadline": "2026-09-29T17:48:49.941691-05:00",
  "scope": "All1080 temporal evaluations never previously launched; no replay of270 prior endtoend or old80.",
  "max_gpus": 2,
  "gpu_slots": {
    "mean": 4,
    "learned": 2
  },
  "disk_floor_gib": 32,
  "max_lane_seconds": 172800,
  "per_seed_timeout_seconds": 64800,
  "no_automatic_retry": true,
  "source_changes": [
    "two-line cache eviction fix in isolated source copy",
    "separate output/gate paths and alternate storage",
    "33real training-only WS prompts with exact post-eviction action check",
    "signal cleanup and runtime disk floor checks",
    "correct native checkpoint API argument binding"
  ],
  "unchanged": [
    "six final trained checkpoints",
    "paired seeds/scenes/conditions/action RNG/metrics",
    "original temporal deadline",
    "frozen original code and failed attempts"
  ],
  "gates": [
    "224original source hashes",
    "full paired training audit",
    "fresh cohort identity",
    "CPU20000cache operations",
    "3idle/ECC samples and GPU lock",
    "33distinct WS training prompts then exact reference action",
    "native3task deployment and training-only full rollout before heldout"
  ],
  "previous_failed_start": "/home/zifanz4/openwam-experiments/studies/temporal-recovery-20260927",
  "engineering_amendment": "Pass checkpoint parent directory plus explicit checkpoint_step_4000.safetensors filename; v1never loaded model or evaluated any scene. No model/scene/outcome changes."
}
