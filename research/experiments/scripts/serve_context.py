"""Fresh original policy for one context trial; never writes an older study."""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OpenWAM"))


def main():
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.server import build_server_from_config
    from safetensors.torch import load_file

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=18848)
    a = p.parse_args()
    assert not (a.out / "server-ready.json").exists()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cfg = OmegaConf.load(ROOT / "OpenWAM/configs/deploy.yaml")
    cfg.optimization.compile.enabled = False
    cfg.server.host = "127.0.0.1"
    cfg.server.port = a.port
    server = build_server_from_config(
        cfg, str(ROOT / "assets/dinov3-policy"), device="cuda:0"
    )
    try:
        server._init_policy()
        enc = server.engine.architecture.video_backbone.video_encoder
        weights = load_file(ROOT / "assets/dinov3-study/encoder.safetensors")
        prefix = "video_backbone.video_encoder."
        subset = {
            k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)
        }
        assert set(subset) == set(enc.state_dict())
        assert all(
            torch.equal(enc.state_dict()[k].cpu(), v.to(enc.state_dict()[k].dtype))
            for k, v in subset.items()
        )
        assert (
            not server._policy._async
            and server._policy._executor.inference_horizon is None
        )
        OmegaConf.save(server.cfg, a.out / "server-effective-config.yaml")
        info = {
            "model_initialized": True,
            "encoder_weights_exact": True,
            "sync_executor": True,
            "inference_horizon": None,
            "representation_hook": False,
            "torch_seed": 42,
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        }
        temp = a.out / "server-ready.json.tmp"
        temp.write_text(json.dumps(info, indent=2) + "\n")
        temp.rename(a.out / "server-ready.json")
        print("CONTEXT_SERVER_READY", flush=True)
        server.run(host="127.0.0.1", port=a.port)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
