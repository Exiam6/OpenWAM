"""Resolve native deploy arguments in the policy Python environment.

The Labtasker orchestration environment intentionally does not install Torch.
This helper runs under the policy interpreter and emits one JSON model spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.utils.eval_manifest import content_hash
from openwam.deploy.model_loader import _find_latest_checkpoint
from openwam.deploy.server import _build_argparser, merge_deploy_cfg, resolve_deploy_args


def resolve(argv: list[str]) -> dict:
    args = _build_argparser().parse_args(argv)
    cfg, directory = resolve_deploy_args(args)
    directory = Path(directory).expanduser().resolve()
    checkpoint = ((directory / args.ckpt_name).resolve() if args.ckpt_name
                  else Path(_find_latest_checkpoint(str(directory))))
    effective = OmegaConf.to_container(merge_deploy_cfg(OmegaConf.load(directory / "config.yaml"), cfg), resolve=False)
    if not isinstance(effective, dict):
        raise ValueError("effective policy configuration must be a mapping")
    for key in ("server", "device", "checkpoint_path"):
        effective.pop(key, None)
    model = {"effective_config": effective, "checkpoint_dir": str(directory), "checkpoint": str(checkpoint)}
    return model | {"identity": content_hash(model)}


if __name__ == "__main__":
    print(json.dumps(resolve(sys.argv[1:]), separators=(",", ":")))
