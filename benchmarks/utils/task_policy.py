"""Native OpenWAM configuration snapshots and one lazily loaded policy server."""

import argparse
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

import labtasker
from omegaconf import OmegaConf

from benchmarks.utils.client import ServerError
from benchmarks.utils.eval_manifest import content_hash
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource
from benchmarks.utils.transport import WSPolicyClient

ROOT = Path(__file__).resolve().parents[2]


def resolve_policy_model_spec(argv: list[str], *, policy_python: Path | None = None) -> dict[str, Any]:
    """Resolve native deploy arguments once so Workers agree on config and latest.

    Identity covers effective parameters and concrete paths, not file contents.
    Users keep model files fixed; Workers do not reread weights to check reuse.
    """
    if policy_python is not None:
        command = [str(policy_python), "-m", "benchmarks.robotwin.resolve_policy_model_spec", *argv]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise ValueError(f"policy config resolver exited {completed.returncode}: {completed.stderr.strip()}")
        model = json.loads(completed.stdout)
        if not isinstance(model, dict) or model.get("identity") != content_hash(
            {key: model[key] for key in ("effective_config", "checkpoint_dir", "checkpoint")}
        ):
            raise ValueError("policy interpreter returned an invalid model specification")
        return model
    from benchmarks.robotwin.resolve_policy_model_spec import resolve

    return resolve(argv)


def load_policy_config(path: str | Path) -> dict[str, Any]:
    """Embed benchmark settings in Task args instead of a mutable config path."""
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("benchmark policy config must be a mapping")
    return value


def write_policy_config(value: dict[str, Any], directory: Path) -> Path:
    """Write the Task's settings for evaluators that accept a YAML file path."""
    import yaml

    path = Path(directory) / "policy.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=True))
    return path


class WorkerPolicyServer:
    """Own one policy endpoint per Worker and reuse it across episode batches."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.owned = ProcessRegistry()
        self.process: subprocess.Popen | None = None
        self.identity: str | None = None

    def close(self) -> None:
        self.identity = None
        process, self.process = self.process, None
        try:
            self.owned.terminate_all()
            if process is not None and process.poll() is None:
                raise RuntimeError("policy process did not exit")
        except Exception as exc:
            raise labtasker.FatalWorkerError("cannot safely clean up policy server") from exc

    def load_or_reuse(self, model: dict[str, Any]) -> None:
        """Keep a matching live model; otherwise stop it before loading the next."""
        if self.identity == model["identity"] and self.process is not None and self.process.poll() is None:
            return
        self.close()
        args = self.args
        directory = args.output_dir / "policy" / f"{os.getpid()}-{time.time_ns()}"
        directory.mkdir(parents=True, exist_ok=True)
        config = directory / "deploy.yml"
        OmegaConf.save(OmegaConf.create(model["effective_config"]), config)
        checkpoint = Path(model["checkpoint"])
        readiness_token = secrets.token_hex(24)
        command = [
            str(args.server_python),
            str(ROOT / "scripts/deploy.py"),
            "--config",
            str(config),
            "--ckpt-dir",
            model["checkpoint_dir"],
            "--ckpt-name",
            str(checkpoint),
            "--device",
            "cuda:0",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--readiness-token",
            readiness_token,
        ]
        log = (directory / "server.log").open("w")
        try:
            self.process = spawn_resource(
                command, cwd=ROOT, stdout=log, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            )
            self.owned.add(self.process, log)
            deadline = time.monotonic() + args.server_start_timeout
            from websockets.exceptions import WebSocketException

            while True:
                if self.process.poll() is not None:
                    self.close()
                    raise RuntimeError("policy server exited")
                if labtasker.cancellation_requested():
                    raise RuntimeError("Task cancelled during model loading")
                try:
                    with WSPolicyClient(
                        f"ws://{args.host}:{args.port}", timeout=1, open_timeout=1
                    ) as probe:
                        response = probe.ping()
                    if not isinstance(response, dict) or response.get("type") != "pong":
                        raise RuntimeError(
                            f"cannot use policy endpoint {args.host}:{args.port}: incompatible response"
                        )
                    if response.get("readiness_token") != readiness_token:
                        raise RuntimeError(f"policy port {args.host}:{args.port} is owned by another server")
                    break
                except ConnectionRefusedError:
                    pass
                except (OSError, WebSocketException, ValueError, ServerError) as exc:
                    raise RuntimeError(
                        f"cannot use policy endpoint {args.host}:{args.port}: occupied, incompatible, or unreachable"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"policy startup timed out; see {log.name}")
                time.sleep(0.2)
            self.identity = model["identity"]
        except BaseException:
            log.close()
            self.close()
            raise
