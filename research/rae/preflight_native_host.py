#!/usr/bin/env python3
"""Check the host before submitting native RoboTwin evaluation jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROBOTWIN_COMMIT = "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
CUROBO_COMMIT = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
POLICY_PACKAGES = {
    "torch": "2.7.1",
    "omegaconf": None,
}
ORCHESTRATOR_PACKAGES = {"labtasker": "2.6.0", "omegaconf": None, "numpy": "2.5.3"}
ROBOTWIN_PACKAGES = {
    "torch": "2.4.1",
    "sapien": "3.0.0b1",
    "warp-lang": "1.13.0",
    "mplib": "0.2.1",
    "websockets": None,
}


def command(args: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout + "\n" + result.stderr).strip()


def version_matches(actual: str, expected: str | None) -> bool:
    return expected is None or actual == expected or actual.startswith(expected + "+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-python", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": passed, "detail": detail}

    for name in ("ROBOTWIN_SOFTWARE_RT", "ROBOTWIN_SOFTWARE_RASTER"):
        value = os.environ.get(name, "")
        record(f"no_{name.lower()}", not value or value == "0", value or "unset")
    icd = os.environ.get("VK_ICD_FILENAMES", "")
    record("no_software_vulkan_icd", not re.search(r"lavapipe|lvp_icd|swiftshader", icd, re.I), icd or "unset")

    gpu_status, gpu_text = command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpu_lines = [line.strip() for line in gpu_text.splitlines() if line.strip()]
    record("rtx_gpu", gpu_status == 0 and any(re.search(r"RTX|A6000", line, re.I) for line in gpu_lines), gpu_lines)

    # Older vulkan-tools builds have no --summary flag. The full report also
    # exposes the ray-tracing extension required by SAPIEN's native renderer.
    vulkan_status, vulkan_text = command(["vulkaninfo"])
    vulkan_devices = re.findall(r"deviceName\s*=\s*([^\n]+)", vulkan_text)
    has_nvidia_vulkan = any("NVIDIA" in name.upper() for name in vulkan_devices)
    record("nvidia_vulkan_device", has_nvidia_vulkan,
           {"devices": vulkan_devices, "exit_code": vulkan_status,
            "error": vulkan_text[-500:] if vulkan_status else ""})
    record("vulkan_ray_tracing", has_nvidia_vulkan and "VK_KHR_ray_tracing_pipeline" in vulkan_text,
           "VK_KHR_ray_tracing_pipeline" in vulkan_text)

    root = args.robotwin_root.resolve()
    commit_status, commit_text = command(["git", "rev-parse", "HEAD"], cwd=root)
    record("robotwin_commit", commit_status == 0 and commit_text == ROBOTWIN_COMMIT,
           {"actual": commit_text, "expected": ROBOTWIN_COMMIT})
    dirty_status, dirty_text = command(["git", "status", "--porcelain"], cwd=root)
    record("robotwin_clean", dirty_status == 0 and not dirty_text, dirty_text or "clean")
    record("robotwin_assets", (root / "assets").is_dir(), str(root / "assets"))

    # RoboTwin's installer clones cuRobo into an ignored, nested checkout and
    # patches mplib in site-packages. Neither change appears in RoboTwin's
    # Git status or the package-version list, but both affect native planning.
    curobo_root = root / "envs" / "curobo"
    curobo_status, curobo_text = command(["git", "rev-parse", "HEAD"], cwd=curobo_root)
    record("curobo_commit", curobo_status == 0 and curobo_text == CUROBO_COMMIT,
           {"actual": curobo_text, "expected": CUROBO_COMMIT})
    curobo_dirty_status, curobo_dirty_text = command(["git", "status", "--porcelain"], cwd=curobo_root)
    record("curobo_clean", curobo_dirty_status == 0 and not curobo_dirty_text,
           curobo_dirty_text or "clean")

    simulator_probe = (
        "import curobo,importlib.util,json,pathlib; "
        "spec=importlib.util.find_spec('mplib'); "
        "planner=pathlib.Path(spec.origin).parent/'planner.py' if spec else None; "
        "source=planner.read_text() if planner and planner.is_file() else ''; "
        "print(json.dumps({'curobo_file':str(pathlib.Path(curobo.__file__).resolve()),"
        "'planner_file':str(planner) if planner else '',"
        "'mplib_patch':'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:' in source}))"
    )
    probe_status, probe_text = command([str(args.robotwin_python.absolute()), "-c", simulator_probe])
    try:
        simulator_info = json.loads(probe_text) if probe_status == 0 else {}
    except json.JSONDecodeError:
        simulator_info = {}
    curobo_file = simulator_info.get("curobo_file", "")
    record("curobo_import", probe_status == 0 and
           curobo_file.startswith(str(curobo_root.resolve()) + os.sep),
           curobo_file or probe_text[-500:])
    record("mplib_planner_patch", simulator_info.get("mplib_patch") is True,
           simulator_info.get("planner_file", probe_text[-500:]))

    def probe_python(label: str, executable: Path, packages: dict[str, str | None],
                     python_version: str) -> None:
        executable = executable.absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            record(f"{label}_python", False, str(executable))
            for name in packages:
                record(f"{label}_{name}", False, "Python unavailable")
            return
        script = (
            "import importlib.metadata,json,sys; "
            f"names={list(packages)!r}; "
            "print(json.dumps({'python':sys.version.split()[0],"
            "'packages':{name:next((d.version for d in importlib.metadata.distributions() "
            "if d.metadata['Name'].lower().replace('_','-')==name.lower().replace('_','-')),'missing') "
            "for name in names}}))"
        )
        sim_status, sim_text = command([str(executable), "-c", script])
        try:
            sim_info = json.loads(sim_text) if sim_status == 0 else {}
        except json.JSONDecodeError:
            sim_info = {}
        record(f"{label}_python", sim_status == 0 and sim_info.get("python", "").startswith(python_version),
               sim_info.get("python", sim_text[-500:]))
        for name, expected in packages.items():
            actual = sim_info.get("packages", {}).get(name, "missing")
            record(f"{label}_{name}", actual != "missing" and version_matches(actual, expected),
                   {"actual": actual, "expected": expected})

    probe_python("orchestrator", Path(sys.executable), ORCHESTRATOR_PACKAGES, "3.12.")
    probe_python("policy", args.policy_python, POLICY_PACKAGES, "3.12.")
    probe_python("robotwin", args.robotwin_python, ROBOTWIN_PACKAGES, "3.10.")

    passed = all(check["passed"] for check in checks.values())
    result = {"passed": passed, "checks": checks}
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
