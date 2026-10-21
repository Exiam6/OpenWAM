"""Unlaunched native adapter candidate. No GPU imports at module import time."""
import os
import re

BLOCKED = {
    "ROBOTWIN_ENABLE_PLANNER_FALLBACK", "ROBOTWIN_SKIP_VERSION_CHECK",
    "ROBOTWIN_TEST_NUM", "ROBOTWIN_EPISODE_START", "ROBOTWIN_EPISODE_COUNT",
    "ROBOTWIN_TOTAL_EPISODES", "ROBOTWIN_EPISODE_MANIFEST", "ROBOTWIN_MANIFEST_HASH",
}

def guard(env, argv):
    assert not any(k in env for k in BLOCKED), "Unexpected evaluator override"
    assert not any(k.startswith("ROBOTWIN_LABTASKER_") for k in env), "Labtasker forbidden"
    assert len(argv) >= 2 and argv[0] == "--config", "Native config required"
    if len(argv) > 2:
        flags = ["--task_name", "--task_config", "--ckpt_setting", "--seed", "--policy_name"]
        assert argv[2] == "--overrides" and list(argv[3::2]) == flags and len(argv) == 13, "Unexpected native overrides"
        values = dict(zip(flags, argv[4::2]))
        assert re.fullmatch(r"[A-Za-z0-9_]+", values["--task_name"])
        assert values["--task_config"] in ["demo_clean", "demo_randomized"]
        assert values["--seed"] == "0" and values["--policy_name"] == "openwam2robotwin_interface"
    uuid = env.get("CUDA_VISIBLE_DEVICES", "")
    assert re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid), "One assigned GPU UUID required"
    assert env.get("OPENWAM_GPU_HOST"), "Dispatcher allocation required"
    assert env.get("OPENWAM_ALLOCATION_END"), "Bounded dispatcher allocation required"
    return uuid

def pci(value):
    """Normalize NVIDIA eight-digit domains and SAPIEN four-digit domains."""
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value or "")
    assert match, "Missing or malformed PCI identity"
    return tuple(int(x, 16) for x in match.groups())

def verify_device(device, expected_pci):
    assert device.is_cuda() and device.can_render() and not device.is_cpu(), "Non-CUDA renderer"
    assert device.cuda_id == 0, "Assigned visible CUDA ordinal must be zero"
    assert pci(device.pci_string) == pci(expected_pci), "Renderer outside assigned allocation"
    return {"cuda_id": device.cuda_id, "pci": device.pci_string, "name": device.name}

def native_render_factory(**kwargs):
    # The official prewarm function logs and swallows failures. Check CUDA again
    # before importing SAPIEN instead of treating its return as proof of success.
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    warm_tensor = torch.zeros(1, device="cuda")
    from sapien.render import RenderSystem
    return RenderSystem(**kwargs)

def delegate(wrapper, argv, env, expected_pci, render_factory=None, record=print):
    """Factory import must occur inside the hook, after native prewarm returns.

    This proves control flow only. Default native scenes and peer graphics
    contexts must be independently checked by a separately registered GPU probe.
    """
    guard(env, argv)
    render_factory = render_factory or native_render_factory
    original = wrapper._prewarm_cuda_for_curobo
    anchor = []
    def prewarm_then_bind():
        assert not anchor, "Unexpected repeated native prewarm"
        original()
        renderer = render_factory(device="cuda:0")
        details = verify_device(renderer.device, expected_pci)
        anchor.append(renderer)  # Keep explicit renderer alive throughout native main.
        record(details)
    wrapper._prewarm_cuda_for_curobo = prewarm_then_bind
    try:
        return wrapper.main(list(argv))
    finally:
        wrapper._prewarm_cuda_for_curobo = original
