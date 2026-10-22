"""CPU-testable native prewarm validation; no CUDA imports at module load."""
import contextlib
import io
from pathlib import Path

def native_environment(runtime, output):
    root = Path(runtime) / "ex8-a_officialrepro_1021"
    return {
        "PATH": f"{root}/bin:{runtime}/benchmark-env/bin:{runtime}/cuda-toolkit/bin:/usr/bin:/bin",
        "CUDA_HOME": f"{runtime}/cuda-toolkit",
        "CPATH": f"{runtime}/cuda-toolkit/targets/x86_64-linux/include",
        "TORCH_CUDA_ARCH_LIST": "12.0", "MAX_JOBS": "2",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "TORCH_EXTENSIONS_DIR": str(Path(output)/"cache/extensions"),
        "TMPDIR": str(Path(output)/"tmp"),
    }

def checked_prewarm(function):
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            function()
    finally:
        print(capture.getvalue(), end="", flush=True)
    text = capture.getvalue()
    if "CUDA/Curobo prewarm failed:" in text or "prewarmed CUDA/Curobo before SAPIEN import" not in text:
        raise RuntimeError("Native CUDA/Curobo prewarm did not succeed; no renderer test admitted")
    return text
