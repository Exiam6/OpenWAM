import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from research.rae import tensor_delta_checkpoint as delta


def audit(path: Path, output: Path) -> None:
    prefix, info = delta.header(path)
    result = {
        "header_size": len(prefix) - 8,
        "weights_sha256": delta.digest(path),
        "tensors": {},
    }
    with path.open("rb") as stream:
        for key, meta in delta.tensors(info, path.stat().st_size, len(prefix)):
            start, end = meta["data_offsets"]
            stream.seek(len(prefix) + start)
            result["tensors"][key] = {
                "bytes": end - start,
                "sha256": hashlib.sha256(stream.read(end - start)).hexdigest(),
            }
    output.write_text(json.dumps(result))


class DeltaCheckpointTest(unittest.TestCase):
    def test_exact_reconstruction_and_corruption(self):
        original_part_bytes = delta.PART_BYTES
        delta.PART_BYTES = 31
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / "base.safetensors"
                target = root / "target.safetensors"
                common = torch.arange(100, dtype=torch.float32)
                save_file({"a": common, "b": torch.zeros(50), "c": common + 1}, base)
                save_file({"a": common, "b": torch.ones(50), "c": common + 1,
                           "d": torch.arange(40, dtype=torch.float32)}, target)
                base_audit, target_audit = root / "base.json", root / "target.json"
                audit(base, base_audit)
                audit(target, target_audit)
                package = root / "package"
                plan = delta.pack(base, target, base_audit, target_audit, package)
                output = root / "copy.safetensors"
                delta.reconstruct(base, package, output)
                self.assertEqual(output.read_bytes(), target.read_bytes())
                self.assertEqual(plan["reused_bytes"], 800)
                self.assertGreater(len(list((package / "parts").glob("part*"))), 1)

                first = package / "parts" / "part000"
                changed = bytearray(first.read_bytes())
                changed[0] ^= 1
                first.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, "Delta stream SHA256 mismatch"):
                    delta.reconstruct(base, package, root / "corrupt.safetensors")
        finally:
            delta.PART_BYTES = original_part_bytes
