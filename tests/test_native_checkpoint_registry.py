"""Transferred checkpoints are registered only after byte and sidecar checks."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from research.rae.build_native_checkpoint_registry import build_registry
from research.rae.run_native_eval import OFFICIAL_SHA256


class NativeCheckpointRegistryTests(unittest.TestCase):
    def test_partial_transfer_registers_only_verified_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "seed42_vae"
            directory.mkdir()
            weight = directory / "checkpoint_step_1000.safetensors"
            weight.write_bytes(b"fixture weights")
            (directory / "config.yaml").write_text("model: fixture\n")
            (directory / "normalization_stats.npy").write_bytes(b"stats")
            (directory / "tokenizer").mkdir()
            manifest = {
                "checkpoints": {"seed42_vae": {
                    "weights_file": weight.name,
                    "weights_bytes": weight.stat().st_size,
                    "registered_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
                    "required_sidecars": ["config.yaml", "normalization_stats.npy", "tokenizer"],
                }},
                "official_checkpoint": {"file": "/source/checkpoint_step_118655.safetensors",
                                        "bytes": 1, "registered_sha256": OFFICIAL_SHA256,
                                        "required_sidecars": ["config.yaml"]},
            }
            registry, missing = build_registry(manifest, root)
            self.assertEqual(list(registry), ["seed42_vae"])
            self.assertIn("official", missing)
            self.assertEqual(registry["seed42_vae"]["directory"], str(directory.resolve()))
            weight.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                build_registry(manifest, root)


if __name__ == "__main__":
    unittest.main()
